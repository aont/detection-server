from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import os
import queue
import threading
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from aiohttp import web
from PIL import Image, ImageOps, UnidentifiedImageError
from tflite_runtime.interpreter import Interpreter

LOG = logging.getLogger("tflite-api")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclasses.dataclass(frozen=True)
class DetectorKey:
    delegate: str | None
    display_names_locale: str | None
    max_results: int
    score_threshold: float
    category_allowlist: tuple[str, ...]
    category_denylist: tuple[str, ...]


def _parse_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise ApiError(400, f"{field_name} must be a list of strings or a comma-separated string")


def _coerce_detector_key(config: dict[str, Any]) -> DetectorKey:
    if "running_mode" in config:
        raise ApiError(400, "running_mode is no longer supported")

    opts = dict(config.get("object_detector_options") or config.get("detector_options") or {})

    for name in (
        "delegate",
        "display_names_locale",
        "max_results",
        "score_threshold",
        "category_allowlist",
        "category_denylist",
    ):
        if name in config and name not in opts:
            opts[name] = config[name]

    delegate = opts.get("delegate")
    if delegate is not None:
        delegate = str(delegate).upper()
        if delegate not in {"CPU", "GPU"}:
            raise ApiError(400, "object_detector_options.delegate must be CPU or GPU")

    max_results = int(opts.get("max_results", -1))
    if max_results == 0 or max_results < -1:
        raise ApiError(400, "object_detector_options.max_results must be -1 or a positive integer")

    score_threshold = float(opts.get("score_threshold", 0.0))
    if not 0.0 <= score_threshold <= 1.0:
        raise ApiError(400, "object_detector_options.score_threshold must be between 0.0 and 1.0")

    allowlist = _parse_string_list(opts.get("category_allowlist"), "object_detector_options.category_allowlist")
    denylist = _parse_string_list(opts.get("category_denylist"), "object_detector_options.category_denylist")
    if allowlist and denylist:
        raise ApiError(400, "category_allowlist and category_denylist are mutually exclusive")

    display_names_locale = opts.get("display_names_locale")
    if display_names_locale is not None:
        display_names_locale = str(display_names_locale)

    return DetectorKey(
        delegate=delegate,
        display_names_locale=display_names_locale,
        max_results=max_results,
        score_threshold=score_threshold,
        category_allowlist=allowlist,
        category_denylist=denylist,
    )


def _get_image_transform(config: dict[str, Any]) -> tuple[int, tuple[float, float, float, float] | None]:
    img_opts = config.get("image_processing_options") or {}
    if not img_opts:
        return 0, None

    rotation_degrees = int(img_opts.get("rotation_degrees", 0))
    if rotation_degrees % 90 != 0:
        raise ApiError(400, "image_processing_options.rotation_degrees must be a multiple of 90")

    roi_config = img_opts.get("region_of_interest") or img_opts.get("roi")
    roi = None

    if roi_config is not None:
        try:
            left = float(roi_config["left"])
            top = float(roi_config["top"])
            right = float(roi_config["right"])
            bottom = float(roi_config["bottom"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(400, "region_of_interest must contain left, top, right, bottom") from exc

        if not (left < right and top < bottom):
            raise ApiError(400, "region_of_interest must satisfy left < right and top < bottom")
        if not all(0.0 <= value <= 1.0 for value in (left, top, right, bottom)):
            raise ApiError(400, "region_of_interest coordinates must be normalized values in [0, 1]")
        roi = (left, top, right, bottom)

    return rotation_degrees % 360, roi


def _decode_image(image_bytes: bytes) -> tuple[np.ndarray, dict[str, int]]:
    if not image_bytes:
        raise ApiError(400, "image part is empty")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.ascontiguousarray(np.asarray(image))
    except UnidentifiedImageError as exc:
        raise ApiError(400, "image part is not a supported image file") from exc

    return array, {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def _resize_for_input(image: np.ndarray, input_detail: dict[str, Any]) -> np.ndarray:
    _, height, width, channels = input_detail["shape"]
    if channels != 3:
        raise ApiError(500, "model input must have 3 color channels")

    resized = Image.fromarray(image).resize((int(width), int(height)), Image.Resampling.BILINEAR)
    input_data = np.asarray(resized)

    if input_detail["dtype"] == np.float32:
        quant = input_detail.get("quantization", (0.0, 0))
        scale, zero_point = quant
        input_data = input_data.astype(np.float32)
        if scale and scale > 0:
            input_data = (input_data - float(zero_point)) * float(scale)
        else:
            input_data = (input_data - 127.5) / 127.5
    else:
        input_data = input_data.astype(input_detail["dtype"])

    return np.expand_dims(input_data, axis=0)


def _apply_image_transform(
    image: np.ndarray,
    rotation_degrees: int,
    roi: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if rotation_degrees:
        image = np.rot90(image, k=(-rotation_degrees // 90) % 4)
    if roi is None:
        return image

    height, width = image.shape[:2]
    left, top, right, bottom = roi
    x1 = int(round(left * width))
    y1 = int(round(top * height))
    x2 = int(round(right * width))
    y2 = int(round(bottom * height))
    return np.ascontiguousarray(image[y1:y2, x1:x2])


class TFLiteWorker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._jobs: queue.Queue[tuple[Callable[[], Any] | None, concurrent.futures.Future | None]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="tflite-worker", daemon=True)
        self._closed = False
        self._detectors: dict[DetectorKey, Any] = {}
        self._thread.start()

    def submit(self, func: Callable[[], Any]) -> concurrent.futures.Future:
        if self._closed:
            raise ApiError(503, "TFLite worker is closed")
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((func, future))
        return future

    def detect(self, image_bytes: bytes, config: dict[str, Any]) -> concurrent.futures.Future:
        return self.submit(lambda: self._detect(image_bytes, config))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._jobs.put((None, None))
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while True:
            func, future = self._jobs.get()
            if func is None:
                break
            assert future is not None
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(func())
            except BaseException as exc:
                future.set_exception(exc)

        self._detectors.clear()

    def _create_detector(self, key: DetectorKey) -> Any:
        if key.delegate == "GPU":
            raise ApiError(400, "object_detector_options.delegate=GPU is not supported by tflite-runtime")
        if key.display_names_locale is not None:
            raise ApiError(400, "object_detector_options.display_names_locale is not supported by tflite-runtime")

        interpreter = Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter

    def _get_detector(self, key: DetectorKey) -> Any:
        detector = self._detectors.get(key)
        if detector is None:
            detector = self._create_detector(key)
            self._detectors[key] = detector
        return detector

    def _detect(self, image_bytes: bytes, config: dict[str, Any]) -> dict[str, Any]:
        key = _coerce_detector_key(config)
        rotation_degrees, roi = _get_image_transform(config)
        image, image_meta = _decode_image(image_bytes)
        image = _apply_image_transform(image, rotation_degrees, roi)
        interpreter = self._get_detector(key)

        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_detail["index"], _resize_for_input(image, input_detail))
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]
        detections = _serialize_tflite_outputs(outputs, image.shape[1], image.shape[0], key)
        return {
            "ok": True,
            "image": image_meta,
            "result": {"detections": detections, "detection_count": len(detections)},
        }


def _serialize_tflite_outputs(
    outputs: list[np.ndarray],
    image_width: int,
    image_height: int,
    key: DetectorKey,
) -> list[dict[str, Any]]:
    if len(outputs) < 4:
        raise ApiError(500, "model must expose TFLite Detection PostProcess outputs")

    boxes = np.squeeze(outputs[0])
    classes = np.squeeze(outputs[1])
    scores = np.squeeze(outputs[2])
    count = int(np.squeeze(outputs[3]))

    max_results = count if key.max_results == -1 else min(count, key.max_results)
    detections: list[dict[str, Any]] = []
    for i in range(max_results):
        score = float(scores[i])
        if score < key.score_threshold:
            continue

        class_index = int(classes[i])
        category_name = str(class_index)
        if key.category_allowlist and category_name not in key.category_allowlist:
            continue
        if key.category_denylist and category_name in key.category_denylist:
            continue

        ymin, xmin, ymax, xmax = [float(value) for value in boxes[i]]
        origin_x = max(0, int(round(xmin * image_width)))
        origin_y = max(0, int(round(ymin * image_height)))
        width = max(0, int(round((xmax - xmin) * image_width)))
        height = max(0, int(round((ymax - ymin) * image_height)))

        detections.append({
            "bounding_box": {
                "origin_x": origin_x,
                "origin_y": origin_y,
                "width": width,
                "height": height,
            },
            "categories": [{
                "index": class_index,
                "score": score,
                "display_name": None,
                "category_name": category_name,
            }],
            "keypoints": None,
        })
    return detections


async def _parse_multipart_request(request: web.Request) -> tuple[bytes, dict[str, Any]]:
    if not request.content_type.startswith("multipart/"):
        raise ApiError(415, "Content-Type must be multipart/form-data")

    reader = await request.multipart()
    image_bytes: bytes | None = None
    config: dict[str, Any] = {}

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "image":
            image_bytes = await part.read(decode=False)
        elif part.name == "config":
            text = await part.text()
            try:
                loaded = json.loads(text) if text else {}
            except json.JSONDecodeError as exc:
                raise ApiError(400, "config part must contain valid JSON") from exc
            if not isinstance(loaded, dict):
                raise ApiError(400, "config JSON must be an object")
            config = loaded

    if image_bytes is None:
        raise ApiError(400, "multipart request must contain an image part")
    return image_bytes, config


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"ok": False, "error": {"message": message}}, status=status)


async def detect_handler(request: web.Request) -> web.Response:
    try:
        image_bytes, config = await _parse_multipart_request(request)
        future = request.app["worker"].detect(image_bytes, config)
        result = await asyncio.wrap_future(future)
        return web.json_response(result)
    except ApiError as exc:
        return _json_error(exc.status, exc.message)
    except ValueError as exc:
        return _json_error(400, str(exc))
    except Exception as exc:
        LOG.exception("unhandled detection error")
        if request.app.get("debug_errors"):
            return web.json_response({"ok": False, "error": {"message": str(exc), "traceback": traceback.format_exc()}}, status=500)
        return _json_error(500, "internal detection error")



async def health_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response({"ok": True})


async def on_cleanup(app: web.Application) -> None:
    app["worker"].close()


def create_app(model_path: str, debug_errors: bool = False) -> web.Application:
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["worker"] = TFLiteWorker(model_path)
    app["debug_errors"] = debug_errors

    app.router.add_get("/health", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_post("/v1/detect", detect_handler)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "efficientdet_lite0.tflite"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(args.model, args.debug_errors), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
