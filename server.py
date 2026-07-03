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

import mediapipe as mp
import numpy as np
from aiohttp import web
from PIL import Image, ImageOps, UnidentifiedImageError
from mediapipe.tasks.python.components.containers import rect as rect_module

LOG = logging.getLogger("mediapipe-api")


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


def _make_image_processing_options(config: dict[str, Any]) -> Any | None:
    img_opts = config.get("image_processing_options") or {}
    if not img_opts:
        return None

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

        roi = rect_module.RectF(left=left, top=top, right=right, bottom=bottom)

    return mp.tasks.vision.ImageProcessingOptions(region_of_interest=roi, rotation_degrees=rotation_degrees)


def _decode_image(image_bytes: bytes) -> tuple[Any, dict[str, int]]:
    if not image_bytes:
        raise ApiError(400, "image part is empty")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.ascontiguousarray(np.asarray(image))
    except UnidentifiedImageError as exc:
        raise ApiError(400, "image part is not a supported image file") from exc

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=array), {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def _serialize_result(result: Any) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for detection in result.detections:
        box = detection.bounding_box
        detections.append({
            "bounding_box": {
                "origin_x": box.origin_x,
                "origin_y": box.origin_y,
                "width": box.width,
                "height": box.height,
            },
            "categories": [
                {
                    "index": category.index,
                    "score": category.score,
                    "display_name": category.display_name,
                    "category_name": category.category_name,
                }
                for category in detection.categories
            ],
            "keypoints": [
                {
                    "x": keypoint.x,
                    "y": keypoint.y,
                    "label": keypoint.label,
                    "score": keypoint.score,
                }
                for keypoint in detection.keypoints
            ] if detection.keypoints else None,
        })
    return {"detections": detections, "detection_count": len(detections)}


class MediaPipeWorker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._jobs: queue.Queue[tuple[Callable[[], Any] | None, concurrent.futures.Future | None]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="mediapipe-worker", daemon=True)
        self._closed = False
        self._detectors: dict[DetectorKey, Any] = {}
        self._thread.start()

    def submit(self, func: Callable[[], Any]) -> concurrent.futures.Future:
        if self._closed:
            raise ApiError(503, "MediaPipe worker is closed")
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

        for detector in self._detectors.values():
            try:
                detector.close()
            except Exception:
                LOG.exception("failed to close MediaPipe detector")
        self._detectors.clear()

    def _create_detector(self, key: DetectorKey) -> Any:
        base_kwargs: dict[str, Any] = {"model_asset_path": self.model_path}
        if key.delegate is not None:
            base_kwargs["delegate"] = getattr(mp.tasks.BaseOptions.Delegate, key.delegate)

        # Work around MediaPipe ctypes binding issue: c_char_p needs bytes, not str.
        display_names_locale = (
            key.display_names_locale.encode("utf-8")
            if key.display_names_locale is not None
            else None
        )

        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(**base_kwargs),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            display_names_locale=display_names_locale,
            max_results=key.max_results,
            score_threshold=key.score_threshold,
            category_allowlist=list(key.category_allowlist) or None,
            category_denylist=list(key.category_denylist) or None,
        )
        return mp.tasks.vision.ObjectDetector.create_from_options(options)

    def _get_detector(self, key: DetectorKey) -> Any:
        detector = self._detectors.get(key)
        if detector is None:
            detector = self._create_detector(key)
            self._detectors[key] = detector
        return detector

    def _detect(self, image_bytes: bytes, config: dict[str, Any]) -> dict[str, Any]:
        key = _coerce_detector_key(config)
        image_processing_options = _make_image_processing_options(config)
        mp_image, image_meta = _decode_image(image_bytes)
        detector = self._get_detector(key)

        result = detector.detect(mp_image, image_processing_options=image_processing_options)
        return {"ok": True, "image": image_meta, "result": _serialize_result(result)}


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
    app["worker"] = MediaPipeWorker(model_path)
    app["debug_errors"] = debug_errors

    app.router.add_get("/health", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_post("/v1/detect", detect_handler)
    app.on_cleanup.append(on_cleanup)
    return app


def _prepare_unix_socket(path: str) -> None:
    socket_path = Path(path).expanduser()
    if socket_path.exists():
        if not socket_path.is_socket():
            raise FileExistsError(f"unix socket path exists and is not a socket: {socket_path}")
        socket_path.unlink()
    socket_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "efficientdet_lite0.tflite"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--unix-socket",
        default=os.environ.get("UNIX_SOCKET"),
        help="Serve HTTP over this Unix domain socket path instead of host/port",
    )
    parser.add_argument("--debug-errors", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = create_app(args.model, args.debug_errors)
    if args.unix_socket:
        _prepare_unix_socket(args.unix_socket)
        web.run_app(app, path=str(Path(args.unix_socket).expanduser()))
    else:
        web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
