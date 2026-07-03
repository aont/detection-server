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
import time
import traceback
import uuid
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
    running_mode: str
    stream_id: str
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
    mode = str(config.get("running_mode", "IMAGE")).upper()
    if mode not in {"IMAGE", "VIDEO", "LIVE_STREAM"}:
        raise ApiError(400, "running_mode must be IMAGE, VIDEO, or LIVE_STREAM")

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

    stream_id = str(config.get("stream_id", "default" if mode != "IMAGE" else "image"))

    display_names_locale = opts.get("display_names_locale")
    if display_names_locale is not None:
        display_names_locale = str(display_names_locale)

    return DetectorKey(
        running_mode=mode,
        stream_id=stream_id,
        delegate=delegate,
        display_names_locale=display_names_locale,
        max_results=max_results,
        score_threshold=score_threshold,
        category_allowlist=allowlist,
        category_denylist=denylist,
    )


def _coerce_timestamp_ms(config: dict[str, Any], mode: str) -> int | None:
    if mode == "IMAGE":
        return None
    if "timestamp_ms" not in config or config.get("timestamp_ms") is None:
        if mode == "LIVE_STREAM":
            return time.monotonic_ns() // 1_000_000
        raise ApiError(400, "timestamp_ms is required when running_mode is VIDEO")
    return int(config["timestamp_ms"])


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
        self._live_lock = threading.Lock()
        self._live_pending: dict[tuple[DetectorKey, int], concurrent.futures.Future] = {}
        self._live_job_by_key_ts: dict[tuple[DetectorKey, int], str] = {}
        self._live_jobs: dict[str, dict[str, Any]] = {}
        self._thread.start()

    def submit(self, func: Callable[[], Any]) -> concurrent.futures.Future:
        if self._closed:
            raise ApiError(503, "MediaPipe worker is closed")
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((func, future))
        return future

    def detect(self, image_bytes: bytes, config: dict[str, Any]) -> concurrent.futures.Future:
        return self.submit(lambda: self._detect(image_bytes, config))

    def get_live_job(self, job_id: str) -> concurrent.futures.Future:
        return self.submit(lambda: self._get_live_job(job_id))

    def reset_stream(self, stream_id: str | None = None) -> concurrent.futures.Future:
        return self.submit(lambda: self._reset_stream(stream_id))

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

    @staticmethod
    def _running_mode_enum(mode: str) -> Any:
        return {
            "IMAGE": mp.tasks.vision.RunningMode.IMAGE,
            "VIDEO": mp.tasks.vision.RunningMode.VIDEO,
            "LIVE_STREAM": mp.tasks.vision.RunningMode.LIVE_STREAM,
        }[mode]

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

        options_kwargs: dict[str, Any] = {
            "base_options": mp.tasks.BaseOptions(**base_kwargs),
            "running_mode": self._running_mode_enum(key.running_mode),
            "display_names_locale": display_names_locale,
            "max_results": key.max_results,
            "score_threshold": key.score_threshold,
            "category_allowlist": list(key.category_allowlist) or None,
            "category_denylist": list(key.category_denylist) or None,
        }

        if key.running_mode == "LIVE_STREAM":
            options_kwargs["result_callback"] = self._make_live_callback(key)

        options = mp.tasks.vision.ObjectDetectorOptions(**options_kwargs)
        return mp.tasks.vision.ObjectDetector.create_from_options(options)

    def _get_detector(self, key: DetectorKey) -> Any:
        detector = self._detectors.get(key)
        if detector is None:
            detector = self._create_detector(key)
            self._detectors[key] = detector
        return detector

    def _make_live_callback(self, key: DetectorKey) -> Callable[[Any, Any, int], None]:
        def callback(result: Any, output_image: Any, timestamp_ms: int) -> None:
            del output_image
            payload = _serialize_result(result)
            with self._live_lock:
                pending_key = (key, int(timestamp_ms))
                future = self._live_pending.pop(pending_key, None)
                job_id = self._live_job_by_key_ts.pop(pending_key, None)
                if job_id:
                    self._live_jobs[job_id] = {
                        "ok": True,
                        "status": "done",
                        "running_mode": "LIVE_STREAM",
                        "stream_id": key.stream_id,
                        "timestamp_ms": int(timestamp_ms),
                        "result": payload,
                    }
            if future is not None and not future.done():
                future.set_result(payload)
        return callback

    def _detect(self, image_bytes: bytes, config: dict[str, Any]) -> dict[str, Any]:
        key = _coerce_detector_key(config)
        timestamp_ms = _coerce_timestamp_ms(config, key.running_mode)
        image_processing_options = _make_image_processing_options(config)
        mp_image, image_meta = _decode_image(image_bytes)
        detector = self._get_detector(key)

        if key.running_mode == "IMAGE":
            result = detector.detect(mp_image, image_processing_options=image_processing_options)
            return {"ok": True, "running_mode": key.running_mode, "stream_id": key.stream_id, "image": image_meta, "result": _serialize_result(result)}

        if key.running_mode == "VIDEO":
            assert timestamp_ms is not None
            result = detector.detect_for_video(mp_image, timestamp_ms, image_processing_options=image_processing_options)
            return {"ok": True, "running_mode": key.running_mode, "stream_id": key.stream_id, "timestamp_ms": timestamp_ms, "image": image_meta, "result": _serialize_result(result)}

        assert key.running_mode == "LIVE_STREAM"
        assert timestamp_ms is not None
        wait_for_result = bool(config.get("wait_for_result", True))
        timeout_s = float(config.get("live_stream_timeout_s", 2.0))
        job_id = uuid.uuid4().hex
        live_future: concurrent.futures.Future = concurrent.futures.Future()
        pending_key = (key, timestamp_ms)

        with self._live_lock:
            self._live_pending[pending_key] = live_future
            self._live_job_by_key_ts[pending_key] = job_id
            self._live_jobs[job_id] = {
                "ok": True,
                "status": "pending",
                "running_mode": "LIVE_STREAM",
                "stream_id": key.stream_id,
                "timestamp_ms": timestamp_ms,
                "result": None,
            }

        detector.detect_async(mp_image, timestamp_ms, image_processing_options=image_processing_options)

        if not wait_for_result:
            return {"ok": True, "status": "accepted", "running_mode": key.running_mode, "stream_id": key.stream_id, "timestamp_ms": timestamp_ms, "image": image_meta, "job_id": job_id, "result_url": f"/v1/live-jobs/{job_id}"}

        try:
            result_payload = live_future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return {"ok": True, "status": "pending_or_dropped", "running_mode": key.running_mode, "stream_id": key.stream_id, "timestamp_ms": timestamp_ms, "image": image_meta, "job_id": job_id, "result_url": f"/v1/live-jobs/{job_id}", "result": None, "warning": "No live-stream callback arrived before live_stream_timeout_s. MediaPipe can drop live-stream frames when busy."}

        return {"ok": True, "status": "done", "running_mode": key.running_mode, "stream_id": key.stream_id, "timestamp_ms": timestamp_ms, "image": image_meta, "job_id": job_id, "result": result_payload}

    def _get_live_job(self, job_id: str) -> dict[str, Any]:
        with self._live_lock:
            job = self._live_jobs.get(job_id)
            if job is None:
                raise ApiError(404, "live job not found")
            return dict(job)

    def _reset_stream(self, stream_id: str | None) -> dict[str, Any]:
        keys = [key for key in self._detectors if stream_id is None or key.stream_id == stream_id]
        for key in keys:
            detector = self._detectors.pop(key)
            detector.close()
        return {"ok": True, "closed_detectors": len(keys), "stream_id": stream_id}


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


async def live_job_handler(request: web.Request) -> web.Response:
    try:
        future = request.app["worker"].get_live_job(request.match_info["job_id"])
        return web.json_response(await asyncio.wrap_future(future))
    except ApiError as exc:
        return _json_error(exc.status, exc.message)


async def reset_stream_handler(request: web.Request) -> web.Response:
    try:
        stream_id = request.match_info.get("stream_id")
        future = request.app["worker"].reset_stream(stream_id)
        return web.json_response(await asyncio.wrap_future(future))
    except ApiError as exc:
        return _json_error(exc.status, exc.message)


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
    app.router.add_get("/v1/live-jobs/{job_id}", live_job_handler)
    app.router.add_delete("/v1/streams/{stream_id}", reset_stream_handler)
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
