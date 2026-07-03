# MediaPipe EfficientDet Lite0 HTTP API

This project exposes MediaPipe Tasks `ObjectDetector` with `efficientdet_lite0.tflite` through an `aiohttp` HTTP server.

The client sends both image bytes and detector configuration as `multipart/form-data`. MediaPipe processing runs on a dedicated worker thread so the `aiohttp` event loop is not blocked by inference.

## Files

| File | Purpose |
|---|---|
| `server.py` | `aiohttp` server wrapping MediaPipe ObjectDetector |
| `client.py` | Multipart client example |
| `requirements.txt` | Python dependencies |
| `efficientdet_lite0.tflite` | Model file, supplied separately |

## Requirements

Recommended Python version:

```bash
python 3.10 - 3.12
```

MediaPipe may not be reliable on unsupported newer Python versions. If you hit binding errors, recreate the virtual environment with Python 3.12.

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the server

```bash
python server.py --model ./efficientdet_lite0.tflite --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

or:

```bash
curl http://127.0.0.1:8080/healthz
```

Expected response:

```json
{"ok": true}
```

## API

### `POST /v1/detect`

Runs object detection.

Request content type:

```text
multipart/form-data
```

Multipart fields:

| Name | Type | Required | Description |
|---|---|---:|---|
| `image` | file | yes | Image file such as PNG or JPEG |
| `config` | JSON string | no | Detector and image-processing options |

### Basic curl example

```bash
curl -s \
  -F 'config={"running_mode":"IMAGE","object_detector_options":{"score_threshold":0.25,"max_results":5}};type=application/json' \
  -F 'image=@sample.jpg;type=image/jpeg' \
  http://127.0.0.1:8080/v1/detect
```

### Python client example

```bash
python client.py --mode IMAGE --score-threshold 0.25 --max-results 5 ./sample.jpg
```

## Configuration

Example `config` JSON:

```json
{
  "running_mode": "IMAGE",
  "stream_id": "camera-1",
  "timestamp_ms": 0,
  "wait_for_result": true,
  "live_stream_timeout_s": 2.0,
  "object_detector_options": {
    "delegate": "CPU",
    "display_names_locale": "en",
    "max_results": 5,
    "score_threshold": 0.25,
    "category_allowlist": ["person", "dog"],
    "category_denylist": null
  },
  "image_processing_options": {
    "rotation_degrees": 0,
    "region_of_interest": {
      "left": 0.0,
      "top": 0.0,
      "right": 1.0,
      "bottom": 1.0
    }
  }
}
```

### Supported running modes

| Mode | Description |
|---|---|
| `IMAGE` | Single-image detection |
| `VIDEO` | Video-frame detection with monotonically increasing `timestamp_ms` |
| `LIVE_STREAM` | Async live-stream detection using MediaPipe callback |

For `VIDEO`, `timestamp_ms` is required and must increase for each `stream_id`.

For `LIVE_STREAM`, `timestamp_ms` is recommended. If omitted, the server uses a monotonic timestamp.

## Response format

Example response:

```json
{
  "ok": true,
  "running_mode": "IMAGE",
  "stream_id": "image",
  "image": {
    "width": 1280,
    "height": 720
  },
  "result": {
    "detection_count": 1,
    "detections": [
      {
        "bounding_box": {
          "origin_x": 100,
          "origin_y": 80,
          "width": 200,
          "height": 300
        },
        "categories": [
          {
            "index": 0,
            "score": 0.91,
            "display_name": null,
            "category_name": "person"
          }
        ],
        "keypoints": null
      }
    ]
  }
}
```

## LIVE_STREAM mode

Send and wait for the callback result:

```bash
python client.py --mode LIVE_STREAM --stream-id camera-1 --timestamp-ms 1000 ./frame.jpg
```

To submit without waiting, send:

```json
{
  "running_mode": "LIVE_STREAM",
  "stream_id": "camera-1",
  "timestamp_ms": 1000,
  "wait_for_result": false,
  "object_detector_options": {
    "score_threshold": 0.25,
    "max_results": 5
  }
}
```

Then poll:

```bash
curl http://127.0.0.1:8080/v1/live-jobs/<job_id>
```

## Reset a stream

Use this when a camera or video stream restarts and timestamps should reset.

```bash
curl -X DELETE http://127.0.0.1:8080/v1/streams/camera-1
```

## Notes

- Do not send both `category_allowlist` and `category_denylist`.
- `display_names_locale` is accepted as a JSON string; the server converts it to bytes internally for the current MediaPipe ctypes binding.
- The server caches detector instances by running mode, stream ID, and detector options.
- MediaPipe inference runs in a separate worker thread. HTTP request parsing and response handling remain asynchronous.
