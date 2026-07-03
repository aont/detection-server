# TFLite EfficientDet Lite0 HTTP API

This project exposes a `tflite-runtime` based object detector for `efficientdet_lite0.tflite` through an `aiohttp` HTTP server.

The client sends both image bytes and detector configuration as `multipart/form-data`. TFLite inference runs on a dedicated worker thread so the `aiohttp` event loop is not blocked by inference.

## Files

| File | Purpose |
|---|---|
| `server.py` | `aiohttp` server wrapping a tflite-runtime Interpreter |
| `client.py` | Multipart client example |
| `requirements.txt` | Python dependencies |
| `efficientdet_lite0.tflite` | Model file, supplied separately |

## Requirements

Recommended Python version depends on the `tflite-runtime` wheels available for your platform.

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
  -F 'config={"object_detector_options":{"score_threshold":0.25,"max_results":5}};type=application/json' \
  -F 'image=@sample.jpg;type=image/jpeg' \
  http://127.0.0.1:8080/v1/detect
```

### Python client example

```bash
python client.py --score-threshold 0.25 --max-results 5 ./sample.jpg
```

## Configuration

Example `config` JSON:

```json
{
  "object_detector_options": {
    "delegate": "CPU",
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

## Response format

Example response:

```json
{
  "ok": true,
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

## Notes

- Do not send both `category_allowlist` and `category_denylist`.
- `delegate` accepts `CPU`; `GPU` is rejected because it is not provided by the base `tflite-runtime` interpreter.
- `display_names_locale` is rejected because `tflite-runtime` does not expose MediaPipe Tasks label localization.
- Without MediaPipe metadata helpers, categories are returned using the numeric class id as `category_name` and `display_name` is `null`.
- The server caches TFLite interpreter instances by detector options.
- TFLite inference runs in a separate worker thread. HTTP request parsing and response handling remain asynchronous.
