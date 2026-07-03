from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

import aiohttp


async def post_detection(url: str, image_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    data = aiohttp.FormData()
    data.add_field("config", json.dumps(config), content_type="application/json")

    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.expanduser().open("rb") as image_file:
        data.add_field("image", image_file, filename=image_path.name, content_type=content_type)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                text = await response.text()
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"raw_body": text}
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {payload}")
                return payload


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/detect")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--display-names-locale", default=None)
    parser.add_argument("--allow", nargs="*", default=None)
    parser.add_argument("--deny", nargs="*", default=None)
    parser.add_argument("--rotation-degrees", type=int, default=0)
    args = parser.parse_args()

    detector_options: dict[str, Any] = {
        "max_results": args.max_results,
        "score_threshold": args.score_threshold,
    }
    if args.display_names_locale:
        detector_options["display_names_locale"] = args.display_names_locale
    if args.allow:
        detector_options["category_allowlist"] = args.allow
    if args.deny:
        detector_options["category_denylist"] = args.deny

    config: dict[str, Any] = {
        "object_detector_options": detector_options,
        "image_processing_options": {"rotation_degrees": args.rotation_degrees},
    }

    result = await post_detection(args.url, args.image, config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
