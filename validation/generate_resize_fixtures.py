"""Generate synthetic Pillow 11.3 reference bytes on GitHub Actions only.

No model, torch import, patient image or research coordinate is used.
Example: python validation/generate_resize_fixtures.py --output validation/fixtures/resize_pillow.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("GITHUB_RUN_ID"):
        raise SystemExit("Refused: Pillow reference fixtures must be generated on GitHub Actions cloud.")
    import numpy as np
    import PIL
    from PIL import Image
    if PIL.__version__ != "11.3.0":
        raise RuntimeError(f"Expected pinned Pillow 11.3.0, found {PIL.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [
        ("single_pixel_up", 1, 1, 13, 11, "constant"),
        ("small_up", 3, 2, 11, 7, "pattern"),
        ("small_down", 11, 9, 3, 2, "pattern"),
        ("single_pixel_down", 12, 9, 1, 1, "pattern"),
        ("horizontal_only", 23, 7, 9, 7, "pattern"),
        ("vertical_only", 11, 23, 11, 8, "pattern"),
        ("mixed_axes", 25, 6, 9, 17, "pattern"),
        ("identity", 19, 17, 19, 17, "pattern"),
        ("original_geometry", 617, 563, 448, 448, "pattern"),
        ("portrait_geometry", 563, 617, 448, 448, "pattern"),
        ("high_frequency_down", 1001, 769, 448, 448, "checker"),
        ("edge_impulse", 617, 563, 448, 448, "impulse"),
        ("up_to_model_grid", 57, 31, 448, 448, "pattern"),
    ]
    result = {"schemaVersion": 1, "synthetic": True, "patientDataUsed": False,
        "generatedBy": "GitHub Actions", "runId": os.environ["GITHUB_RUN_ID"],
        "pillowVersion": PIL.__version__, "numpyVersion": np.__version__,
        "source": "PIL.Image.frombytes('RGB').resize(size, Image.Resampling.BILINEAR)",
        "sourceAlgorithm": "https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c",
        "mode": "RGB", "resample": "BILINEAR", "reducingGap": None, "cases": []}
    for name, width, height, out_width, out_height, pattern in cases:
        pixels = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                for channel in range(3):
                    if pattern == "constant":
                        value = (17, 128, 247)[channel]
                    elif pattern == "checker":
                        value = 255 * ((x + y + channel) % 2)
                    elif pattern == "impulse":
                        value = 255 if x in (0, width // 2, width - 1) or y in (0, height // 2, height - 1) else channel * 13
                    else:
                        value = (x * 17 + y * 31 + channel * 71 + ((x // 3 + y // 5) % 2) * 89) % 256
                    pixels[(y * width + x) * 3 + channel] = value
        source = bytes(pixels)
        expected = Image.frombytes("RGB", (width, height), source).resize(
            (out_width, out_height), resample=Image.Resampling.BILINEAR).tobytes()
        result["cases"].append({"name": name, "width": width, "height": height,
            "outputWidth": out_width, "outputHeight": out_height, "pattern": pattern,
            "sourceRGBBase64": base64.b64encode(source).decode("ascii"),
            "expectedRGBBase64": base64.b64encode(expected).decode("ascii"),
            "sourceSHA256": hashlib.sha256(source).hexdigest(),
            "expectedSHA256": hashlib.sha256(expected).hexdigest()})

    # Float32 arithmetic mirrors torchvision ToTensor followed by Normalize.
    # The LUT tests all 256 byte levels for each channel without a large tensor.
    levels = np.arange(256, dtype=np.float32) / np.float32(255)
    result["normalizationLookup"] = [
        ((levels - np.float32(mean)) / np.float32(std)).tolist()
        for mean, std in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"synthetic": True, "pillowVersion": PIL.__version__, "cases": len(cases),
        "fixture": str(args.output), "bytes": args.output.stat().st_size,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
