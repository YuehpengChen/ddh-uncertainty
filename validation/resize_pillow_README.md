# RGB preprocessing compatibility

`resize_pillow.mjs` processes **decoded opaque RGB8 samples**. It expands triangular-filter support during downsampling, normalizes boundary weights, quantizes coefficients to 22 bits, and rounds to RGB8 after each separable axis. The algorithm follows the numerical rules in [Pillow 11.3.0 Resample.c](https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c). ImageNet normalization reproduces the float32 `/255`, subtraction and division sequence.

This module contains no model and performs no neural inference.

## Cloud reference proof

Run on GitHub Actions with pinned `Pillow==11.3.0` and NumPy:

```sh
python validation/generate_resize_fixtures.py --output validation/fixtures/resize_pillow.json
node validation/verify_resize_pillow.mjs validation/fixtures/resize_pillow.json validation/resize_pillow_validation.json
```

The helper generates 13 synthetic RGB fixtures. Tests cover small up/downsampling, identity, one-axis resizing, opposing axis scales, 617×563 and 563×617 images, high-frequency checkerboards and isolated edges. All 256 levels in all three channels test normalization. The engineering gate permits at most one RGB8 level of resize difference. The report separately records exact byte parity, so passing the tolerance does not imply exact parity.

The generated JSON contains synthetic pixel buffers, not patient information. Keep the large fixture file in CI artifacts; only the module and compact proof need public deployment.

## Frontend integration

1. Draw the decoded image at its original width and height; do not resize it with Canvas.
2. Read original-resolution RGBA bytes.
3. Call `opaqueRgbaToRgb(rgba, width, height)`; transparent input is explicitly rejected.
4. Call `preprocessRgbPillow(rgb, width, height, 448)` for the float32 NCHW tensor.
5. Keep original width and height for restoring predicted normalized coordinates.

Image decoding, EXIF orientation and ICC/color-space conversion occur before these functions and remain outside this RGB8 compatibility proof. JPEG/WebP decoder differences and transparent-image premultiplication are not covered. Never describe the proof as full file-decoder parity.

Pure arithmetic unit tests, with no model:

```sh
node --test validation/resize_pillow_unit.test.mjs
```
