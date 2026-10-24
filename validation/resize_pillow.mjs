/**
 * RGB8 bilinear resampling compatible with Pillow's separable filter rules.
 * Independent JavaScript implementation of the numerical algorithm documented
 * by Pillow 11.3.0 src/libImaging/Resample.c (BILINEAR, whole-image box).
 * https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c
 *
 * Scope: decoded, opaque RGB8 samples; no file decoding, EXIF or color-profile
 * conversion. Compatibility is established only by the cloud fixture report.
 * No neural model is imported or evaluated by this module.
 */
const COEFFICIENT_SCALE = 2 ** 22;
const HALF_UNIT = COEFFICIENT_SCALE / 2;
export const IMAGENET_MEAN = Object.freeze([0.485, 0.456, 0.406]);
export const IMAGENET_STD = Object.freeze([0.229, 0.224, 0.225]);

function checkDimensions(width, height) {
  if (![width, height].every(v => Number.isSafeInteger(v) && v > 0 && v <= 100000)) {
    throw new RangeError('Positive integer image dimensions at most 100000 are required.');
  }
  if (width * height > 100000000) throw new RangeError('Image exceeds the 100-megapixel processing limit.');
}

function checkRgb(samples, width, height) {
  checkDimensions(width, height);
  if (!(samples instanceof Uint8Array || samples instanceof Uint8ClampedArray) || samples.length !== width * height * 3) {
    throw new RangeError('Expected exactly width*height*3 unsigned RGB bytes.');
  }
}

/** Precompute the normalized triangular filter in Pillow's 22-bit precision. */
function axisCoefficients(inputSize, outputSize) {
  const scale = inputSize / outputSize;
  const support = Math.max(1, scale);
  const inverseSupport = 1 / support;
  return Array.from({length: outputSize}, (_, outputIndex) => {
    const center = (outputIndex + 0.5) * scale;
    const first = Math.max(0, Math.trunc(center - support + 0.5));
    const end = Math.min(inputSize, Math.trunc(center + support + 0.5));
    const weights = [];
    let total = 0;
    for (let inputIndex = first; inputIndex < end; inputIndex++) {
      const distance = Math.abs((inputIndex - center + 0.5) * inverseSupport);
      const weight = distance < 1 ? 1 - distance : 0;
      weights.push(weight);
      total += weight;
    }
    return {
      first,
      weights: Int32Array.from(weights, weight => Math.trunc(0.5 + weight / total * COEFFICIENT_SCALE)),
    };
  });
}

const byteFromAccumulator = accumulator => Math.min(255, Math.max(0, Math.floor(accumulator / COEFFICIENT_SCALE)));

/** Return interleaved RGB8; round after each axis, matching the RGB8 pipeline. */
export function resizeRgbBilinearPillow(rgb, width, height, outputWidth = 448, outputHeight = 448) {
  checkRgb(rgb, width, height);
  checkDimensions(outputWidth, outputHeight);
  if (outputWidth * height > 100000000) throw new RangeError('Intermediate resize buffer exceeds the processing limit.');
  let intermediate = rgb;
  if (outputWidth !== width) {
    const horizontal = axisCoefficients(width, outputWidth);
    intermediate = new Uint8Array(outputWidth * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < outputWidth; x++) {
        const {first, weights} = horizontal[x];
        const destination = (y * outputWidth + x) * 3;
        for (let channel = 0; channel < 3; channel++) {
          let accumulator = HALF_UNIT;
          let source = (y * width + first) * 3 + channel;
          for (const weight of weights) {
            accumulator += rgb[source] * weight;
            source += 3;
          }
          intermediate[destination + channel] = byteFromAccumulator(accumulator);
        }
      }
    }
  }
  if (outputHeight === height) return new Uint8Array(intermediate);
  const vertical = axisCoefficients(height, outputHeight);
  const output = new Uint8Array(outputWidth * outputHeight * 3);
  for (let y = 0; y < outputHeight; y++) {
    const {first, weights} = vertical[y];
    for (let x = 0; x < outputWidth; x++) {
      const destination = (y * outputWidth + x) * 3;
      for (let channel = 0; channel < 3; channel++) {
        let accumulator = HALF_UNIT;
        let source = (first * outputWidth + x) * 3 + channel;
        for (const weight of weights) {
          accumulator += intermediate[source] * weight;
          source += outputWidth * 3;
        }
        output[destination + channel] = byteFromAccumulator(accumulator);
      }
    }
  }
  return output;
}

/** Mirror float32 ToTensor(/255), subtraction and division, in NCHW order. */
export function normalizeImageNetPlanar(rgb, width, height) {
  checkRgb(rgb, width, height);
  const area = width * height;
  const output = new Float32Array(area * 3);
  for (let channel = 0; channel < 3; channel++) {
    const mean = Math.fround(IMAGENET_MEAN[channel]);
    const std = Math.fround(IMAGENET_STD[channel]);
    for (let pixel = 0; pixel < area; pixel++) {
      const scaled = Math.fround(rgb[pixel * 3 + channel] / 255);
      output[channel * area + pixel] = Math.fround(Math.fround(scaled - mean) / std);
    }
  }
  return output;
}

/** Decode-free conversion. Reject transparency rather than silently composite. */
export function opaqueRgbaToRgb(rgba, width, height) {
  checkDimensions(width, height);
  if (!(rgba instanceof Uint8Array || rgba instanceof Uint8ClampedArray) || rgba.length !== width * height * 4) {
    throw new RangeError('Expected exactly width*height*4 unsigned RGBA bytes.');
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel++) {
    if (rgba[pixel * 4 + 3] !== 255) throw new RangeError('Transparent images are unsupported; provide an opaque RGB image.');
    rgb.set(rgba.subarray(pixel * 4, pixel * 4 + 3), pixel * 3);
  }
  return rgb;
}

export function preprocessRgbPillow(rgb, width, height, size = 448) {
  const resized = resizeRgbBilinearPillow(rgb, width, height, size, size);
  return normalizeImageNetPlanar(resized, size, size);
}
