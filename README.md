# DDH landmark and angle uncertainty

A browser-only research demonstration accompanying **Estimating Landmark and Angle Uncertainty from Multiple Rater Annotations in Infant Hip Ultrasound**.

Website: https://yuehpengchen.github.io/ddh-uncertainty/

Verified plain-layout deployment with all three public examples: [complete cloud browser run](https://github.com/YuehpengChen/ddh-uncertainty/actions/runs/40070107732), [downloadable models and validation reports](https://github.com/YuehpengChen/ddh-uncertainty/releases/tag/browser-v1).

All three public examples passed 50 genuine stochastic passes on the deployed URL, with complete 4,096-draw intervals, one explicit model load, no image uploads and no external inference requests. Original example pixels are displayed without placeholder landmarks before prediction.

In the [initial synthetic-fixture run](https://github.com/YuehpengChen/ddh-uncertainty/actions/runs/40059061109), model download/loading took 9.8 seconds and prediction plus interval sampling took 60.8 seconds. These hosted-run measurements do not predict a visitor device's speed or clinical performance.

## What visitors can do

- Select one of three public, de-identified infant hip ultrasound examples, or open a PNG, JPEG or WebP image in their own browser.
- Explicitly download the inference model and predict five landmarks.
- Run 10, 20 or 50 genuine stochastic passes; 50 matches the study's inference setting.
- Inspect the angle at the mean landmarks, marginal coordinate contours, and nominal central 90% angle intervals.
- Compare a joint Gaussian with its diagonal-covariance counterpart while preserving the coordinate means and variances.
- Drag landmarks to explore geometry without rerunning the model.
- Download the two inference graphs and inspect the published validation reports.

The initial illustration and its landmarks are synthetic and are explicitly labeled as explanatory material. They are not a patient case or an AI prediction. Selecting a public example replaces the illustration with a real, unchanged source image and clears the previous prediction. No placeholder landmarks or angles appear on a public example until the visitor requests a prediction or explicitly edits manual coordinates.

## Public examples

```
Example 1 / 2 / 3 -> Load browser model -> Predict landmarks & intervals
```

The three 256×256 RGB images are from the original authors' [SPJ dataset](https://github.com/hidden-ops/NHBS-Net_SPJ_dataset), distributed under its [Apache License 2.0](examples/LICENSE-SPJ.txt). The authors' [QualityDDH paper, Section 4.1](https://doi.org/10.1007/s00371-025-04121-2), identifies this dataset and reports anonymized reports and ultrasound images. The supplied copies were also inspected for visible identifying text, and their PNG files contain no text or EXIF metadata chunks.

Files are copied byte-for-byte from a pinned source revision, with only local filenames changed. Source filenames, hashes, attribution, anonymization basis and reuse notes are in [the example manifest](examples/manifest.json) and [NOTICE](examples/NOTICE.md). The first three distinct files in numeric standard-plane order were selected before any model inference. The standard-plane folder is not a per-image diagnosis. Individual Graf types, reference angles, clinical outcomes and training-set overlap are not established here. These examples demonstrate execution, not independent clinical validation or calibrated uncertainty.

Example images are fetched only when selected, from the same GitHub Pages origin as the website. The model is still loaded explicitly and inference still runs on the visitor's device. There is no precomputed example prediction.

## Computation and privacy

```
GitHub Pages -- website + model files --> visitor browser
public sample -- selected on demand ---> visitor browser
local image  -- decoded RGB -----------> browser worker
browser CPU  -- landmarks + intervals -> display
```

There is no image-upload endpoint, remote inference API, analytics service, or connection to the author's computer. Loading the page does not run the model. GitHub receives ordinary requests for website and model assets, not the selected image. Use de-identified images and follow your institution's data policies.

The application uses ONNX Runtime Web, one-thread WASM CPU execution, and same-origin runtime files. It does not require WebGPU. Model downloads can be large; first-load time and performance depend on the visitor's device.

## Scientific interpretation

P1–P2 defines the baseline; P4–P3 defines the bony roof; P4–P5 defines the cartilage roof. The unsigned segment angles retain the full 0–180° range. Predictions are restored to original-image pixel coordinates before calculating angles.

For T stochastic passes, the application averages the predicted diagonal conditional variance and adds the full covariance of the pass means, using denominator T. The mixture sampler chooses one shared pass for all ten coordinates in each of 4,096 draws. The controlled covariance comparison uses two Gaussians with identical means and diagonal variances and shared standard-normal samples. The finite mixture is displayed separately.

The backbone runs once per image. Dropout remains active in the prediction head, including attention probability dropout. Batch-normalization buffers remain fixed at the evaluated train-only statistics. The archived Stage 2 checkpoint omits the dropout probability; the export reconstructs 0.3 from the archived source and discloses this limitation in the model manifest.

Independent binary masks are generated afresh by the browser for all 26 dropout sites, with inverse keep-probability scaling inside the graph. Spatial channel masks retain singleton spatial dimensions. Native ONNX Dropout is not used because the initial runtime check found that its stochastic effect was removed.

Decoded opaque RGB8 resizing and float32 normalization pass 13 synthetic Pillow-reference cases with exact equality. This proof does not establish equality of browser image decoding, color-profile handling, or EXIF orientation. Transparent images are rejected. Inputs are limited to 16 megapixels and 20 MB.

This is an internal-validation research model, not a diagnostic device. Nominal intervals do not establish coverage for a new patient. The demo does not validate the acquisition plane, establish diagnostic classification, or reproduce the study's aggregate performance on user-selected images. Manual landmark changes do not update the AI uncertainty distribution.

## Reproducibility and deployment

The original training checkpoints remain in a private build repository. GitHub-hosted CPU conversion checks strict parameter loading and deterministic parity with the source architecture. A separate hosted Chromium run checks the real WASM runtime, 50 distinct dropout predictions, finite variances, and fixed deterministic state after sampling. Only a passing inference bundle is released publicly.

The public deployment workflow downloads the verified release bundle, runs pure geometry tests and full browser UI inference on GitHub-hosted runners, and deploys the static website to GitHub Pages. A final cloud browser check exercises the deployed URL. No neural-network inference or export is performed on the author's Mac or training host.

Source files are divided into the website, `model/` (minimal export architecture), `cloud/` (cloud-only conversion and verification), `validation/` (geometry and preprocessing checks), and `tests/` (cloud browser checks). Model and runtime binaries are release assets rather than Git source files.

Upstream software and the public example images retain their own licenses. Only the three documented public, de-identified example images are distributed. Private study images, annotation tables, training records and full training checkpoints remain excluded.
