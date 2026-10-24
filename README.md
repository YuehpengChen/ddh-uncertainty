# DDH landmark and angle uncertainty

A browser-only research demonstration accompanying **Estimating Landmark and Angle Uncertainty from Multiple Rater Annotations in Infant Hip Ultrasound**.

Website: https://yuehpengchen.github.io/ddh-uncertainty/

Verified reference-style deployment with measured download/pass progress, switchable full-covariance contours, and all three public examples: [complete cloud browser run](https://github.com/YuehpengChen/ddh-uncertainty/actions/runs/40084376464), [model artifacts and validation reports](https://github.com/YuehpengChen/ddh-uncertainty/releases/tag/browser-v1).

All three public examples passed 50 genuine stochastic passes on the deployed URL, with complete 4,096-draw intervals, one explicit model load, no image uploads and no external inference requests. Original example pixels are displayed without placeholder landmarks before prediction.

The same run verified cumulative download-byte updates, phase-specific progress, measured elapsed time, timer cleanup after completion/cancellation, collapsed explanations, and absence of the manual model-download controls. All 15 public-example landmark contours in each mode were checked directly against the genuine worker covariance, before deployment and again on the public URL. Mode switching preserved landmark positions, angles, interval tables and plots, and performed no new network requests or worker inference messages. The checks record actual x-y correlations, principal-axis angles, axis ratios, and contour-equation errors rather than requiring a visually obvious tilt. Desktop and 390-pixel mobile layouts passed without horizontal overflow.

In the [initial synthetic-fixture run](https://github.com/YuehpengChen/ddh-uncertainty/actions/runs/40059061109), model download/loading took 9.8 seconds and prediction plus interval sampling took 60.8 seconds. These hosted-run measurements do not predict a visitor device's speed or clinical performance.

## What visitors can do

- Select one of three public, de-identified infant hip ultrasound examples, or open a PNG, JPEG or WebP image in their own browser.
- Explicitly download the inference model and predict five landmarks.
- Follow actual model-download bytes, completed prediction passes, and elapsed time. Preparation and interval calculation use indeterminate progress rather than an estimated percentage.
- Run 10, 20 or 50 genuine stochastic passes; 50 matches the study's inference setting.
- Inspect the angle at the mean landmarks, marginal coordinate contours, and nominal central 90% angle intervals.
- Switch coordinate contours between total uncertainty and variation of repeated prediction centers without rerunning the model or changing the angle results.
- Compare a joint Gaussian with its diagonal-covariance counterpart while preserving the coordinate means and variances.
- Drag landmarks to explore geometry without rerunning the model.
- Inspect published model and validation records in the source repository.

The plain interface keeps operation controls and results visible. Method, interval accounting, example provenance, and limitations are available in collapsed details. The header uses the manuscript's exact title. Authors remain pending confirmation because the current manuscript contains no author list.

The layout follows the related [hip-ultrasound editing demonstration](https://yuehpengchen.github.io/hip-ultrasound-editing/): a centered paper heading, white Arial text layout, black navigation links, simple native-style controls, and a narrow control column. The single DDH image is capped at 720 pixels; the page switches to one column on small screens. Paper content and model behavior remain specific to this uncertainty demonstration.

```
Download bytes -> Verify -> Prepare -> Completed passes -> Angle intervals
measured %       waiting   waiting   measured %           waiting
```

The download percentage covers the two model graphs, not runtime files or setup time. The elapsed clock measures the current operation; it is not a remaining-time estimate. Completion, cancellation, and errors stop that clock.

The interface has no separate model-details/download disclosure or manual weight-download buttons. Loading the browser model still retrieves the same verified graphs for inference; removing these controls does not make public model assets private.

## Paper and licensing

The accompanying paper is a research manuscript. Author names and order remain to be confirmed; no publication status or DOI is assigned by this demo. No reuse license has been specified for the paper, demo code, or model. Model downloads are available without asserting an additional reuse grant. The SPJ example images retain Apache-2.0 terms, and ONNX Runtime Web retains its MIT license and third-party notices.

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

Gray coordinate contours use each landmark's complete 2-by-2 block of total covariance, including its x-y covariance. They can be oblique; their orientation and axis lengths are computed from that block, not forced to the image axes. The conditional covariance predicted by this model is diagonal. Within-landmark x-y covariance comes from the shared stochastic-pass means, and adding conditional variance can make the total contour look more axis-aligned or circular. Cross-landmark covariance affects the joint angle distribution and is not represented in a single landmark's marginal ellipse.

The Coordinate spread selector defaults to 總不確定性 (total uncertainty). 多次預測中心的變動 selects only the covariance of repeated prediction means. Both are centered at the same model mean, use radius 2 and identical original-pixel scaling, and retain the same point-marker size. Center-only ellipses are not magnified to match total ellipses. Small center markers and drawing contours above measurement lines keep the smaller ellipses visible. The switch redraws existing results only; it does not rerun inference, change landmark coordinates, or recalculate angle intervals. The choice remains available after restoring manually moved landmarks; it is disabled without a prediction, after manual edits, or while contours are hidden.

Pure geometry tests cover positive/negative oblique ellipses, axis-aligned ellipses, circles, and original-pixel scaling. Cloud browser checks independently compare the actual canvas contour vertices against their genuine inference covariance using a Mahalanobis-radius equation; they do not alter the predictions or force a visible tilt.

The backbone runs once per image. Dropout remains active in the prediction head, including attention probability dropout. Batch-normalization buffers remain fixed at the evaluated train-only statistics. The archived Stage 2 checkpoint omits the dropout probability; the export reconstructs 0.3 from the archived source and discloses this limitation in the model manifest.

Independent binary masks are generated afresh by the browser for all 26 dropout sites, with inverse keep-probability scaling inside the graph. Spatial channel masks retain singleton spatial dimensions. Native ONNX Dropout is not used because the initial runtime check found that its stochastic effect was removed.

Decoded opaque RGB8 resizing and float32 normalization pass 13 synthetic Pillow-reference cases with exact equality. This proof does not establish equality of browser image decoding, color-profile handling, or EXIF orientation. Transparent images are rejected. Inputs are limited to 16 megapixels and 20 MB.

This is an internal-validation research model, not a diagnostic device. Nominal intervals do not establish coverage for a new patient. The demo does not validate the acquisition plane, establish diagnostic classification, or reproduce the study's aggregate performance on user-selected images. Manual landmark changes do not update the AI uncertainty distribution.

## Reproducibility and deployment

The original training checkpoints remain in a private build repository. GitHub-hosted CPU conversion checks strict parameter loading and deterministic parity with the source architecture. A separate hosted Chromium run checks the real WASM runtime, 50 distinct dropout predictions, finite variances, and fixed deterministic state after sampling. Only a passing inference bundle is released publicly.

The public deployment workflow downloads the verified release bundle, runs pure geometry tests and full browser UI inference on GitHub-hosted runners, and deploys the static website to GitHub Pages. A final cloud browser check exercises the deployed URL. No neural-network inference or export is performed on the author's Mac or training host.

Source files are divided into the website, `model/` (minimal export architecture), `cloud/` (cloud-only conversion and verification), `validation/` (geometry and preprocessing checks), and `tests/` (cloud browser checks). Model and runtime binaries are release assets rather than Git source files.

Upstream software and the public example images retain their own licenses. Only the three documented public, de-identified example images are distributed. Private study images, annotation tables, training records and full training checkpoints remain excluded.
