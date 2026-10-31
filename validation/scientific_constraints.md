# Scientific and privacy requirements

This validation folder contains synthetic-coordinate tests. It neither imports a trained model nor runs neural-network inference. Research-source files, checkpoint files and manuscript files remain unchanged.

## Inference boundary

- The public website is static. A visitor downloads the model and runs inference in that visitor's browser. No localhost endpoint, Mac service, SSH inference or remote forwarding is part of the deployed application.
- A browser opened on the author's Mac would use that Mac's resources. Automated neural-model smoke tests must therefore run on a hosted CI runner, not in a local browser. Local JavaScript geometry tests are not neural-model inference.
- Browser model files are public downloads; patient images, names, rater annotations and per-case prediction files are not public model assets.
- Ship synthetic images by default. Permission to distribute model weights does not grant permission to distribute research images or per-case measurements.
- A visitor-selected image must remain in browser memory. No inference upload, analytics upload, remote image URL fetch or user-image persistence is allowed. Explain the browser/CDN model download separately from image privacy.

## Exact geometry and units

- Input order is P1, P2, P3, P4, P5, with each point in [x,y] order.
- The baseline is P2 minus P1. Alpha uses P3 minus P4. Beta uses P5 minus P4. Both unsigned angles use atan2(abs(cross), dot), in degrees from 0 through 180; do not fold obtuse angles to 0 through 90.
- Restore original-image coordinates before calculating angles. Multiply normalized x by image width and normalized y by image height, not width minus one and height minus one. Anisotropic resize generally changes angles.
- Reject nonfinite coordinates and angle-defining segment lengths at or below 1e-6 original-image pixels. The local Jacobian also excludes 0/180-degree branch boundaries.
- P4 contributes position uncertainty to both angles. P1 and P2 contribute to both through the common baseline.
- Coordinates, SD and covariance are in original-image px, px and px². Physical pixel spacing and aspect ratio were unverified in the study; the website must not label these distances in millimeters.

## Model preprocessing and provenance

- The archived replay reads RGB, resizes to 448 by 448 with torchvision/PIL bilinear interpolation, scales channels by 255, and normalizes by mean [0.485,0.456,0.406] and SD [0.229,0.224,0.225].
- Validate browser resize against an explicit synthetic RGB fixture. Canvas bilinear sampling, antialiasing and EXIF orientation can differ from PIL. Record the actual browser implementation and measured tolerances; do not claim bitwise parity without a test.
- Record both checkpoint SHA-256 hashes, source-file hashes, ONNX hashes, export/runtime versions, input/output names and shapes, normalization-buffer origin, public model artifact URL and release tag.
- Training-only re-estimated BatchNorm buffers must match the manuscript evaluation model if manuscript performance is quoted. Stored checkpoint buffers and training-re-estimated buffers are separate variants. Never silently substitute one for the other.
- The archived Stage 2 configuration omits its training dropout probability; replay assumes the source default 0.3. This is an assumption, not a verified training hyperparameter.

## MC dropout and uncertainty

- For the full model, activate predictor dropout, including internal MultiheadAttention attention-weight dropout, while keeping the backbone, codebook, decoder and all BatchNorm normalization fixed. The full-model fixed Stage 1 decoder has no active dropout.
- Never export a globally training-mode model that lets BatchNorm adapt to one uploaded image. ONNX training behavior must preserve Dropout while keeping BatchNormalization in evaluation behavior.
- Inspect the exported graph and test stochastic behavior on hosted CI. Export optimizations or unsupported provider kernels may remove dropout or make all repeated runs deterministic. A deterministic export cannot support a between-pass uncertainty claim.
- Use 50 full five-landmark passes for the study-matching mode. A smaller interactive pass count is exploratory and must display its count; the study's nested-pass analysis gives no independent accuracy guarantee.
- Each pass supplies ten normalized coordinate means and diagonal coordinate variances. A soft code-probability vector is decoded once; it is not an exact marginalization over all discrete code combinations.
- After restoring image coordinates, within-pass covariance is the average predicted diagonal variance. Between-pass covariance is the covariance of the ten-dimensional pass means with denominator T, not T minus one. Their sum is the finite-mixture total covariance.
- Use literal labels such as "within-pass predicted spread", "between-pass prediction variation" and "total spread". These components are not uniquely identified as biological noise and exact Bayesian epistemic uncertainty.
- For a mixture draw, choose one pass for all ten coordinates, then draw the coordinate-axis Gaussian noise conditional on that pass. Independent pass selection per point destroys cross-coordinate dependence.
- Study-matching angle intervals use 4,096 joint coordinate draws and linearly interpolated 5th/95th percentiles. The interval is nominal 90%, uncalibrated and separate for each angle; it does not guarantee simultaneous alpha–beta coverage, clinical safety or an anatomical constraint.
- Angles of the mean coordinates and the mean of sampled angles are different quantities. Label the displayed central marker explicitly.

## Covariance comparison

- To isolate covariance, compare a full Gaussian N(mean,total covariance) with a diagonal Gaussian N(mean,diag(total covariance)), using the same standard-normal draws. Match means and every coordinate variance.
- The finite Gaussian mixture is a third distribution. Comparing it with a Gaussian changes distributional shape as well as dependence; do not attribute that difference to covariance alone.
- Per-landmark ellipses summarize 2-by-2 marginal covariance blocks and cannot display dependence between different landmarks. A point ellipse does not by itself establish an angle interval.
- Do not universally label full covariance as "better" or "smaller". Its angle-variance effect can have either sign.

## What the demo can claim

- Landmark dragging is a geometry illustration, not a new model prediction. Keep its outputs visually and textually separate from neural inference.
- Inference on a synthetic phantom checks execution and coordinate processing. It does not measure clinical localization accuracy or demonstrate the paper's reported performance.
- The study used internal-evaluation images. Public arbitrary-image predictions have no established external accuracy, calibration or expert-review benefit.
- Graf assessment also requires the correct acquisition plane, anatomy and infant age. Angle thresholds alone are not a clinical diagnosis. Use "research demonstration; not for clinical decisions" prominently.

## Source anchors

- Current manuscript: DDH_Main_Concise_20261024, Sections III-D/E and IV-A.
- Current supplement: DDH_Supplement_Concise_20261024, Sections S1, S4/S5 and S14C/D.
- Archived replay: manuscript/revision_20260912/scripts/corrected_inference.py, functions set_dropout_only, angles_from_pixels, sigma_from_logvar, evaluate_variant and main preprocessing definition.
- [ONNX Dropout operator](https://onnx.ai/onnx/operators/onnx__Dropout.html): the runtime training-mode input controls stochastic masking; omitting it or setting it false disables masking. A retained node alone does not prove a stochastic run.
- [ONNX Runtime Web deployment overview](https://onnxruntime.ai/docs/tutorials/web/): WASM and GPU execution have different operator-support coverage. Provider compatibility must be tested on the actual exported graph rather than inferred from a successful native CPU run.
