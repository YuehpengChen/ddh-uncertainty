# Executable saved-prediction analysis

This folder contains executable Python and JavaScript. It performs geometry, finite-mixture moments, local angular covariance propagation, image-balanced coverage, and interval rescaling. It does not load a network, train a model, run ultrasound inference, generate occlusion masks, or reproduce the study's clinical results without the corresponding authorized saved inputs.

The supplied fixtures and example outputs are **synthetic**. They contain artificial coordinates, with 12 calibration and 12 assessment cases. No infant images, patient annotations, study prediction arrays, or model weights are included. [provenance.json](provenance.json) identifies reviewed source programs and exact copies.

## Run the supplied example

From `repository_review/`, using Python 3.10+ with NumPy 1.26–2.x:

```sh
python3 -B code/eval_saved_predictions.py \
  --input code/fixtures/synthetic_predictions.json \
  --output-dir code/example_output/json \
  --bootstrap 2000

python3 -B -m unittest discover -s code/tests -p 'test_analysis.py' -v
node --test code/tests/test_geometry.mjs
```

The Python test suite uses only NumPy and the standard library. The Node tests use built-in `node:test` and `node:assert`; no npm packages are required. If NumPy is absent on a separate machine, install the dependency listed in [requirements.txt](requirements.txt) in that machine's chosen Python environment.

Verified runtimes: Python 3.12.14, NumPy 2.3.5 and Node 24.19.0. Python passed 17 tests and Node passed 16 tests. [verification.json](verification.json) records the executed scope; [example_output/json/metrics.json](example_output/json/metrics.json) contains the synthetic example result.

The same artificial values are provided in the original research cache layout. Test the NPZ adapter:

```sh
python3 -B code/eval_saved_predictions.py \
  --predictions code/fixtures/synthetic_predictions.npz \
  --uncertainty code/fixtures/synthetic_uncertainty.npz \
  --partition-manifest code/fixtures/synthetic_partitions.csv \
  --output-dir code/example_output/npz \
  --bootstrap 2000
```

Both paths produce `metrics.json` and `interval_metrics.csv`. The JSON contains moment matrices and degree-valued propagation diagnostics; the CSV contains point estimates and complete bootstrap CI endpoints. The program checks that inputs remain unchanged and refuses an output path that would overwrite an input. Outputs omit image identifiers and use input row numbers. Coordinates derived from author-supplied research inputs remain research data.

Regenerate the fixtures, if desired:

```sh
python3 -B code/make_synthetic_fixture.py \
  --output code/fixtures/synthetic_predictions.json \
  --npz-dir code/fixtures
```

## JSON input

See [input_schema.json](input_schema.json) and the complete [synthetic JSON](fixtures/synthetic_predictions.json). All coordinates must already be restored to original-image pixel axes. Normalized or resized-grid values must be transformed before use. The code does not infer physical spacing or correct an unknown pixel aspect ratio.

```text
schema_version: 1
coordinate_units: px
angle_units: deg
coordinate_order: [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
nominal_level: 0.90
data_kind: synthetic | author_supplied
images[]:
  id: unique string
  partition: calibration | assessment
  raters_px: [R,5,2]
  raw_angle_intervals_deg: [lower/upper, alpha/beta] = [2,2]
  pass_means_px: [T,10]
  pass_variances_px2: [T,10]
```

Instead of pass arrays, a record may supply `mean_px` (`[10]` or `[5,2]`), `within_variance_px2` (`[10]`), and `between_covariance_px2` (`[10,10]`). Optional `total_covariance_px2` is checked against their sum. The declared nominal level must match `--level`; the program does not relabel saved 90% bounds as another nominal level.

Raters contribute intact five-point configurations. A separate reference is calculated by averaging their coordinates first and then calculating angles. That reference can differ from an average of rater angles. Angle-defining segments must exceed `1e-6` pixels. Degenerate unsigned-angle branches receive no local Jacobian result.

## Original-layout saved NPZ inputs

These inputs are not distributed. The adapter accepts:

| File | Required arrays |
| --- | --- |
| predictions NPZ | `img_names[N]`, `rater_px[N,Rmax,5,2]`, `rater_valid[N,Rmax]`; optional `gt_px[N,5,2]` must equal mean available-rater coordinates |
| uncertainty NPZ | matching `img_names[N]`, `interval_levels[L]`, `mixture_angle_quantiles[N,L,2,2]`, `mu_px[N,5,2]`, `expected_aleatoric_variance_px2[N,10]`, `cov_epistemic_px2[N,10,10]`, `cov_total_px2[N,10,10]` |
| partition CSV | `image,partition`, one unique row for every cached image |

NPZ files are opened with `allow_pickle=False`. Padding is removed using `rater_valid`. Alternatively, `--calibration-per-stratum '2:94,5:6'` selects fixed counts by SHA-256 ordering within rater-count strata, using `--split-salt`. This option implements the archived ordering rule when those exact saved inputs and counts are available; it does not create an untouched independent cohort. An example with external paths:

```sh
python3 -B code/eval_saved_predictions.py \
  --predictions "$DDH_PREDICTIONS" \
  --uncertainty "$DDH_UNCERTAINTY" \
  --calibration-per-stratum '2:94,5:6' \
  --output-dir "$DDH_ANALYSIS_OUTPUT"
```

Source array names `expected_aleatoric_variance_px2` and `cov_epistemic_px2`, and JavaScript names `conditional` and `epistemic`, are retained for compatibility. They mean within-pass and between-pass model-process components here. The program does not identify physical image noise or annotation noise from those labels.

## Definitions implemented by the code

The coordinate vector is $y=(x_1,y_1,\ldots,x_5,y_5)^T$. With $u=P_2-P_1$, $v=P_3-P_4$, and $w=P_5-P_4$, the unsigned angle function is

$$\theta(a,b)=\frac{180}{\pi}\operatorname{atan2}(|a_xb_y-a_yb_x|,a^Tb),\qquad \alpha=\theta(u,v),\quad\beta=\theta(w,u).$$

For $T$ saved prediction passes, $\mu_t$ contains means in pixels and $D_t$ is a diagonal variance matrix in squared pixels. The retained mixture moments use divisor $T$:

$$\bar\mu=T^{-1}\sum_t\mu_t,\quad\Sigma_A=T^{-1}\sum_tD_t,\quad\Sigma_E=T^{-1}\sum_t(\mu_t-\bar\mu)(\mu_t-\bar\mu)^T,\quad\Sigma=\Sigma_A+\Sigma_E.$$

The local Jacobian $J=\partial(\alpha,\beta)/\partial y$ has degree-per-pixel units. $J\Sigma J^T$ has squared-degree units. The diagonal control uses the same means and marginal variances with off-diagonal entries removed. The propagation is a local approximation; interval coverage is evaluated separately.

For raw endpoints $[L_{ia},U_{ia}]$, define midpoint $m_{ia}=(L_{ia}+U_{ia})/2$, half-width $h_{ia}=(U_{ia}-L_{ia})/2$, and individual-rater angle $\theta_{ira}$. The score is $s_{ira}=|\theta_{ira}-m_{ia}|/h_{ia}$. The coefficient is the left inverse of the image-balanced empirical CDF:

$$\lambda_a=\inf\left\{s:\frac{1}{N_{cal}}\sum_{i\in cal}\frac{1}{R_i}\sum_r1[s_{ira}\le s]\ge\gamma\right\},\qquad I'_{ia}=[m_{ia}-\lambda_a h_{ia},m_{ia}+\lambda_a h_{ia}].$$

Python uses exact `Fraction(1,R_i)` weights to avoid cumulative weight rounding. The one-rater sensitivity rule selects a rater by the archived salted hash and uses rank $\lceil(N_{cal}+1)\gamma\rceil$. It is omitted if that rank exceeds the number of calibration cases. Neither procedure receives a finite-sample coverage guarantee in this package.

Coverage first averages raters within each image, then images:

$$\widehat C_a=\frac{100}{N}\sum_i\frac{1}{R_i}\sum_r1[\theta_{ira}\in I_{ia}]\quad(\%).$$

The coefficient is fitted using calibration records only. Assessment changes cannot alter it. Bootstrap resampling retains each image with all its raters and uses the same draws across methods. CIs condition on fixed coefficients and exclude coefficient-estimation uncertainty. Endpoints are not clipped to 0–180 degrees; midpoint-relative scaling does not change point predictions.

## Verification scope

The Python and Node suites test known geometry, translation cancellation, Jacobians, finite-mixture covariance, invalid inputs, weighted coverage and rescaling; Python also tests the JSON/NPZ adapter agreement and calibration-only fitting. These tests verify arithmetic and execution. They do not establish reproduction of the original numerical study, historical training settings, calibration on new patients, or clinical utility.
