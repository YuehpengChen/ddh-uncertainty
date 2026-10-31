# Shared-encoder spatial heatmap comparator

This directory contains the saved implementation, recorded run configuration,
aggregate results and artifact identifiers for a separate spatial heatmap
comparator. It uses a frozen Swin-Tiny encoder and a trainable feature-pyramid
head with five spatial heatmaps. It is distinct from the historical
fresh-decoder Token comparator.

| File | Purpose |
| --- | --- |
| [heatmap_baseline.py](heatmap_baseline.py) | Standalone training and evaluation implementation; unchanged from the deployed source |
| [requirements.txt](requirements.txt) | Implementation dependencies |
| [actual_config.json](actual_config.json) | Arguments recorded by the completed run, with private paths removed |
| [training_history.csv](training_history.csv) | Aggregate selection/refit losses and learning rates, without image identities |
| [aggregate_results.json](aggregate_results.json) | Reference-based localization, interval and paired-comparison metrics with confidence intervals |
| [provenance.json](provenance.json) | Code, input-list, cache, protocol and checkpoint digests; evidence limits |

## Recorded training

The head was selected using a fixed split of the training partition: 1,962
fitting images and 218 validation images. Minimum validation cross-entropy
selected epoch 14; the selection run stopped after 29 epochs with patience
15. A new head initialization was then refit on all 2,180 training images
for 14 epochs. Both phases use an 80-epoch cosine schedule, AdamW with initial
learning rate 0.0002, weight decay 0.01 and batch size 12. The recorded runtime
is PyTorch 2.10.0+cu128 and timm 1.0.19.

The target averages normalized Gaussian kernels from the available raters,
with standard deviation two heatmap cells. Image weights are proportional to
the number of rater rows. The head has 0.1 spatial dropout. Evaluation is
deterministic; heatmap moments define five independent anisotropic Gaussian
blocks, retaining within-landmark x-y covariance and kernel smoothing.

## Inputs supplied by the operator

Study images, annotation files, image-name lists, frozen encoder weights,
feature-cache arrays and trained checkpoint binaries are not included.
The recorded run used the frozen-encoder input artifact and float16 feature
cache identified in `provenance.json`. Their digests must match the supplied
files; using a different artifact or computing raw-image features in a
different precision is a new execution and does not establish bitwise replay.

The script is self-contained apart from the packages in `requirements.txt`.
It has no imports from private study source files. `--data-root` expects the
operator's `multi_image_list.txt`, `multi_train_list.txt`, `multi_val_list.txt`,
`landmarks_a.txt` through `landmarks_e.txt`, and `images/` directory.

The optional `--feature-cache` input expects `level0.npy` through `level3.npy`
as float16 `[N,C,H,W]` arrays and `meta.npz` with `img_names`, `rater_norm`
(or `raters_norm`), `rater_px`, `rater_valid` and `wh`. Its `manifest.json`
must record `status: complete`, `img_size: 448`, `feature_dtype: float16`,
the `labels_sha256` of `meta.npz`, and the matching encoder artifact digest
in `source_hashes`. These arrays and metadata remain private inputs.

With the appropriate CUDA build of the recorded dependencies and the
operator-supplied inputs, the recorded settings are:

```sh
python -B heatmap_baseline.py train \
  --data-root /path/to/DDH_processed \
  --backbone-checkpoint /path/to/encoder_artifact.pt \
  --feature-cache /path/to/shared_cache \
  --output /path/to/new_heatmap_run \
  --device cuda --image-size 448 --heatmap-size 112 --hidden 128 \
  --epochs 80 --patience 15 --batch 12 --workers 4 \
  --lr 0.0002 --weight-decay 0.01 --target-sigma-grid 2 \
  --internal-validation 0.1 --seed 20261031 \
  --freeze-encoder --no-augment --refit-all --annotation-row-weighting

python -B heatmap_baseline.py eval \
  --data-root /path/to/DDH_processed \
  --checkpoint /path/to/new_heatmap_run/refit_last.pt \
  --feature-cache /path/to/shared_cache \
  --output /path/to/new_heatmap_evaluation \
  --device cuda --batch 12 --workers 4 --draws 4096 --seed 20261031
```

Use new output directories. Outputs created by these commands can contain
image identities and per-image arrays; they are not part of this public
package. This export only checked saved source/configuration/logs and
aggregate arithmetic. It did not train a model or perform image inference.

## Interpretation

The new head used training-partition validation for selection. Its frozen
encoder retains the study's existing training history. The comparison is an
internal practical pipeline comparison with common references and assessment,
and does not isolate the causal effect or necessity of tokenization.

Each model/covariance condition has separate interval-rescaling coefficients
fitted on the same calibration subset. Reported assessment confidence
intervals condition on those fitted coefficients and do not include patient
clustering. Intervals containing zero do not establish equivalence, and no
external clinical-validation or distribution-free coverage guarantee is
assigned by this package.

The historical Token batch sizes, label smoothing and Smooth-L1 provenance
remain separately flagged in [config.yaml](../config.yaml). Recorded heatmap
arguments do not resolve the missing historical Token launch records.

The two CLI commands produce heatmap predictions and raw moment-based
intervals. `aggregate_results.json` also records subsequent common rescaling
and paired-bootstrap analysis from saved predictions. That downstream
analysis is identified by code/output digests; this minimal export does not
claim that the two heatmap commands alone recreate every paired table or
confidence interval.
