# Research materials: landmark covariance and Graf-angle uncertainty

Public research documentation accompanying **Propagating Landmark Covariance to Graf Angle Uncertainty in Infant Hip Ultrasound** by Yueh-Peng Chen, Po-Sen Hsu, Tzuo-Yau Fan, Kuo-Cheng Liu, Hsuan-Kai Kao, and Shih-Lin Wu. The manuscript is under author review; no journal acceptance or DOI is asserted.

| Material | Contents |
| --- | --- |
| [Training and evaluation configuration](config.yaml) | Architecture, reported hyperparameters, loss schedules and provenance flags (source Table S1) |
| [Artifact SHA-256 identifiers](model_manifest.json) | Original model/evaluation identifiers (source Table S2); private checkpoints and image lists are excluded |
| [Technical note PDF](docs/technical_note.pdf) · [Markdown](docs/technical_note.md) | Detailed native-equation derivations and complete training objectives (source Sections S14–S15) |
| [Extended experiments](docs/extended_experiments.md) | Complete supporting results, experiment descriptions and figure captions |
| [Tables S1–S17](tables/README.md) | All 17 tables in CSV and Markdown, including complete confidence intervals |
| [Supporting figures](figures/figure_sources.json) | Source main Figures 4/7 and supporting Figures S1–S4, as finished SVG/PNG composites |
| [Saved-prediction analysis and tests](code/README.md) | Executable Python/Node analysis, input schema, synthetic fixtures and example outputs |
| [References](docs/references.md) | All 37 references, with numbering from the full source manuscript |
| [Provenance and interpretation](docs/provenance_notes.md) | Historical settings, internal evaluation and remaining uncertainty |
| [Package inventory](package_manifest.json) | File sizes and SHA-256 digests |

## Run the analysis

From `research/`, with Python, NumPy and Node available:

```sh
python3 -B code/eval_saved_predictions.py --input code/fixtures/synthetic_predictions.json --output-dir code/example_output/json
python3 -B -m unittest discover -s code/tests -p 'test_analysis.py' -v
node --test code/tests/test_geometry.mjs
```

The supplied inputs and outputs are synthetic. The code evaluates saved predictions; it does not perform model training or neural-network image inference. The 33 included tests check arithmetic, geometry, input handling and fitting invariants. Passing them does not establish clinical performance or numerical reproduction of the study results. See [code/verification.json](code/verification.json).

## Interpretation and provenance

The 100/142 calibration/assessment split is an internal rescaling analysis, not a fresh independent test set or a distribution-free coverage guarantee.

At 30 versus 50 shared-sequence dropout passes, mean relative alpha/beta SD differences were 0.43%/0.38% for total uncertainty and 6.35%/6.92% for the dropout-only component. These are within-sequence sensitivity results and do not prove convergence across independent repetitions.

Measurements are restored to the original 617×563 exported-image grid. Physical millimeter spacing was not retained in the exported files; distances are in pixels. No verified 1:1 physical pixel-aspect-ratio claim is made.

The technical-note PDF preserves the frozen source notation and its author-review cover. Its “unpublished” designation refers to manuscript publication status, not to access to this repository. Its equation, figure and section numbers refer to the full source record; they may differ from compact-main numbering. The Markdown companion contains the current interpretation notes. Essential methods, evidence and limitations remain in the main manuscript; this repository is not a TMI supplementary attachment.

## Data and reuse

No raw study images, patient-level annotations, per-image prediction arrays, image-name lists, credentials or model-weight binaries are included in this directory. Finished figure composites were reviewed for visible identifying text; the only ultrasound panel is a cropped, annotated manuscript example. Source identifiers and hashes are retained in [figure_sources.json](figures/figure_sources.json).

This public repository permits reading and downloading the files. No additional software, figure, model or dataset reuse license is assigned here. Existing third-party terms remain in force. The browser demonstration and its separately licensed public examples are described in the [main README](../README.md).
