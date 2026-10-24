# Public ultrasound examples

These three unchanged image files are redistributed from the NHBS-Net authors' [SPJ dataset](https://github.com/hidden-ops/NHBS-Net_SPJ_dataset), fixed at commit `de5fb68c9094c668205fde3ce1225d337c3e3200`, under its [Apache License 2.0](LICENSE-SPJ.txt). The original repository does not include a separate NOTICE file. All source licensing and attribution notices are retained with this subset.

Original research: Liu et al. (2021), IEEE Transactions on Medical Imaging, [doi:10.1109/TMI.2021.3087857](https://doi.org/10.1109/TMI.2021.3087857).

The authors' follow-up paper, [QualityDDH (2025), Section 4.1](https://doi.org/10.1007/s00371-025-04121-2), explicitly identifies the NHU, SPJ and SPD datasets and reports anonymized reports and ultrasound images. The three frames were additionally reviewed for visible identifying text. Their PNG files contain only image-header, compressed-pixel and end-of-file chunks. No identifying text or EXIF metadata was observed. This documents the reuse review; it is not an independent certification of the source institution's anonymization procedure.

| Demo file | Original file | Changes |
| --- | --- | --- |
| hip-01.png | SPJ_dataset/standard_ultrasound/standard_0.png | Filename only; identical bytes |
| hip-02.png | SPJ_dataset/standard_ultrasound/standard_2.png | Filename only; identical bytes |
| hip-03.png | SPJ_dataset/standard_ultrasound/standard_3.png | Filename only; identical bytes |

The files were chosen by numeric source order, skipping the duplicate `standard_1.png`, before any model prediction. They are examples from a DDH screening dataset. The dataset folder describes standard scan planes, not a per-image DDH diagnosis. No diagnosis, reference angle, outcome or independent validation status is assigned to these examples. Overlap with the model's training data has not been assessed.

Selecting an example downloads it from the demo's own GitHub Pages origin. Predictions and interval calculations run in the visitor's browser only after an explicit request. Displayed overlays are separate canvas layers and never modify these source files.
