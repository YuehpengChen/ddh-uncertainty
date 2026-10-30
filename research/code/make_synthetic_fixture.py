#!/usr/bin/env python3
"""Make artificial five-point coordinates. Contains no research input data."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from eval_saved_predictions import AXES, mixture_moments


def points(alpha, beta, offset):
    a, b = math.radians(alpha), math.radians(beta)
    p4 = np.array([35.0, 5.0])
    return np.array([[0, 0], [80, 0], p4 + [40 * math.cos(a), 40 * math.sin(a)], p4, p4 + [30 * math.cos(b), -30 * math.sin(b)]]) + offset


def fixture():
    images = []
    shape = np.array([1, .25, 1.1, .3, .7, -.45, .9, .2, 1.25, -.1])
    for i in range(24):
        alpha, beta = 50 + .35 * i, 40 + .3 * i
        base = points(alpha, beta, np.array([.4 * i, .2 * i]))
        rater_count = 5 if i % 3 == 0 else 2
        multipliers = np.linspace(-1, 1, rater_count)
        factor = 1.0 if i < 12 else 1.3
        raters = [points(alpha + m * factor * (2.8 + .02 * i), beta + m * factor * (3.8 + .01 * i), np.array([.4 * i, .2 * i]) + [.1 * m, -.2 * m]) for m in multipliers]
        tau = np.array([-.6, -.2, .2, .6])
        means = base.reshape(1, 10) + tau[:, None] * shape
        variances = np.stack([np.linspace(.8, 1.8, 10) * (1 + .05 * t) for t in tau])
        images.append({"id": f"synthetic-{i:03d}", "partition": "calibration" if i < 12 else "assessment",
                       "raters_px": np.round(raters, 8).tolist(), "pass_means_px": np.round(means, 8).tolist(),
                       "pass_variances_px2": np.round(variances, 8).tolist(),
                       "raw_angle_intervals_deg": [[alpha - 2, beta - 3], [alpha + 2, beta + 3]]})
    return {"schema_version": 1, "data_kind": "synthetic", "description": "Artificial coordinates for arithmetic and CLI tests; no infant/patient image or annotation is used.", "coordinate_units": "px", "angle_units": "deg", "coordinate_order": AXES, "nominal_level": .90, "images": images}


def write_npz(document, output):
    output.mkdir(parents=True, exist_ok=True)
    records = document["images"]
    names = np.array([r["id"] for r in records])
    rater_px = np.zeros((len(records), 5, 5, 2))
    valid = np.zeros((len(records), 5), bool)
    means, within, between, total = [], [], [], []
    for i, r in enumerate(records):
        rr = np.asarray(r["raters_px"])
        rater_px[i, :len(rr)] = rr
        valid[i, :len(rr)] = True
        m, a, e, s = mixture_moments(r["pass_means_px"], r["pass_variances_px2"])
        means.append(m.reshape(5, 2)); within.append(np.diag(a)); between.append(e); total.append(s)
    gt = np.array([np.asarray(r["raters_px"]).mean(axis=0) for r in records])
    np.savez_compressed(output / "synthetic_predictions.npz", img_names=names, rater_px=rater_px, rater_valid=valid, gt_px=gt, data_kind=np.array("synthetic"))
    np.savez_compressed(output / "synthetic_uncertainty.npz", img_names=names, interval_levels=np.array([.90]), mixture_angle_quantiles=np.array([r["raw_angle_intervals_deg"] for r in records])[:, None], mu_px=np.array(means), expected_aleatoric_variance_px2=np.array(within), cov_epistemic_px2=np.array(between), cov_total_px2=np.array(total), data_kind=np.array("synthetic"))
    with (output / "synthetic_partitions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "partition"])
        writer.writeheader(); writer.writerows({"image": r["id"], "partition": r["partition"]} for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--npz-dir", type=Path)
    args = parser.parse_args()
    document = fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.npz_dir:
        write_npz(document, args.npz_dir)
    print(json.dumps({"status": "PASS", "data_kind": "synthetic", "images": len(document["images"]), "research_data_used": False}))


if __name__ == "__main__":
    main()
