#!/usr/bin/env python3
"""Evaluate saved landmark predictions; never load or run a neural model.

Adapted analysis: cached calibration and occlusion-table scripts identified in
code/provenance.json. Symbols and units follow code/README.md. Only NumPy is
required. No patient images, model checkpoints, or study inputs are distributed.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

DEG = 180.0 / math.pi
MIN_SEGMENT_PX = 1e-6
SALT = "DDH-rescaling-prespecified-20261030-v1"
AXES = [f"{a}{k}" for k in range(1, 6) for a in ("x", "y")]


def finite_array(value, shape=None, label="array"):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{label}: expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: values must be finite")
    return result


def angles(points):
    """Unsigned alpha(P1->P2,P4->P3), beta(P4->P5,P1->P2), degrees."""
    p = finite_array(points, label="landmarks")
    if p.shape[-2:] != (5, 2):
        raise ValueError("Landmarks must end in five [x,y] coordinates")
    u, v, w = p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 3, :], p[..., 4, :] - p[..., 3, :]
    def theta(a, b):
        if (np.linalg.norm(a, axis=-1) <= MIN_SEGMENT_PX).any() or (np.linalg.norm(b, axis=-1) <= MIN_SEGMENT_PX).any():
            raise ValueError("An angle-defining segment is too short")
        cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return DEG * np.arctan2(np.abs(cross), np.sum(a * b, axis=-1))
    return np.stack([theta(u, v), theta(w, u)], axis=-1)


def angle_jacobian(points):
    """Return J (2 x 10), degrees/pixel; undefined at unsigned branches."""
    p = finite_array(points, (5, 2), "landmarks")
    angles(p)
    u, v, w = p[1] - p[0], p[2] - p[3], p[4] - p[3]
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    da, db = cross(u, v), cross(w, u)
    if abs(da) < 1e-10 or abs(db) < 1e-10:
        raise ValueError("Unsigned-angle Jacobian is undefined at a branch boundary")
    g = lambda a: np.array([-a[1], a[0]]) / np.dot(a, a)
    gu, gv, gw = g(u), g(v), g(w)
    return DEG * np.stack([
        np.concatenate([gu, -gu, gv, -gv, [0, 0]]) * np.sign(da),
        np.concatenate([-gu, gu, [0, 0], gw, -gw]) * np.sign(db),
    ])


def checked_covariance(value, label):
    cov = finite_array(value, (10, 10), label)
    scale = max(1.0, float(np.max(np.abs(cov))))
    if not np.allclose(cov, cov.T, atol=1e-10 * scale, rtol=0):
        raise ValueError(f"{label}: covariance must be symmetric")
    if np.linalg.eigvalsh(cov).min() < -1e-9 * scale:
        raise ValueError(f"{label}: covariance must be positive semidefinite")
    return cov


def mixture_moments(means, variances):
    """Finite mixture moments. The between-pass divisor is T, not T-1."""
    means = finite_array(means, label="pass means")
    variances = finite_array(variances, label="pass variances")
    if means.ndim != 2 or means.shape[1] != 10 or not len(means) or variances.shape != means.shape:
        raise ValueError("Pass means and diagonal variances must both have shape [T,10], T>=1")
    if (variances < 0).any():
        raise ValueError("Pass variances must be nonnegative")
    mean = means.mean(axis=0)
    delta = means - mean
    within = np.diag(variances.mean(axis=0))
    between = delta.T @ delta / len(means)
    return mean, within, between, within + between


def record_moments(record):
    if "pass_means_px" in record or "pass_variances_px2" in record:
        if "pass_means_px" not in record or "pass_variances_px2" not in record:
            raise ValueError("Both pass means and pass variances are required")
        return mixture_moments(record["pass_means_px"], record["pass_variances_px2"])
    mean = finite_array(record["mean_px"], label="cached mean")
    if mean.shape not in {(10,), (5, 2)}:
        raise ValueError("Cached mean must have shape [10] or [5,2]")
    mean = mean.reshape(10)
    diagonal = finite_array(record["within_variance_px2"], (10,), "within variance")
    if (diagonal < 0).any():
        raise ValueError("Within-pass diagonal variances must be nonnegative")
    within = np.diag(diagonal)
    between = checked_covariance(record["between_covariance_px2"], "between covariance")
    return mean, within, between, within + between


def validate_document(document):
    if document.get("schema_version") != 1 or document.get("coordinate_units") != "px" or document.get("angle_units") != "deg":
        raise ValueError("Schema 1 requires coordinate_units='px' and angle_units='deg'")
    if document.get("coordinate_order") != AXES:
        raise ValueError("Coordinate order must be x1,y1,...,x5,y5")
    if document.get("data_kind") not in {"synthetic", "author_supplied"}:
        raise ValueError("Declare data_kind as synthetic or author_supplied")
    records = document.get("images", [])
    if not isinstance(records, list) or not records or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"] for r in records):
        raise ValueError("Images must be a nonempty list of records with nonempty string IDs")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Nonempty images with unique IDs are required")
    prepared = []
    for record in records:
        if record.get("partition") not in {"calibration", "assessment"}:
            raise ValueError("Every record needs a calibration or assessment partition")
        raters = finite_array(record["raters_px"], label="rater coordinates")
        if raters.ndim != 3 or raters.shape[1:] != (5, 2) or not len(raters):
            raise ValueError("Rater coordinates must have shape [R,5,2], R>=1")
        refs = angles(raters)
        mean_reference = angles(raters.mean(axis=0))
        bounds = finite_array(record["raw_angle_intervals_deg"], (2, 2), "interval bounds")
        if (bounds[1] <= bounds[0]).any():
            raise ValueError("Each provided interval must have positive width")
        mean, within, between, total = record_moments(record)
        point_angles = angles(mean.reshape(5, 2))
        if "total_covariance_px2" in record:
            supplied_total = checked_covariance(record["total_covariance_px2"], "total covariance")
            if not np.allclose(total, supplied_total, atol=1e-8, rtol=1e-8):
                raise ValueError("Cached total covariance disagrees with within+between")
        prepared.append({"id": str(record["id"]), "partition": record["partition"], "raters": raters, "refs": refs,
                         "mean_reference": mean_reference, "bounds": bounds, "mean": mean,
                         "within": within, "between": between, "total": total, "point_angles": point_angles})
    if {r["partition"] for r in prepared} != {"calibration", "assessment"}:
        raise ValueError("Nonempty, disjoint calibration and assessment groups are required")
    return prepared


def weighted_score_quantile(scores_by_image, probability):
    """Left inverse of image-balanced empirical CDF, with exact 1/R weights."""
    if not 0 < probability < 1:
        raise ValueError("Probability must be between zero and one")
    terms = []
    for values in scores_by_image:
        values = finite_array(values, label="scores")
        if values.ndim != 1 or not len(values) or (values < 0).any():
            raise ValueError("Each image needs nonnegative rater scores")
        terms.extend((float(v), Fraction(1, len(values))) for v in values)
    if not terms:
        raise ValueError("Calibration scores are empty")
    target = Fraction(str(probability)) * len(scores_by_image)
    cumulative = Fraction(0)
    for value, weight in sorted(terms, key=lambda x: x[0]):
        cumulative += weight
        if cumulative >= target:
            return value
    return max(v for v, _ in terms)


def fit_scales(records, probability, salt=SALT):
    calibration = [r for r in records if r["partition"] == "calibration"]
    weighted, one_rater = [], []
    rank = math.ceil((len(calibration) + 1) * probability)
    for a in range(2):
        scores, selected = [], []
        for r in calibration:
            center, half = r["bounds"].mean(axis=0), np.diff(r["bounds"], axis=0)[0] / 2
            values = np.abs(r["refs"][:, a] - center[a]) / half[a]
            scores.append(values)
            pick = int(sha256((salt + "|rater|" + r["id"]).encode()).hexdigest(), 16) % len(values)
            selected.append(values[pick])
        weighted.append(weighted_score_quantile(scores, probability))
        one_rater.append(float(np.sort(selected)[rank - 1]) if rank <= len(selected) else None)
    return np.asarray(weighted), (np.asarray(one_rater) if all(v is not None for v in one_rater) else None), rank


def rescale_intervals(bounds, scale):
    bounds = finite_array(bounds, (2, 2), "interval bounds")
    scale = finite_array(scale, (2,), "scales")
    if (bounds[1] < bounds[0]).any() or (scale < 0).any():
        raise ValueError("Ordered endpoints and nonnegative scales are required")
    center = bounds.mean(axis=0)
    half = (bounds[1] - bounds[0]) / 2
    return np.stack([center - half * scale, center + half * scale])


def image_coverage(bounds, refs):
    return np.mean((refs >= bounds[0]) & (refs <= bounds[1]), axis=0)


def interval_metrics(records, scale, bootstrap_indices):
    coverage, mean_coverage, widths = [], [], []
    outside = np.zeros(2, dtype=int)
    for r in records:
        bounds = rescale_intervals(r["bounds"], scale)
        coverage.append(image_coverage(bounds, r["refs"]))
        mean_coverage.append((r["mean_reference"] >= bounds[0]) & (r["mean_reference"] <= bounds[1]))
        widths.append(bounds[1] - bounds[0])
        outside += (bounds[0] < 0) | (bounds[1] > 180)
    result = {}
    for a, label in enumerate(("alpha", "beta")):
        out = {"n_images": len(records), "n_rater_measurements": sum(len(r["refs"]) for r in records),
               "outside_0_180_endpoint_images": int(outside[a])}
        for values, name, factor, units in [(coverage, "rater_coverage", 100, "pct"), (mean_coverage, "mean_coordinate_coverage", 100, "pct"), (widths, "mean_width", 1, "deg")]:
            v = np.asarray(values, dtype=float)[:, a]
            out[name + "_" + units] = float(v.mean() * factor)
            out[name + "_ci95_" + units] = (np.quantile(v[bootstrap_indices].mean(axis=1) * factor, [.025, .975]).tolist() if len(bootstrap_indices) else None)
        result[label] = out
    return result


def local_variance_diagnostics(record):
    try:
        j = angle_jacobian(record["mean"].reshape(5, 2))
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}
    diag = np.diag(np.diag(record["total"]))
    covs = {"within": record["within"], "between_full": record["between"], "between_diagonal": np.diag(np.diag(record["between"])), "total_full": record["total"], "total_diagonal": diag}
    result = {"available": True, "angle_variance_units": "deg2", "angle_sd_units": "deg"}
    for name, cov in covs.items():
        variance = np.diag(j @ cov @ j.T)
        if (variance < -1e-7).any():
            raise ValueError("Negative propagated variance beyond numerical tolerance")
        variance = np.maximum(0, variance)
        result[name + "_variance_deg2"] = variance.tolist()
        result[name + "_sd_deg"] = np.sqrt(variance).tolist()
    if not np.allclose(np.asarray(result["within_variance_deg2"]) + result["between_full_variance_deg2"], result["total_full_variance_deg2"], atol=1e-7, rtol=1e-8):
        raise ValueError("Within+between propagation identity failed")
    return result


def analyze(document, probability=.90, bootstrap=2000, seed=20261030, salt=SALT):
    if not 0 < probability < 1 or bootstrap < 0:
        raise ValueError("Require 0<level<1 and bootstrap>=0")
    if not math.isclose(document.get("nominal_level", -1), probability, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Input nominal_level must match the level of the saved intervals and --level")
    records = validate_document(document)
    scales, sensitivity, rank = fit_scales(records, probability, salt)
    modes = {"original_mixture": np.ones(2), "image_balanced_rescaling": scales}
    if sensitivity is not None:
        modes["one_rater_sensitivity"] = sensitivity
    groups = {name: [r for r in records if r["partition"] == name] for name in ("calibration", "assessment")}
    groups["all"] = records
    rng = np.random.default_rng(seed)
    draws = {name: rng.integers(0, len(rr), (bootstrap, len(rr))) for name, rr in groups.items()}
    metrics = {mode: {group: interval_metrics(rr, scale, draws[group]) for group, rr in groups.items()} for mode, scale in modes.items()}
    diagnostics = [local_variance_diagnostics(r) for r in records]
    return {"scope": "model-free analysis of author-supplied saved predictions", "data_kind": document.get("data_kind", "author_supplied"),
            "units": {"coordinates": "px", "coordinate_covariance": "px2", "angles": "deg", "angular_covariance": "deg2"},
            "n_images": len(records), "nominal_level": probability,
            "scales": {name: dict(zip(("alpha", "beta"), map(float, scale))) for name, scale in modes.items()},
            "one_rater_order_rank": rank, "one_rater_sensitivity_available": sensitivity is not None,
            "bootstrap": {"resamples": bootstrap, "seed": seed, "unit": "image", "quantile_method": "linear", "coefficients_fixed": True},
            "partition_counts": {name: len(rr) for name, rr in groups.items()}, "metrics": metrics,
            "moment_diagnostics": [{"input_row": k, "partition": r["partition"], "mean_px": r["mean"].tolist(), "within_covariance_px2": r["within"].tolist(), "between_covariance_px2": r["between"].tolist(), "total_covariance_px2": r["total"].tolist(), "local_propagation": diag} for k, (r, diag) in enumerate(zip(records, diagnostics))],
            "interpretation": ["The equal total image weight is distributed as 1/R among its available raters.", "Scaling changes interval midpoint-relative half-widths, not point predictions; endpoints are not clipped.", "Bootstrap intervals condition on fitted coefficients and exclude their estimation uncertainty.", "Within-pass and between-pass are model-process components, not identified physical noise sources.", "These computations provide no fresh-cohort or finite-sample coverage guarantee, external validation, or clinical benefit claim.", "No model training, image inference, or research-data reproduction was performed by this program."]}


def load_cached(predictions, uncertainty, partition_manifest, per_stratum, salt, level):
    with np.load(predictions, allow_pickle=False) as z:
        p = {k: z[k] for k in z.files}
    with np.load(uncertainty, allow_pickle=False) as z:
        u = {k: z[k] for k in z.files}
    names = p["img_names"].astype(str)
    n = len(names)
    if len(set(names)) != n or not np.array_equal(names, u["img_names"].astype(str)):
        raise ValueError("Prediction and uncertainty image order/IDs must match uniquely")
    valid = np.asarray(p["rater_valid"])
    if valid.ndim != 2 or valid.shape[0] != n or not np.isin(valid, [0, 1]).all():
        raise ValueError("Invalid rater mask")
    valid = valid.astype(bool)
    counts = valid.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("Every image needs at least one rater")
    if partition_manifest:
        with Path(partition_manifest).open(newline="", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        if len(manifest) != n or len({row["image"] for row in manifest}) != n:
            raise ValueError("Partition CSV must contain one unique image row per cached image")
        partitions = {row["image"]: row["partition"] for row in manifest}
        if set(partitions) != set(names):
            raise ValueError("Partition image IDs differ from cache")
    elif per_stratum:
        sizes = dict((int(a), int(b)) for a, b in (term.split(":") for term in per_stratum.split(",")))
        if set(sizes) != set(map(int, np.unique(counts))):
            raise ValueError("Specify a calibration count for every rater-count stratum")
        calibration = set()
        for nr, wanted in sizes.items():
            eligible = np.flatnonzero(counts == nr)
            if not 0 < wanted < len(eligible):
                raise ValueError("Each stratum needs calibration and assessment images")
            ordered = sorted(eligible, key=lambda i: (sha256((salt + "|split|" + names[i]).encode()).hexdigest(), names[i]))
            calibration.update(map(int, ordered[:wanted]))
        partitions = {name: "calibration" if i in calibration else "assessment" for i, name in enumerate(names)}
    else:
        raise ValueError("NPZ input requires --partition-manifest or --calibration-per-stratum")
    levels = finite_array(u["interval_levels"], label="interval levels")
    found = np.flatnonzero(np.isclose(levels, level, atol=1e-10, rtol=0))
    if len(found) != 1:
        raise ValueError("Exactly one saved interval level must match --level")
    bounds = finite_array(u["mixture_angle_quantiles"][:, int(found[0])], (n, 2, 2), "cached intervals")
    images = []
    for i, name in enumerate(names):
        raters = finite_array(p["rater_px"][i, valid[i]], label="rater coordinates")
        if "gt_px" in p and not np.allclose(raters.mean(axis=0), p["gt_px"][i], atol=1e-8, rtol=0):
            raise ValueError("Cached mean-coordinate reference disagrees with available raters")
        images.append({"id": str(name), "partition": partitions[name], "raters_px": raters.tolist(), "raw_angle_intervals_deg": bounds[i].tolist(),
                       "mean_px": u["mu_px"][i].reshape(10).tolist(), "within_variance_px2": u["expected_aleatoric_variance_px2"][i].tolist(),
                       "between_covariance_px2": u["cov_epistemic_px2"][i].tolist(), "total_covariance_px2": u["cov_total_px2"][i].tolist()})
    synthetic = all("data_kind" in cache and cache["data_kind"].shape == () and str(cache["data_kind"].item()) == "synthetic" for cache in (p, u))
    return {"schema_version": 1, "coordinate_units": "px", "angle_units": "deg", "coordinate_order": AXES, "nominal_level": level, "data_kind": "synthetic" if synthetic else "author_supplied", "images": images}


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for mode, groups in result["metrics"].items():
        for group, values in groups.items():
            for angle, metrics in values.items():
                row = {"mode": mode, "partition": group, "angle": angle, "scale_lambda": result["scales"][mode][angle]}
                for key, value in metrics.items():
                    if isinstance(value, list):
                        row[key + "_low"], row[key + "_high"] = value
                    elif key.endswith("ci95_pct") or key.endswith("ci95_deg"):
                        row[key + "_low"], row[key + "_high"] = None, None
                    else:
                        row[key] = value
                rows.append(row)
    with (output / "interval_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Schema-1 JSON containing saved predictions, intervals and rater coordinates")
    inputs.add_argument("--predictions", type=Path, help="Original-layout saved predictions NPZ; requires --uncertainty")
    parser.add_argument("--uncertainty", type=Path, help="Original-layout propagated uncertainty NPZ")
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--partition-manifest", type=Path, help="CSV with image,partition columns")
    split.add_argument("--calibration-per-stratum", help="Fixed counts, e.g. '2:94,5:6'; never selected from outcome values")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--level", type=float, default=.90)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--split-salt", default=SALT)
    args = parser.parse_args(argv)
    if args.input and (args.uncertainty or args.partition_manifest or args.calibration_per_stratum):
        parser.error("JSON input carries its own partitions; NPZ-only arguments cannot be combined with --input")
    if args.predictions and not args.uncertainty:
        parser.error("--predictions requires --uncertainty")
    try:
        inputs_used = {"schema_json": args.input} if args.input else {"predictions_npz": args.predictions, "uncertainty_npz": args.uncertainty}
        if args.partition_manifest:
            inputs_used["partition_csv"] = args.partition_manifest
        if any((args.output_dir / name).resolve() == path.resolve() for name in ("metrics.json", "interval_metrics.csv") for path in inputs_used.values()):
            raise ValueError("An output file would overwrite an input file")
        before = {name: sha256(path.read_bytes()).hexdigest() for name, path in inputs_used.items()}
        if args.input:
            document = json.loads(args.input.read_text(encoding="utf-8"))
        else:
            document = load_cached(args.predictions, args.uncertainty, args.partition_manifest, args.calibration_per_stratum, args.split_salt, args.level)
        result = analyze(document, args.level, args.bootstrap, args.seed, args.split_salt)
        result["source_sha256"] = before
        if before != {name: sha256(path.read_bytes()).hexdigest() for name, path in inputs_used.items()}:
            raise RuntimeError("Input files changed during analysis")
        write_outputs(result, args.output_dir)
        print(json.dumps({"status": "PASS", "data_kind": result["data_kind"], "n_images": result["n_images"], "partition_counts": result["partition_counts"], "scales": result["scales"], "no_training_or_inference": True}))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
