"""Arithmetic/schema tests only. No research data, training, or inference."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_saved_predictions as ev
from make_synthetic_fixture import fixture, points

HERE = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def test_known_angles_in_degrees(self):
        np.testing.assert_allclose(ev.angles(points(60, 45, 0)), [60, 45], atol=1e-12)

    def test_translation_invariance(self):
        p = points(63, 39, 0)
        np.testing.assert_allclose(ev.angles(p), ev.angles(p + [50, -31]), atol=1e-12)

    def test_jacobian_matches_central_difference(self):
        p = points(60, 45, 0)
        delta = 1e-5
        columns = []
        for j in range(10):
            step = np.zeros(10); step[j] = delta
            columns.append((ev.angles(p + step.reshape(5, 2)) - ev.angles(p - step.reshape(5, 2))) / (2 * delta))
        np.testing.assert_allclose(ev.angle_jacobian(p), np.array(columns).T, atol=2e-8, rtol=1e-7)

    def test_shared_translation_cancels(self):
        j = ev.angle_jacobian(points(60, 45, 0))
        b = np.tile(np.eye(2), (5, 1))
        cov = b @ np.array([[4, 1], [1, 2]]) @ b.T
        np.testing.assert_allclose(j @ cov @ j.T, 0, atol=1e-12)
        self.assertTrue((np.diag(j @ np.diag(np.diag(cov)) @ j.T) > 0).all())

    def test_moments_use_divisor_T(self):
        mean, within, between, total = ev.mixture_moments(np.array([[0] * 10, [2] * 10]), np.ones((2, 10)))
        np.testing.assert_array_equal(mean, np.ones(10))
        np.testing.assert_array_equal(within, np.eye(10))
        np.testing.assert_array_equal(between, np.ones((10, 10)))
        np.testing.assert_array_equal(total, np.ones((10, 10)) + np.eye(10))

    def test_invalid_variance_rejected(self):
        with self.assertRaises(ValueError):
            ev.mixture_moments(np.zeros((2, 10)), -np.ones((2, 10)))
        with self.assertRaises(ValueError):
            ev.checked_covariance(np.diag([-1] + [1] * 9), "invalid")

    def test_total_covariance_does_not_broadcast_vector(self):
        f = fixture()
        base = points(60, 45, 0).reshape(10)
        r = f["images"][0]
        r["pass_means_px"] = np.stack([base - 1, base + 1]).tolist()
        r["pass_variances_px2"] = np.zeros((2, 10)).tolist()
        r["total_covariance_px2"] = [1.0] * 10
        with self.assertRaises(ValueError):
            ev.validate_document(f)

    def test_zero_length_segment_rejected(self):
        p = points(60, 45, 0); p[1] = p[0]
        with self.assertRaises(ValueError): ev.angles(p)

    def test_unsigned_branch_not_given_jacobian(self):
        with self.assertRaises(ValueError): ev.angle_jacobian(points(0, 45, 0))

    def test_weighted_quantile_equal_images(self):
        self.assertEqual(ev.weighted_score_quantile([np.array([0]), np.array([1, 2, 3, 4])], .5), 0)
        self.assertEqual(ev.weighted_score_quantile([np.array([0]), np.array([1, 2, 3, 4])], .6), 1)
        with self.assertRaises(ValueError): ev.weighted_score_quantile([], .9)

    def test_inclusive_coverage(self):
        np.testing.assert_array_equal(ev.image_coverage(np.array([[0, 0], [10, 10]]), np.array([[0, 10], [10, 0]])), [1, 1])

    def test_coverage_is_image_balanced(self):
        r = [{"bounds": np.array([[0, 0], [10, 10]]), "refs": np.array([[5, 5], [15, 15]]), "mean_reference": np.array([10, 10])},
             {"bounds": np.array([[0, 0], [10, 10]]), "refs": np.array([[5, 5]] * 5), "mean_reference": np.array([5, 5])}]
        metrics = ev.interval_metrics(r, np.ones(2), np.empty((0, 2), int))
        self.assertEqual(metrics["alpha"]["rater_coverage_pct"], 75)
        self.assertNotAlmostEqual(metrics["alpha"]["rater_coverage_pct"], 100 * 6 / 7)

    def test_rescaling_keeps_midpoint_and_does_not_clip(self):
        raw = np.array([[2., 150.], [10., 170.]])
        scaled = ev.rescale_intervals(raw, np.array([2., 4.]))
        np.testing.assert_array_equal(scaled.mean(axis=0), raw.mean(axis=0))
        np.testing.assert_array_equal(scaled, [[-2, 120], [14, 200]])

    def test_schema_and_level_checked(self):
        f = fixture(); f["coordinate_units"] = "mm"
        with self.assertRaises(ValueError): ev.analyze(f, bootstrap=0)
        f = fixture(); f["nominal_level"] = .95
        with self.assertRaises(ValueError): ev.analyze(f, bootstrap=0)
        f = fixture(); f["images"][0]["id"] = f["images"][1]["id"]
        with self.assertRaises(ValueError): ev.analyze(f, bootstrap=0)

    def test_assessment_changes_do_not_change_coefficients(self):
        f = fixture(); before = ev.analyze(f, bootstrap=0)
        modified = copy.deepcopy(f)
        for image in modified["images"]:
            if image["partition"] == "assessment":
                image["raters_px"] = [points(80, 20, 0).tolist()] * len(image["raters_px"])
        after = ev.analyze(modified, bootstrap=0)
        self.assertEqual(before["scales"], after["scales"])
        self.assertNotEqual(before["metrics"]["image_balanced_rescaling"]["assessment"], after["metrics"]["image_balanced_rescaling"]["assessment"])

    def test_synthetic_end_to_end_repeatable(self):
        a = ev.analyze(fixture(), bootstrap=64, seed=11)
        b = ev.analyze(fixture(), bootstrap=64, seed=11)
        self.assertEqual(a, b)
        self.assertEqual(a["data_kind"], "synthetic")
        self.assertEqual(a["partition_counts"], {"calibration": 12, "assessment": 12, "all": 24})
        self.assertTrue(all(d["local_propagation"]["available"] for d in a["moment_diagnostics"]))

    def test_original_cache_adapter_matches_json(self):
        f = json.loads((HERE / "fixtures/synthetic_predictions.json").read_text())
        adapted = ev.load_cached(HERE / "fixtures/synthetic_predictions.npz", HERE / "fixtures/synthetic_uncertainty.npz", HERE / "fixtures/synthetic_partitions.csv", None, ev.SALT, .90)
        a, b = ev.analyze(f, bootstrap=16, seed=19), ev.analyze(adapted, bootstrap=16, seed=19)
        self.assertEqual(a["scales"], b["scales"])
        self.assertEqual(a["metrics"], b["metrics"])
        self.assertEqual(b["data_kind"], "synthetic")
        for aa, bb in zip(a["moment_diagnostics"], b["moment_diagnostics"]):
            np.testing.assert_allclose(aa["total_covariance_px2"], bb["total_covariance_px2"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
