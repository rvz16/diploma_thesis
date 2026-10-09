import math
import unittest

import numpy as np

from trajectory_baselines import (
    HTC_FEATURE_NAMES,
    htc_features,
    mc_response_uncertainty,
    normalized_nll,
    saup_distance_weights,
    saup_score,
    simple_uncertainty_propagation,
    uprop_pmi,
    uprop_score,
    uprop_tdp_score,
)


class SingleStepUncertaintyTests(unittest.TestCase):
    def test_normalized_nll(self):
        self.assertAlmostEqual(normalized_nll([0.5, 0.25]),
                               -(math.log(0.5) + math.log(0.25)) / 2)

    def test_mc_response_uncertainty(self):
        got = mc_response_uncertainty([[0.5, 0.5], [0.25]])
        self.assertAlmostEqual(got, (math.log(2) + math.log(4)) / 2)


class SAUPTests(unittest.TestCase):
    def test_paper_weighted_rms_equation(self):
        got = saup_score([1.0, 2.0], [2.0, 3.0])
        self.assertAlmostEqual(got, math.sqrt((2.0 ** 2 + 6.0 ** 2) / 2))

    def test_distance_surrogate(self):
        np.testing.assert_allclose(
            saup_distance_weights([0.1, 0.2], [0.3, 0.5]), [0.4, 0.7]
        )

    def test_single_step_degeneracy(self):
        self.assertEqual(saup_score([2.5], [1.0]), 2.5)
        self.assertEqual(simple_uncertainty_propagation([2.5]), 2.5)


class UPropTests(unittest.TestCase):
    @staticmethod
    def unit_kernel(_distance, _sharpness):
        return 1.0

    def test_pmi_matches_equation_8(self):
        # -log(sum_n K_N(d_n)); use a controlled kernel to isolate aggregation.
        got = uprop_pmi([0.1, 0.2, 0.3], kernel=lambda d, n: d / n)
        self.assertAlmostEqual(got, -math.log(0.2))

    def test_no_extrinsic_term_reduces_to_mean_intrinsic(self):
        got = uprop_tdp_score([2.0, 4.0], [[], [[0.0]]],
                              kernel=self.unit_kernel)
        self.assertEqual(got, 3.0)

    def test_validates_triangular_predecessor_structure(self):
        with self.assertRaises(ValueError):
            uprop_tdp_score([1.0, 2.0], [[], [[0.0], [0.0]]])

    def test_averages_tdp_samples(self):
        tdps = [
            {"intrinsic_uncertainties": [2.0], "predecessor_distances": [[]]},
            {"intrinsic_uncertainties": [4.0], "predecessor_distances": [[]]},
        ]
        self.assertEqual(uprop_score(tdps, kernel=self.unit_kernel), 3.0)


class HTCTests(unittest.TestCase):
    def setUp(self):
        self.steps = [
            {
                "token_confidences": [0.5, 0.25],
                "top1_confidences": [0.6, 0.5],
                "topk_confidences": [0.9, 0.8],
            },
            {
                "token_confidences": [0.8, 0.4, 0.2],
                "top1_confidences": [0.8, 0.7, 0.6],
                "topk_confidences": [0.95, 0.9, 0.85],
            },
        ]

    def test_exact_48_feature_map(self):
        feats = htc_features(self.steps)
        self.assertEqual(tuple(feats), HTC_FEATURE_NAMES)
        self.assertEqual(len(feats), 48)
        self.assertTrue(all(math.isfinite(x) for x in feats.values()))

    def test_known_structure_and_dynamics(self):
        feats = htc_features(self.steps)
        self.assertAlmostEqual(feats["normalized_step_count"], 0.2)
        self.assertEqual(feats["first_token_count"], 2.0)
        self.assertEqual(feats["last_token_count"], 3.0)
        self.assertAlmostEqual(feats["avg_tokens_per_step"], 2.5)
        self.assertAlmostEqual(feats["top1_confidence_change"], 0.15)

    def test_one_step_has_zero_cross_step_dynamics(self):
        feats = htc_features(self.steps[:1])
        self.assertEqual(feats["top1_gradient_mean"], 0.0)
        self.assertEqual(feats["topk_confidence_change"], 0.0)
        self.assertEqual(feats["std_tokens_per_step"], 0.0)

    def test_probability_mass_tolerates_float32_roundoff_only(self):
        rounded = [dict(self.steps[0])]
        rounded[0]["topk_confidences"] = [1.000000238418579, 0.8]
        self.assertTrue(all(math.isfinite(x) for x in htc_features(rounded).values()))

        invalid = [dict(self.steps[0])]
        invalid[0]["topk_confidences"] = [1.01, 0.8]
        with self.assertRaises(ValueError):
            htc_features(invalid)


if __name__ == "__main__":
    unittest.main()
