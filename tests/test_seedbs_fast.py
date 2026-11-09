"""Compiled scans must match the original integers and RNG stream, including ties."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opt_hmm.l0 import seedbs as sb


@unittest.skipIf(sb._load_fast() is None, "optional numba accelerator is not installed")
class FastSeedBSTests(unittest.TestCase):
    def test_integer_scans_match_reference_with_ties_and_dtype_boundaries(self):
        rng = np.random.default_rng(27)
        for length in (2, 5, 31, 124, 180, 181, 496):
            values = rng.random((4, length))
            values[1] = np.round(values[1], 1)
            values[2] = 1.
            values[3] = np.arange(length) % 2
            with self.subTest(length=length):
                expected = sb._ks_numpy(values)
                actual = sb._load_fast().ks_integers(values)
                np.testing.assert_array_equal(actual, expected)
                # At, just below and just above actual maxima: >= must remain inclusive.
                maxima = expected.max(axis=1)
                for threshold in set([0, 1, length * length] +
                                     [int(x + delta) for x in maxima for delta in (-1, 0, 1)]):
                    np.testing.assert_array_equal(sb._load_fast().ks_exceeds(values, threshold),
                                                  maxima >= threshold)

    def test_bootstrap_matches_across_chunks_and_preserves_rng_state(self):
        q = np.round(np.random.default_rng(9).random(496), 2)
        observed = int(sb._ks_numpy(q[None]).max())
        first, second = np.random.default_rng(21), np.random.default_rng(21)
        with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numpy"):
            expected = sb.block_bootstrap_pvalue(q, 20, 131, first, observed)
        with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numba"):
            actual = sb.block_bootstrap_pvalue(q, 20, 131, second, observed)
        self.assertEqual(actual, expected)
        self.assertEqual(first.bit_generator.state, second.bit_generator.state)

    def test_full_precompute_cuts_pvalues_and_selections_match(self):
        for seed in (0, 1, 7):
            rng = np.random.default_rng(seed)
            y = rng.random(124)
            y[:50] = rng.beta(5, 1, 50)
            with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numpy"):
                expected = sb.seedbs_precompute(y, 10, seed, T=99, Tp=79)
            with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numba"):
                actual = sb.seedbs_precompute(y, 10, seed, T=99, Tp=79)
            for key in expected:
                if key != "seconds":
                    np.testing.assert_equal(actual[key], expected[key], err_msg=key)
            for zeta in (.05, .01, .005, .001):
                self.assertEqual(sb.seedbs_select(actual, zeta), sb.seedbs_select(expected, zeta))

    def test_nonfinite_input_uses_reference(self):
        values = np.array([[0., .2, np.nan, np.inf]])
        with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numba"):
            np.testing.assert_array_equal(sb._ks_integers(values), sb._ks_numpy(values))

    def test_process_pool_and_existing_cache_remain_compatible(self):
        from opt_hmm.l2.seedbs_run import SeedBSRunner
        project = Path(__file__).resolve().parents[1]
        (project / "tmp").mkdir(exist_ok=True)
        config = {"l2": {"global_seed": 27, "seedbs":
                         {"T": 39, "Tp": 39, "a": 2**-.5, "min_factor": 2.5}}}
        docs = [{"id": str(i), "y": np.random.default_rng(i).random(31)} for i in range(2)]
        with tempfile.TemporaryDirectory(prefix="sb-", dir=project / "tmp") as directory:
            with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numba"):
                results = SeedBSRunner(config, directory, workers=2).precompute(docs, 6)
            with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numpy"), \
                 patch.object(sb, "seedbs_precompute", side_effect=AssertionError("cache was recomputed")):
                cached = SeedBSRunner(config, directory, workers=1).precompute(docs, 6)
            for actual, saved, doc in zip(results, cached, docs):
                with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numpy"):
                    expected = sb.seedbs_precompute(doc["y"], 6, sb.article_seed(27, doc["id"]), T=39, Tp=39)
                for key in expected:
                    if key != "seconds":
                        np.testing.assert_equal(actual[key], expected[key], err_msg=key)
                    np.testing.assert_equal(saved[key], actual[key], err_msg=key)


class BackendTests(unittest.TestCase):
    def test_numpy_backend_and_invalid_option(self):
        with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numpy"):
            self.assertIsNone(sb._fast_kernels())
        with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="invalid"):
            with self.assertRaises(ValueError):
                sb._fast_kernels()

    def test_optional_dependency_can_be_absent(self):
        with patch.object(sb, "_load_fast", return_value=None):
            with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="auto"):
                self.assertIsNone(sb._fast_kernels())
            with patch.dict(os.environ, OPT_HMM_SEEDBS_BACKEND="numba"):
                with self.assertRaises(ImportError):
                    sb._fast_kernels()


if __name__ == "__main__":
    unittest.main()
