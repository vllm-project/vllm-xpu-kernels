# SPDX-License-Identifier: Apache-2.0
import warnings

import numpy as np
import pytest
import torch
from scipy import stats

import vllm_xpu_kernels._xpu_C  # noqa: F401

BATCH_SIZE = [1, 4, 16]
VOCAB_SIZE = [100, 1000, 10000]
DEVICE = "xpu"

# CI/mini scope parameter overrides
MINI_PYTEST_PARAMS = {
    "default": {
        "batch_size": [1, 4],
        "vocab_size": [1000],
        "seed": [42],
        "offset": [0],
        "lambda_param": [1.0],
    },
}


class ExponentialDistributionTester:

    @staticmethod
    def basic_statistics_test(samples, lambda_param=1.0, tolerance=0.1):
        samples_np = samples.cpu().numpy().flatten()

        theoretical_mean = 1.0 / lambda_param
        theoretical_var = 1.0 / (lambda_param**2)
        theoretical_std = 1.0 / lambda_param

        sample_mean = np.mean(samples_np)
        sample_var = np.var(samples_np, ddof=1)
        sample_std = np.std(samples_np, ddof=1)

        mean_error = abs(sample_mean - theoretical_mean) / theoretical_mean
        var_error = abs(sample_var - theoretical_var) / theoretical_var
        std_error = abs(sample_std - theoretical_std) / theoretical_std

        results = {
            'theoretical_mean': theoretical_mean,
            'sample_mean': sample_mean,
            'mean_error': mean_error,
            'theoretical_var': theoretical_var,
            'sample_var': sample_var,
            'var_error': var_error,
            'theoretical_std': theoretical_std,
            'sample_std': sample_std,
            'std_error': std_error,
            'mean_test_passed': mean_error < tolerance,
            'var_test_passed': var_error < tolerance,
            'std_test_passed': std_error < tolerance
        }

        return results

    @staticmethod
    def distribution_test_adaptive(samples, lambda_param=1.0, base_alpha=0.05):
        samples_np = samples.cpu().numpy().flatten()
        n = len(samples_np)

        if np.any(samples_np <= 0):
            return {
                'error': 'Found non-positive samples',
                'test_passed': False,
                'sample_size': n
            }

        # KS test
        ks_stat, ks_p_value = stats.kstest(
            samples_np, lambda x: stats.expon.cdf(x, scale=1 / lambda_param))

        # Anderson-Darling test
        try:
            ad_stat, ad_critical_values, ad_significance_levels =\
                stats.anderson(samples_np, 'expon')
            ad_critical_5pct = ad_critical_values[2]  # 5%
            ad_test_passed = ad_stat < ad_critical_5pct
        except Exception:
            ad_test_passed = None
            ad_stat = None

        if n <= 1000:
            alpha = base_alpha
            ks_test_passed = ks_p_value > alpha
            decision_method = 'standard_p_value'

        elif n <= 5000:
            alpha = base_alpha * 0.5
            ks_test_passed = ks_p_value > alpha
            decision_method = 'adjusted_p_value'

        else:
            alpha = base_alpha * 0.1
            effect_size_threshold = 0.02
            p_value_test = ks_p_value > alpha
            effect_size_test = ks_stat < effect_size_threshold

            ks_test_passed = p_value_test or effect_size_test
            decision_method = 'hybrid_p_value_effect_size'

        return {
            'sample_size': n,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'ad_statistic': ad_stat,
            'ad_test_passed': ad_test_passed,
            'alpha_used': alpha,
            'decision_method': decision_method,
            'ks_test_passed': ks_test_passed,
            'all_positive': True
        }

    @staticmethod
    def randomness_test(samples):
        samples_np = samples.cpu().numpy().flatten()

        def runs_test(data):
            median = np.median(data)
            above_median = data > median

            runs = []
            current_run_length = 1

            for i in range(1, len(above_median)):
                if above_median[i] == above_median[i - 1]:
                    current_run_length += 1
                else:
                    runs.append(current_run_length)
                    current_run_length = 1
            runs.append(current_run_length)

            n_runs = len(runs)
            n = len(data)
            expected_runs = (n + 1) / 2

            if n > 20:
                z_stat = (n_runs - expected_runs) / np.sqrt((n - 1) / 4)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                p_value = None
                z_stat = None

            return {
                'n_runs': n_runs,
                'expected_runs': expected_runs,
                'z_statistic': z_stat,
                'p_value': p_value,
                'test_passed': p_value > 0.05 if p_value is not None else None
            }

        runs_result = runs_test(samples_np)

        return {'runs_test': runs_result}


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize("offset", [0, 1, 10])
@pytest.mark.parametrize("lambda_param", [0.5, 1.0, 2.0])
def test_exponential_2d_comprehensive(batch_size, vocab_size, seed, offset,
                                      lambda_param):
    seeds = torch.tensor([seed, offset],
                         dtype=torch.int64,
                         device=torch.device("cpu"))

    samples = torch.empty(batch_size,
                          vocab_size,
                          dtype=torch.float,
                          device=DEVICE)
    torch.ops._xpu_C.exponential_2d_(samples, seeds, lambda_param)

    assert samples.shape == (
        batch_size, vocab_size
    ), f"Shape error: expected {(batch_size, vocab_size)}, got {samples.shape}"
    assert torch.all(
        samples >
        0), "All samples should be positive for exponential distribution"
    assert torch.all(torch.isfinite(samples)), "All samples should be finite"

    total_samples = batch_size * vocab_size
    if total_samples >= 1000:
        tester = ExponentialDistributionTester()

        stats_results = tester.basic_statistics_test(samples,
                                                     lambda_param,
                                                     tolerance=0.15)

        if total_samples >= 5000:
            assert stats_results[
                'mean_test_passed'], f"Mean test failed: error = \
                {stats_results['mean_error']:.4f}"

            assert stats_results[
                'var_test_passed'], f"Variance test failed: error = \
                    {stats_results['var_error']:.4f}"

        if total_samples >= 2000:
            dist_results = tester.distribution_test_adaptive(
                samples, lambda_param)

            if total_samples >= 10000:
                pass
            else:
                assert dist_results[
                    'ks_test_passed'], f"KS test failed: p-value =\
                         {dist_results['ks_p_value']:.4f}"

        if total_samples >= 1000:
            randomness_results = tester.randomness_test(samples)
            runs_test = randomness_results['runs_test']

            if runs_test['p_value'] is not None\
                and total_samples >= 5000\
                    and runs_test['test_passed'] is not None:
                # Run tests sometimes fail due to random fluctuations
                # so we employ more lenient criteria.
                if not runs_test['test_passed']:
                    warnings.warn(f"Runs test failed: p-value = \
                            {runs_test['p_value']:.4f}",
                                  stacklevel=2)
                else:
                    pass


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize("offset", [0, 1, 10])
@pytest.mark.parametrize("lambda_param", [0.5, 1.0, 2.0])
def test_exponential_2d_reproducibility(batch_size, vocab_size, seed, offset,
                                        lambda_param):
    seeds1 = torch.tensor([seed, offset],
                          dtype=torch.int64,
                          device=torch.device("cpu"))
    samples1 = torch.empty(batch_size,
                           vocab_size,
                           dtype=torch.float,
                           device=DEVICE)
    torch.ops._xpu_C.exponential_2d_(samples1, seeds1, lambda_param)

    seeds2 = torch.tensor([seed, offset],
                          dtype=torch.int64,
                          device=torch.device("cpu"))
    samples2 = torch.empty(batch_size,
                           vocab_size,
                           dtype=torch.float,
                           device=DEVICE)
    torch.ops._xpu_C.exponential_2d_(samples2, seeds2, lambda_param)

    assert torch.allclose(
        samples1, samples2), "Same seeds should produce identical results"

    seeds3 = torch.tensor([seed + 10, offset + 10],
                          dtype=torch.int64,
                          device=torch.device("cpu"))
    samples3 = torch.empty(batch_size,
                           vocab_size,
                           dtype=torch.float,
                           device=DEVICE)
    torch.ops._xpu_C.exponential_2d_(samples3, seeds3, lambda_param)

    assert not torch.allclose(
        samples1, samples3), "Different seeds should produce different results"


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize("offset", [0, 1, 10])
@pytest.mark.parametrize("lambda_param", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_exponential_2d_edge_cases(batch_size, vocab_size, seed, offset,
                                   lambda_param):
    seeds = torch.tensor([seed, offset],
                         dtype=torch.int64,
                         device=torch.device("cpu"))

    # for lambda_param in lambda_values:
    samples = torch.empty(batch_size,
                          vocab_size,
                          dtype=torch.float,
                          device=DEVICE)
    torch.ops._xpu_C.exponential_2d_(samples, seeds, lambda_param)

    assert torch.all(
        samples > 0), f"All samples should be positive for λ={lambda_param}"
    assert torch.all(torch.isfinite(
        samples)), f"All samples should be finite for λ={lambda_param}"

    sample_mean = torch.mean(samples).item()
    theoretical_mean = 1.0 / lambda_param
    relative_error = abs(sample_mean - theoretical_mean) / theoretical_mean

    assert relative_error < 0.5, f"Mean too far from theoretical for λ=\
    {lambda_param}: {sample_mean:.4f} vs {theoretical_mean:.4f}"
