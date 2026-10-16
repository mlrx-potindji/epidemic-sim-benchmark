"""Epidemiological parity: compare distributions of ECS vs Mesa.

This module is part of the benchmark experiment suite, not the library. Use it
only in ``experiments/parity/compare_implementations.py``.

Structural parity (same parameters, same mechanisms, same outputs) is checked by unit
tests. *Epidemiological* parity asks whether the two implementations give the same
distribution of epidemic indicators over independent replicates, in particular the same
mean AND the same variance.

Estimands (one value per replicated run, ``n`` = number of agents):

* ``attack_rate``      = (n - S_final) / n : share of the population ever infected;
* ``peak_prevalence``  = max_t I(t) / n    : highest share infected at the same time;
* ``time_to_peak``     = argmax_t I(t)     : step at which the peak is reached.

For each estimand, ECS and Mesa are compared with:

* the difference of means (Welch t-test) and the two-sample KS test (whole distribution);
* the ratio of variances Var_mesa / Var_ecs with a bootstrap percentile confidence
  interval, and the Brown-Forsythe (median-centred Levene) test of equal variances.
  The F test is not used because the estimands are not normal.
The ``parity`` field is a conservative screening result: no corrected test
rejects and the matching confidence interval for the variance ratio includes 1.
It is not evidence that the implementations are equivalent. Formal equivalence
is evaluated only when users supply a mean-difference margin and variance-ratio
bounds chosen before analysis. Welch intervals and bootstrap intervals must
both fall within those bounds; constant samples yield an inconclusive result.
"""
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

ESTIMANDS: tuple[str, ...] = ("attack_rate", "peak_prevalence", "time_to_peak")


def summarise(series: Mapping[str, Sequence[float]], n_agents: int) -> dict[str, float]:
    """The three estimands of one run, from the time series of the model."""
    infected = list(series["infected"])
    return {"attack_rate": (n_agents - series["susceptible"][-1]) / n_agents,
            "peak_prevalence": max(infected) / n_agents,
            "time_to_peak": float(infected.index(max(infected)))}


def variance_ratio_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                      n_boot: int = 4000, level: float = 0.95
                      ) -> tuple[float | None, float | None, float | None]:
    """Var(b) / Var(a) and its bootstrap percentile confidence interval."""
    variance_a = float(np.var(a, ddof=1))
    variance_b = float(np.var(b, ddof=1))
    if variance_a == 0 or variance_b == 0:
        return None, None, None
    ratio = variance_b / variance_a
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample_a = np.var(rng.choice(a, len(a)), ddof=1)
        sample_b = np.var(rng.choice(b, len(b)), ddof=1)
        boot[i] = sample_b / sample_a if sample_a > 0 else np.nan
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return ratio, None, None
    low, high = np.percentile(boot, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return ratio, float(low), float(high)


def welch_mean_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    level: float,
) -> tuple[float, float, float]:
    """Welch confidence interval for mean(b) - mean(a)."""
    difference = float(b.mean() - a.mean())
    variance_a = float(np.var(a, ddof=1) / len(a))
    variance_b = float(np.var(b, ddof=1) / len(b))
    standard_error = float(np.sqrt(variance_a + variance_b))
    if standard_error == 0:
        return difference, difference, difference
    degrees_freedom = (variance_a + variance_b) ** 2 / (
        variance_a**2 / (len(a) - 1) + variance_b**2 / (len(b) - 1)
    )
    critical_value = stats.t.ppf((1 + level) / 2, degrees_freedom)
    return (
        difference,
        float(difference - critical_value * standard_error),
        float(difference + critical_value * standard_error),
    )


def compare_estimand(ecs: Sequence[float], mesa: Sequence[float], rng: np.random.Generator,
                     alpha: float = 0.01, mean_equivalence_margin: float | None = None,
                     variance_ratio_bounds: tuple[float, float] | None = None,
                     equivalence_alpha: float | None = None) -> dict[str, float | bool | None]:
    """Compare one estimand between the two implementations (replicates are independent)."""
    a, b = np.asarray(ecs, dtype=float), np.asarray(mesa, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("at least two replicates per implementation are required")
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be between 0 and 0.5, got {alpha}")
    equivalence_alpha = alpha if equivalence_alpha is None else equivalence_alpha
    if not 0 < equivalence_alpha < 0.5:
        raise ValueError(f"equivalence_alpha must be between 0 and 0.5, got {equivalence_alpha}")
    if mean_equivalence_margin is not None and mean_equivalence_margin <= 0:
        raise ValueError("mean_equivalence_margin must be positive")
    if variance_ratio_bounds is not None and not (
        0 < variance_ratio_bounds[0] < 1 < variance_ratio_bounds[1]
    ):
        raise ValueError("variance-ratio bounds must bracket 1 and have positive limits")

    ratio, low, high = variance_ratio_ci(a, b, rng, level=1 - alpha)
    difference, difference_low, difference_high = welch_mean_difference_ci(a, b, 1 - alpha)
    both_constant = np.var(a) == 0 and np.var(b) == 0
    p_mean = None if both_constant else stats.ttest_ind(a, b, equal_var=False).pvalue
    p_ks = stats.ks_2samp(a, b, method="asymp").pvalue
    p_variance = None if both_constant else stats.levene(a, b, center="median").pvalue
    result = {
        "mean_ecs": float(a.mean()),
        "mean_mesa": float(b.mean()),
        "sd_ecs": float(a.std(ddof=1)),
        "sd_mesa": float(b.std(ddof=1)),
        "mean_difference_mesa_minus_ecs": difference,
        "mean_difference_ci_low": difference_low,
        "mean_difference_ci_high": difference_high,
        "mean_difference_ci_level": 1 - alpha,
        "p_mean_welch": float(p_mean) if p_mean is not None and np.isfinite(p_mean) else None,
        "p_ks": float(p_ks) if np.isfinite(p_ks) else None,
        "p_variance_brown_forsythe": (
            float(p_variance) if p_variance is not None and np.isfinite(p_variance) else None
        ),
        "variance_ratio": ratio,
        "variance_ratio_low": low,
        "variance_ratio_high": high,
        "ci_level": 1 - alpha,
    }
    result["mean_equivalence_margin"] = mean_equivalence_margin
    result["variance_ratio_bounds"] = variance_ratio_bounds
    result["equivalence_alpha"] = equivalence_alpha
    if mean_equivalence_margin is None:
        result["mean_equivalence_ci_low"] = None
        result["mean_equivalence_ci_high"] = None
        result["mean_equivalent"] = None
    else:
        _, equivalence_low, equivalence_high = welch_mean_difference_ci(
            a, b, 1 - 2 * equivalence_alpha
        )
        result["mean_equivalence_ci_low"] = equivalence_low
        result["mean_equivalence_ci_high"] = equivalence_high
        result["mean_equivalent"] = bool(
            equivalence_low > -mean_equivalence_margin
            and equivalence_high < mean_equivalence_margin
        )
    if variance_ratio_bounds is None:
        result["variance_equivalent"] = None
    else:
        _, equivalence_ratio_low, equivalence_ratio_high = variance_ratio_ci(
            a, b, rng, level=1 - 2 * equivalence_alpha
        )
        result["variance_ratio_equivalence_ci_low"] = equivalence_ratio_low
        result["variance_ratio_equivalence_ci_high"] = equivalence_ratio_high
        if equivalence_ratio_low is None or equivalence_ratio_high is None:
            result["variance_equivalent"] = None
        else:
            result["variance_equivalent"] = bool(
                equivalence_ratio_low > variance_ratio_bounds[0]
                and equivalence_ratio_high < variance_ratio_bounds[1]
            )
    if (
        mean_equivalence_margin is None
        or variance_ratio_bounds is None
        or result["variance_equivalent"] is None
    ):
        result["equivalent"] = None
    else:
        result["equivalent"] = bool(result["mean_equivalent"] and result["variance_equivalent"])
    p_values = (
        result["p_mean_welch"],
        result["p_ks"],
        result["p_variance_brown_forsythe"],
    )
    result["parity"] = bool(
        all(value is not None and np.isfinite(value) for value in p_values)
        and result["variance_ratio"] is not None
        and result["variance_ratio_low"] is not None
        and result["variance_ratio_high"] is not None
        and min(p_values) > alpha
        and result["variance_ratio_low"] <= 1.0 <= result["variance_ratio_high"]
    )
    return result
