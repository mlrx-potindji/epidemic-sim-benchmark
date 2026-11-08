"""Confirmatory epidemiological equivalence and descriptive difference screens."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

ESTIMANDS: tuple[str, ...] = ("attack_rate", "peak_prevalence", "time_to_peak")


def summarise(series: Mapping[str, Sequence[float]], n_agents: int) -> dict[str, float]:
    """Reduce one run's epidemic curve to its three prespecified outcomes."""
    infected = list(series["infected"])
    peak = max(infected)
    return {
        "attack_rate": (n_agents - series["susceptible"][-1]) / n_agents,
        "peak_prevalence": peak / n_agents,
        "time_to_peak": float(infected.index(peak)),
    }


def pooled_sd(x: Sequence[float], y: Sequence[float]) -> float:
    """Pooled within-arm SD, used to fix confirmatory mean margins from pilot data."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("pilot data must contain at least two replicates per implementation")
    return float(np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                          (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))


def welch_tost(
    x: Sequence[float], y: Sequence[float], margin: float, alpha: float = 0.05
) -> dict:
    """Welch TOST for mean(y)-mean(x), reported as its (1-2*alpha) CI."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("at least two replicates per implementation are required")
    if margin < 0 or not 0 < alpha < 0.5:
        raise ValueError("margin must be nonnegative and alpha must be between 0 and 0.5")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    diff = float(b.mean() - a.mean())
    if se == 0:
        lo = hi = diff
    else:
        df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        critical = stats.t.ppf(1 - alpha, df)
        lo, hi = float(diff - critical * se), float(diff + critical * se)
    return {
        "difference": diff,
        "ci": [lo, hi],
        "margin": float(margin),
        "equivalent": bool(lo > -margin and hi < margin),
        "minimum_supported_margin": max(abs(lo), abs(hi)),
    }


def _wilson(k: int, n: int, z: float) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)


def newcombe_difference_ci(
    successes_x: int, n_x: int, successes_y: int, n_y: int, alpha: float = 0.05
) -> tuple[float, list[float]]:
    """Newcombe hybrid-score interval for the difference in two proportions."""
    z = stats.norm.ppf(1 - alpha)
    px, py = successes_x / n_x, successes_y / n_y
    lx, ux = _wilson(successes_x, n_x, z)
    ly, uy = _wilson(successes_y, n_y, z)
    difference = py - px
    lo = difference - np.sqrt((py - ly) ** 2 + (ux - px) ** 2)
    hi = difference + np.sqrt((uy - py) ** 2 + (px - lx) ** 2)
    return float(difference), [float(lo), float(hi)]


def _scale(values: np.ndarray, axis: int, kind: str) -> np.ndarray:
    if kind == "variance":
        return np.var(values, axis=axis, ddof=1)
    q75, q25 = np.percentile(values, [75, 25], axis=axis)
    return q75 - q25


def scale_equivalence(
    x: Sequence[float],
    y: Sequence[float],
    bounds: tuple[float, float],
    kind: str,
    alpha: float = 0.05,
    n_resamples: int = 9999,
    seed: int = 0,
) -> dict:
    """Bootstrap equivalence interval for Var(y)/Var(x) or IQR(y)/IQR(x)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lower_bound, upper_bound = bounds
    if not 0 < lower_bound < 1 < upper_bound:
        raise ValueError("scale-ratio bounds must be positive and bracket 1")
    if kind not in {"variance", "iqr"}:
        raise ValueError(f"unknown scale statistic: {kind}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    sx, sy = float(_scale(a, 0, kind)), float(_scale(b, 0, kind))
    if sx <= 0 or sy <= 0:
        return {"ratio": None, "ci": None, "method": None, "equivalent": None}

    def log_ratio(sample_x, sample_y, axis=-1):
        scale_x = _scale(sample_x, axis, kind)
        scale_y = _scale(sample_y, axis, kind)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(scale_y / scale_x)

    method_used = "BCa"
    interval = None
    for method in ("BCa", "percentile"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = stats.bootstrap(
                    (a, b),
                    log_ratio,
                    vectorized=True,
                    n_resamples=n_resamples,
                    confidence_level=1 - 2 * alpha,
                    method=method,
                    random_state=np.random.default_rng(seed),
                )
            except ValueError:
                if method == "BCa":
                    continue
                raise
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if not (np.isnan(low) or np.isnan(high)):
            interval = [float(np.exp(low)), float(np.exp(high))]
            method_used = method
            break
    ratio = sy / sx
    if interval is None:
        return {"ratio": ratio, "ci": None, "method": "failed", "equivalent": None}
    equivalent = bool(interval[0] > lower_bound and interval[1] < upper_bound)
    return {"ratio": float(ratio), "ci": interval, "method": method_used,
            "equivalent": equivalent}


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    grid = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(np.sort(x), grid, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), grid, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def wasserstein_statistic(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.wasserstein_distance(x, y))


def brown_forsythe_statistic(x: np.ndarray, y: np.ndarray) -> float:
    dx, dy = np.abs(x - np.median(x)), np.abs(y - np.median(y))
    nx, ny = len(dx), len(dy)
    pooled = ((nx - 1) * dx.var(ddof=1) + (ny - 1) * dy.var(ddof=1)) / (nx + ny - 2)
    if pooled == 0:
        return 0.0
    return float((dx.mean() - dy.mean()) ** 2 / (pooled * (1 / nx + 1 / ny)))


def permutation_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    statistic,
    n_permutations: int,
    seed: int,
) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    observed = statistic(a, b)
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        exceedances += statistic(shuffled[:len(a)], shuffled[len(a):]) >= observed - 1e-12
    return float((exceedances + 1) / (n_permutations + 1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _major_outbreaks(attack_rates: Sequence[float], n_agents: int, n_seed: int, k: int):
    return np.asarray(attack_rates, dtype=float) * n_agents >= k * n_seed - 1e-9


def analyze_outcomes(
    ecs_runs: Sequence[Mapping[str, float]],
    mesa_runs: Sequence[Mapping[str, float]],
    pilot_sd: Mapping[str, float],
    n_agents: int,
    n_seed: int,
    *,
    alpha: float = 0.05,
    margin_sd: float = 0.8,
    outbreak_margin: float = 0.10,
    scale_bounds: tuple[float, float] = (0.5, 2.0),
    n_boot: int = 9999,
    n_permutations: int = 4999,
    min_major_outbreaks: int = 30,
    seed: int = 0,
) -> dict:
    """Run the prespecified mean, scale, outbreak, conditional, and screen analyses."""
    if not ecs_runs or not mesa_runs:
        raise ValueError("both implementations must have confirmatory replicates")
    outcome: dict = {"estimands": {}, "major_outbreak": {}, "conditional_major": {},
                     "difference_screens": []}
    screen_rows = []
    raw_p = []

    for index, key in enumerate(ESTIMANDS):
        x = np.asarray([row[key] for row in ecs_runs], dtype=float)
        y = np.asarray([row[key] for row in mesa_runs], dtype=float)
        margin = margin_sd * pilot_sd[key]
        mean = welch_tost(x, y, margin, alpha)
        mean.update({"pilot_sd": float(pilot_sd[key]), "margin_sd": margin_sd})
        scales = {
            kind: scale_equivalence(
                x, y, scale_bounds, kind, alpha, n_boot, seed + index * 2 + offset
            )
            for offset, kind in enumerate(("variance", "iqr"))
        }
        outcome["estimands"][key] = {
            "ecs": {"mean": float(x.mean()), "sd": float(x.std(ddof=1))},
            "mesa": {"mean": float(y.mean()), "sd": float(y.std(ddof=1))},
            "mean_equivalence": mean,
            "scale_equivalence": scales,
        }
        for screen_name, statistic in (
            ("KS", ks_statistic),
            ("Wasserstein-1", wasserstein_statistic),
            ("Brown-Forsythe", brown_forsythe_statistic),
        ):
            p_value = permutation_pvalue(
                x, y, statistic, n_permutations,
                seed + index * 3 + len(screen_rows),
            )
            screen_rows.append({"estimand": key, "test": screen_name, "p_value": p_value})
            raw_p.append(p_value)

    adjusted = holm_adjust(raw_p)
    for row, p_adjusted in zip(screen_rows, adjusted):
        row["holm_p_value"] = p_adjusted
    outcome["difference_screens"] = screen_rows

    major_x = _major_outbreaks(
        [row["attack_rate"] for row in ecs_runs], n_agents, n_seed, 5
    )
    major_y = _major_outbreaks(
        [row["attack_rate"] for row in mesa_runs], n_agents, n_seed, 5
    )
    successes_x, successes_y = int(major_x.sum()), int(major_y.sum())
    difference, ci = newcombe_difference_ci(
        successes_x, len(major_x), successes_y, len(major_y), alpha
    )
    outcome["major_outbreak"] = {
        "threshold_k": 5,
        "ecs_rate": successes_x / len(major_x),
        "mesa_rate": successes_y / len(major_y),
        "difference": difference,
        "ci": ci,
        "margin": outbreak_margin,
        "equivalent": bool(ci[0] > -outbreak_margin and ci[1] < outbreak_margin),
        "sensitivity": {},
    }
    for k in (3, 10):
        mx = _major_outbreaks(
            [row["attack_rate"] for row in ecs_runs], n_agents, n_seed, k
        )
        my = _major_outbreaks(
            [row["attack_rate"] for row in mesa_runs], n_agents, n_seed, k
        )
        d, interval = newcombe_difference_ci(
            int(mx.sum()), len(mx), int(my.sum()), len(my), alpha
        )
        outcome["major_outbreak"]["sensitivity"][str(k)] = {
            "ecs_rate": float(mx.mean()), "mesa_rate": float(my.mean()),
            "difference": d, "ci": interval,
        }

    if successes_x >= min_major_outbreaks and successes_y >= min_major_outbreaks:
        for key in ESTIMANDS:
            x = np.asarray([row[key] for row, major in zip(ecs_runs, major_x) if major])
            y = np.asarray([row[key] for row, major in zip(mesa_runs, major_y) if major])
            outcome["conditional_major"][key] = welch_tost(
                x, y, margin_sd * pilot_sd[key], alpha
            )
    else:
        outcome["conditional_major"] = None

    outcome["summary"] = {
        "all_mean_equivalent": all(
            item["mean_equivalence"]["equivalent"]
            for item in outcome["estimands"].values()
        ),
        "all_scale_equivalent": all(
            scale["equivalent"] is True
            for item in outcome["estimands"].values()
            for scale in item["scale_equivalence"].values()
        ),
        "major_outbreak_equivalent": outcome["major_outbreak"]["equivalent"],
    }
    outcome["summary"]["all_primary_equivalent"] = bool(
        outcome["summary"]["all_mean_equivalent"]
        and outcome["summary"]["all_scale_equivalent"]
        and outcome["summary"]["major_outbreak_equivalent"]
    )
    return outcome
