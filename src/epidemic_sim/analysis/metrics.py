"""Metrics for epidemic simulation output."""


def peak_infected(time_series: dict) -> int:
    return max(time_series["infected"])


def time_to_peak(time_series: dict) -> int:
    return time_series["infected"].index(peak_infected(time_series))