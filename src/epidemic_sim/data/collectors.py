"""Result collection helpers shared by simulation implementations."""


def final_value(series: dict, key: str):
    """Return the final observation for a named result series."""
    return series[key][-1]