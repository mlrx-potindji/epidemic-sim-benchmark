
"""
SDV-based synthetic data generation
----------------------------------
This script:
1. Trains SDV models on baseline immunity and vaccination datasets
2. Generates a joint synthetic exposure dataset (vaccination + infection)
"""

import pandas as pd
from pathlib import Path
from epidemic_sim.data.synthetic import generate_synthetic_data


def main():
    data_root = Path(__file__).resolve().parents[1] / "data"
    generate_synthetic_data(data_root)


if __name__ == "__main__":
    main()
