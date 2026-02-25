
"""
SDV-based synthetic data generation
----------------------------------
This script:
1. Trains SDV models on baseline immunity and vaccination datasets
2. Generates a joint synthetic exposure dataset (vaccination + infection)
"""

import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def train_baseline_model(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df)
    return synth


def train_response_model(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df)
    return synth


def main():
    baseline = pd.read_csv("../raw/baseline_immunity.csv")
    vaccine = pd.read_csv("../raw/vaccination_data.csv")

    baseline_model = train_baseline_model(baseline)
    response_model = train_response_model(vaccine)

    population = baseline_model.sample(5000)
    population.to_csv("../processed/synthetic_population.csv", index=False)

    events = response_model.sample(10000)
    events.to_csv("../processed/synthetic_exposures.csv", index=False)


if __name__ == "__main__":
    main()
