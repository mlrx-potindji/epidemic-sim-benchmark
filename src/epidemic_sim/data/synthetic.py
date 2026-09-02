"""SDV-backed synthetic population and exposure generation."""

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


def train_model(dataframe: pd.DataFrame) -> GaussianCopulaSynthesizer:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataframe)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(dataframe)
    return synthesizer


def generate_synthetic_data(data_root, population_size: int = 5000,
                            exposure_size: int = 10000) -> None:
    baseline = pd.read_csv(data_root / "raw" / "baseline_immunity.csv")
    vaccine = pd.read_csv(data_root / "raw" / "vaccination_data.csv")
    population = train_model(baseline).sample(population_size)
    events = train_model(vaccine).sample(exposure_size)
    population.to_csv(data_root / "processed" / "synthetic_population.csv", index=False)
    events.to_csv(data_root / "processed" / "synthetic_exposures.csv", index=False)
