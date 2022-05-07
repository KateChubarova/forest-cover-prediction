from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(
    csv_path_train: Path, csv_path_test
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_data = pd.read_csv(csv_path_train)
    test_data = pd.read_csv(csv_path_test)
    click.echo(f"Train dataset shape: {train_data.shape}.")
    click.echo(f"Test dataset shape: {test_data.shape}.")
    features_val = train_data["Cover_Type"]
    features_train = train_data.drop("Cover_Type", axis=1)
    click.echo(f"Feature train shape: {features_train.shape}")
    click.echo(f"Feature target shape: {features_val.shape}")
    return features_train, test_data, features_val
