from pathlib import Path

from joblib import dump

import click
import mlflow
import mlflow.sklearn

from src.forest_cover_ml.data import get_dataset
from src.forest_cover_ml.pipeline import create_pipeline


@click.command()
@click.option(
    "--dataset-path-train",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--dataset-path-test",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
    dataset_path_train: Path,
    dataset_path_test: Path,
    save_model_path: Path,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features_train, target_train, features_val = get_dataset(
        dataset_path_train, dataset_path_test
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c)
        pipeline.fit(features_train, features_val)
        target_val = pipeline.predict(target_train)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
