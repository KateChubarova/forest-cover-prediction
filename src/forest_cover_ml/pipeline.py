from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(use_scaler: bool, max_iter: int, logreg_C: float) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(max_iter=max_iter, C=logreg_C),
        )
    )
    return Pipeline(steps=pipeline_steps)