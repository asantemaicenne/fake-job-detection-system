import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from configs.settings import settings
from src.models.training.preprocessing import build_feature_pipeline

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """
    Orchestrates temporal cross-validation, hyperparameter tuning via Optuna,
    and final model serialization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "is_fake",
        timestamp_column: str = "posted_at",
        text_column: str = "description",
        numerical_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes trainer with data sorted strictly in chronological order.
        """
        if timestamp_column in data.columns:
            self.data = data.sort_values(by=timestamp_column).reset_index(drop=True)
        else:
            self.data = data.copy()

        self.target_column = target_column
        self.text_column = text_column
        self.numerical_columns = numerical_columns

        # Separate feature matrix from targets
        self.y = self.data[self.target_column].astype(int).values
        drop_cols = [self.target_column]
        if timestamp_column in self.data.columns:
            drop_cols.append(timestamp_column)
        if "job_id" in self.data.columns:
            drop_cols.append("job_id")
        if "_id" in self.data.columns:
            drop_cols.append("_id")

        self.X = self.data.drop(columns=drop_cols)
        self.preprocessor = build_feature_pipeline(
            text_column=self.text_column,
            numerical_columns=self.numerical_columns,
        )

    def _objective(self, trial: optuna.Trial, n_splits: int = 5) -> float:
        """
        Optuna objective evaluated over temporal rolling splits.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        classifier = XGBClassifier(**params)
        candidate_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", classifier),
            ]
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_f1_scores: List[float] = []

        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            candidate_pipeline.fit(X_train, y_train)
            preds = candidate_pipeline.predict(X_val)
            fold_f1 = f1_score(y_val, preds, zero_division=0)
            fold_f1_scores.append(float(fold_f1))

        return float(np.mean(fold_f1_scores))

    def train_and_tune(
        self,
        n_trials: int = 25,
        n_splits: int = 4,
    ) -> Tuple[Pipeline, Dict[str, Any], float]:
        """
        Runs the hyperparameter optimization sweep and trains the final pipeline
        on the full dataset using the best discovered parameters.

        Args:
            n_trials: Number of Bayesian optimization iterations.
            n_splits: Number of temporal cross-validation folds.

        Returns:
            Tuple of (fitted_pipeline, best_hyperparameters, best_mean_f1).
        """
        logger.info(f"Initiating Optuna search across {n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, n_splits=n_splits), n_trials=n_trials)

        best_params = study.best_params
        best_f1 = study.best_value
        logger.info(f"Optimization finished. Best Temporal F1: {best_f1:.4f}")

        # Construct production pipeline with best parameters
        best_params.update(
            {
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }
        )
        final_classifier = XGBClassifier(**best_params)
        final_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", final_classifier),
            ]
        )

        logger.info("Fitting final production pipeline on all training records...")
        final_pipeline.fit(self.X, self.y)

        return final_pipeline, best_params, best_f1

    @staticmethod
    def save_artifact(pipeline: Pipeline, output_path: Optional[str] = None) -> str:
        """
        Serializes the end-to-end pipeline to disk.
        """
        if output_path is None:
            os.makedirs(settings.MODEL_ARTIFACTS_DIR, exist_ok=True)
            output_path = os.path.join(
                settings.MODEL_ARTIFACTS_DIR, f"{settings.MODEL_VERSION}.pkl"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)
        logger.info(f"Model artifact persisted to: {output_path}")
        return output_path
