import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from configs.settings import settings

logger = logging.getLogger(__name__)


class EthicsAuditor:
    """Audit model fairness and parity across job subgroups."""

    @staticmethod
    def audit_demographic_error_rates(
        data: pd.DataFrame,
        y_true_col: str,
        y_pred_col: str,
        category_col: str,
    ) -> list[dict[str, Any]]:
        """Compute FPR/FNR by group for fairness reviews.

        Returns:
            List of breakdown summaries containing error rates per segment.
        """
        audit_results: list[dict[str, Any]] = []
        unique_groups = data[category_col].dropna().unique()

        for group in unique_groups:
            subset = data[data[category_col] == group]
            total_samples = len(subset)
            if total_samples < 5:
                continue

            y_true = subset[y_true_col].astype(int)
            y_pred = subset[y_pred_col].astype(int)

            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tp = int(((y_pred == 1) & (y_true == 1)).sum())

            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            audit_results.append(
                {
                    "category": str(group),
                    "sample_count": total_samples,
                    "false_positive_rate": round(fpr, 4),
                    "false_negative_rate": round(fnr, 4),
                    "true_positive_rate": round(
                        float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                        4,
                    ),
                }
            )

        return audit_results


class HumanInTheLoop:
    """Route uncertain predictions to a human-review queue."""

    def __init__(
        self,
        lower_threshold: float | None = None,
        upper_threshold: float | None = None,
    ) -> None:
        self.lower_threshold = (
            lower_threshold
            if lower_threshold is not None
            else settings.HITL_UNCERTAINTY_LOWER
        )
        self.upper_threshold = (
            upper_threshold
            if upper_threshold is not None
            else settings.HITL_UNCERTAINTY_UPPER
        )

    def evaluate(self, confidence_score: float) -> bool:
        """Return True when the confidence falls in the review band."""
        return self.lower_threshold <= confidence_score <= self.upper_threshold


class ExplainabilityEngine:
    """Extract SHAP values from the fitted XGBoost pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        background_sample: pd.DataFrame,
    ) -> None:
        """Initialize TreeExplainer using the preprocessing pipeline."""
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]

        transformed_background = self.preprocessor.transform(background_sample)
        # TreeExplainer optimized for tree ensembles. Use tree-path-dependent
        # mode so XGBoost categorical splits do not fail while still returning
        # stable per-feature attribution values.
        self.explainer = shap.TreeExplainer(
            self.classifier,
            data=transformed_background,
            feature_perturbation="tree_path_dependent",
        )
        self.feature_names = self._extract_feature_names()

    def _extract_feature_names(self) -> list[str]:
        """Extract text tokens and numeric column names from the transformer."""
        text_names: list[str] = []
        try:
            tfidf = self.preprocessor.named_transformers_[
                "text_branch"
            ].named_steps["tfidf"]
            text_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
        except (AttributeError, KeyError, TypeError):
            logger.debug("Text tokenizer feature names unavailable", exc_info=True)

        num_names: list[str] = []
        try:
            num_names = list(self.preprocessor.transformers_[1][2])
        except (AttributeError, IndexError, KeyError, TypeError):
            logger.debug("Numeric feature names unavailable", exc_info=True)

        return text_names + num_names

    @staticmethod
    def _fallback_attribution(instance_row: dict[str, Any]) -> dict[str, float]:
        """Return stable attribution values when SHAP is unavailable."""
        risk_features = {
            "has_payment_request": float(
                bool(instance_row.get("has_payment_request", False))
            ),
            "has_pii_request": float(
                bool(instance_row.get("has_pii_request", False))
            ),
            "salary_anomaly_score": float(
                instance_row.get("salary_anomaly_score", 0.0)
            ),
            "urgency_score": float(instance_row.get("urgency_score", 0.0)),
            "grammar_anomaly_score": float(
                instance_row.get("grammar_anomaly_score", 0.0)
            ),
            "is_generic_email": float(
                bool(instance_row.get("is_generic_email", False))
            ),
            "missing_company_url": float(
                bool(instance_row.get("missing_company_url", False))
            ),
            "poster_reputation_score": float(
                instance_row.get("poster_reputation_score", 1.0)
            ),
            "duplicate_count": float(instance_row.get("duplicate_count", 0)),
        }
        active_scores = {
            key: value for key, value in risk_features.items() if abs(value) > 0
        }
        return {
            k: round(v, 4)
            for k, v in sorted(
                active_scores.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:10]
        }

    def explain_instance(
        self,
        instance_df: pd.DataFrame,
        top_k: int = 10,
    ) -> dict[str, float]:
        """Generate signed attribution scores for the most influential features.

        Args:
            instance_df: Single row DataFrame containing raw input data.
            top_k: Number of most impactful features to return.

        Returns:
            Dictionary of top feature names mapped to their SHAP attribution value.
        """
        try:
            transformed_row = self.preprocessor.transform(instance_df)
            shap_values = self.explainer.shap_values(transformed_row)

            if hasattr(shap_values, "values"):
                row_shap = np.asarray(shap_values.values)
                if row_shap.ndim == 3:
                    row_shap = row_shap[0, 0]
                elif row_shap.ndim == 2:
                    row_shap = row_shap[0]
            elif isinstance(shap_values, list):
                if len(shap_values) > 1:
                    row_shap = np.asarray(shap_values[1][0])
                else:
                    row_shap = np.asarray(shap_values[0][0])
            else:
                row_shap = np.asarray(shap_values)

            if not isinstance(row_shap, np.ndarray) or row_shap.size == 0:
                raise ValueError("SHAP produced no feature values.")

            num_features = min(len(self.feature_names), len(row_shap))
            attributions = {
                self.feature_names[i]: float(row_shap[i])
                for i in range(num_features)
            }

            sorted_attributions = dict(
                sorted(
                    attributions.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:top_k]
            )
            if sorted_attributions:
                return {k: round(v, 4) for k, v in sorted_attributions.items()}
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            logger.warning(
                "SHAP explanation failed for instance; using heuristic "
                "attribution fallback: %s",
                exc,
            )

        instance_row = (
            instance_df.iloc[0].to_dict() if not instance_df.empty else {}
        )
        return self._fallback_attribution(instance_row)
