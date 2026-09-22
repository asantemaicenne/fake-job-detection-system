import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from configs.settings import settings

logger = logging.getLogger(__name__)


class EthicsAuditor:
    """
    Audits algorithmic fairness and calculates parity metrics across subcategories
    (e.g., job source, posting category, employment type).
    """

    @staticmethod
    def audit_demographic_error_rates(
        data: pd.DataFrame,
        y_true_col: str,
        y_pred_col: str,
        category_col: str,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates False Positive Rates (FPR) and False Negative Rates (FNR) per group.
        Disparate impact occurs when certain segments experience disproportionate false bans.

        Returns:
            List of breakdown summaries containing error rates per segment.
        """
        audit_results: List[Dict[str, Any]] = []
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
                    "true_positive_rate": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
                }
            )

        return audit_results


class HumanInTheLoop:
    """
    Determines whether a prediction requires manual human review based on uncertainty bands.
    """

    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
    ) -> None:
        self.lower_threshold = (
            lower_threshold if lower_threshold is not None else settings.HITL_UNCERTAINTY_LOWER
        )
        self.upper_threshold = (
            upper_threshold if upper_threshold is not None else settings.HITL_UNCERTAINTY_UPPER
        )

    def evaluate(self, confidence_score: float) -> bool:
        """
        Returns True if the posterior probability falls in the uncertain decision boundary.
        """
        return self.lower_threshold <= confidence_score <= self.upper_threshold


class ExplainabilityEngine:
    """
    Extracts SHAP (SHapley Additive exPlanations) values from the fitted XGBoost
    pipeline to satisfy the GDPR right-to-explanation requirement.
    """

    def __init__(self, pipeline: Pipeline, background_sample: pd.DataFrame) -> None:
        """
        Initializes TreeExplainer using the transformer preprocessor and booster.
        """
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]

        transformed_background = self.preprocessor.transform(background_sample)
        # TreeExplainer optimized for tree ensembles
        self.explainer = shap.TreeExplainer(self.classifier, data=transformed_background)
        self.feature_names = self._extract_feature_names()

    def _extract_feature_names(self) -> List[str]:
        """Extracts text vocabulary names and numerical feature labels from the ColumnTransformer."""
        text_names: List[str] = []
        try:
            tfidf = self.preprocessor.named_transformers_["text_branch"].named_steps["tfidf"]
            text_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
        except Exception:
            pass

        num_names: List[str] = []
        try:
            num_names = list(self.preprocessor.transformers_[1][2])
        except Exception:
            pass

        return text_names + num_names

    def explain_instance(
        self,
        instance_df: pd.DataFrame,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Generates signed attribution scores for the most influential features.

        Args:
            instance_df: Single row DataFrame containing raw input data.
            top_k: Number of most impactful features to return.

        Returns:
            Dictionary of top feature names mapped to their SHAP attribution value.
        """
        transformed_row = self.preprocessor.transform(instance_df)
        shap_values = self.explainer.shap_values(transformed_row)

        if isinstance(shap_values, list):
            row_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            row_shap = shap_values[0]
        else:
            row_shap = shap_values

        num_features = min(len(self.feature_names), len(row_shap))
        attributions = {
            self.feature_names[i]: float(row_shap[i])
            for i in range(num_features)
        }

        # Sort by absolute magnitude of importance
        sorted_attributions = dict(
            sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        )

        return {k: round(v, 4) for k, v in sorted_attributions.items()}
