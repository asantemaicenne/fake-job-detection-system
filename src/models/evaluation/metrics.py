from typing import Any, Dict, List, Sequence, Union
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Comprehensive performance assessment and statistical distribution drift monitoring."""

    @staticmethod
    def calculate_metrics(
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Union[np.ndarray, List[float]],
    ) -> Dict[str, Any]:
        """
        Computes standard discrimination, calibration, and threshold-dependent metrics.

        Args:
            y_true: Ground truth binary indicators (0 or 1).
            y_pred: Predicted class labels (0 or 1).
            y_prob: Posterior probabilities for class 1 (is_fake).

        Returns:
            Dictionary containing metrics and nested confusion matrix.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        precisions, recalls, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_auc = float(auc(recalls, precisions))
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        brier = float(brier_score_loss(y_true_arr, y_prob_arr))

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        tn, fp, fn, tp = map(int, cm.ravel())

        return {
            "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
        }

    @staticmethod
    def compute_pr_curve_data(
        y_true: Union[np.ndarray, List[int]],
        y_prob: Union[np.ndarray, List[float]],
        num_points: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Computes downsampled Precision-Recall curve coordinates for charting.
        """
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        indices = np.linspace(0, len(precisions) - 1, min(len(precisions), num_points), dtype=int)
        return {
            "precision": [round(float(p), 4) for p in precisions[indices]],
            "recall": [round(float(r), 4) for r in recalls[indices]],
        }

    @staticmethod
    def detect_prediction_drift(
        reference_probs: Sequence[float],
        current_probs: Sequence[float],
        alpha_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Performs a two-sample Kolmogorov-Smirnov test to detect distribution drift
        in model confidence outputs between training baselines and live inferences.

        Args:
            reference_probs: Baseline prediction probabilities from training/validation.
            current_probs: Observed inference probabilities over recent time window.
            alpha_threshold: Significance level below which the null hypothesis of identical
                             distributions is rejected.

        Returns:
            Dictionary containing test statistic, p-value, and drift indicator.
        """
        ks_result = ks_2samp(reference_probs, current_probs)
        statistic = float(ks_result.statistic)
        p_value = float(ks_result.pvalue)

        drift_detected = bool(p_value < alpha_threshold)

        return {
            "drift_detected": drift_detected,
            "ks_statistic": round(statistic, 5),
            "p_value": round(p_value, 5),
            "alpha_threshold": alpha_threshold,
            "reference_sample_size": len(reference_probs),
            "current_sample_size": len(current_probs),
        }
