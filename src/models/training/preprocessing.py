import re
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer to normalize and sanitize raw job description text
    prior to n-gram tokenization and TF-IDF extraction.
    """

    def __init__(self) -> None:
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
        self.whitespace_pattern = re.compile(r"\s+")

    def fit(self, X: pd.Series, y: Optional[np.ndarray] = None) -> "TextCleaner":
        """Fit stub to satisfy scikit-learn transformer protocol."""
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Cleans HTML markup, hyperlinks, non-alphanumeric symbols, and redundant whitespace.
        
        Args:
            X: Pandas Series of text strings.

        Returns:
            Pandas Series of normalized lowercase strings.
        """
        def clean_record(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = self.html_pattern.sub(" ", text)
            text = self.url_pattern.sub(" ", text)
            text = self.special_char_pattern.sub(" ", text)
            text = self.whitespace_pattern.sub(" ", text)
            return text.lower().strip()

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str).apply(clean_record)
        return pd.Series(X).astype(str).apply(clean_record)


def build_feature_pipeline(
    text_column: str = "description",
    numerical_columns: Optional[List[str]] = None,
    max_tfidf_features: int = 5000,
) -> ColumnTransformer:
    """
    Constructs a dual-branch ColumnTransformer combining NLP TF-IDF features
    and scaled numerical/heuristic features.

    Args:
        text_column: Name of the raw job description column.
        numerical_columns: List of tabular numeric/boolean feature names.
        max_tfidf_features: Upper bound for TF-IDF vocabulary dictionary.

    Returns:
        ColumnTransformer: Preprocessing transformer ready for an estimator pipeline.
    """
    if numerical_columns is None:
        numerical_columns = [
            "has_payment_request",
            "has_pii_request",
            "salary_anomaly_score",
            "urgency_score",
            "grammar_anomaly_score",
            "is_generic_email",
            "missing_company_url",
            "poster_reputation_score",
            "duplicate_count",
        ]

    text_pipeline = Pipeline(
        steps=[
            ("cleaner", TextCleaner()),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_tfidf_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", RobustScaler(with_centering=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text_branch", text_pipeline, text_column),
            ("num_branch", numerical_pipeline, numerical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor
