"""Core preprocessing, metrics, and labeling utilities for Case 05."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


FUTURE_OUTCOMES = [
    "revenue_next_90d",
    "margin_next_90d",
    "churn_next_90d",
    "product_adoption_next_90d",
]


@dataclass
class FrozenPreprocessor:
    """Preprocessor whose parameters are learned only from the development quarter."""

    features: list[str]
    log_features: list[str]
    scaler_kind: str = "robust"
    sentinel_columns: dict[str, float] = field(
        default_factory=lambda: {"days_since_last_login": 9999.0}
    )
    missing_indicator_features: list[str] = field(
        default_factory=lambda: [
            "credit_utilization",
            "days_since_last_login",
            "push_open_rate_90d",
            "email_open_rate_90d",
        ]
    )

    medians_: dict[str, float] = field(default_factory=dict, init=False)
    lower_: dict[str, float] = field(default_factory=dict, init=False)
    upper_: dict[str, float] = field(default_factory=dict, init=False)
    output_columns_: list[str] = field(default_factory=list, init=False)
    scaler_: object | None = field(default=None, init=False)

    def _prepare(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        x = frame[self.features].copy()
        for col, sentinel in self.sentinel_columns.items():
            if col in x:
                x[col] = x[col].mask(x[col].eq(sentinel))

        for col in self.missing_indicator_features:
            if col in x:
                x[f"{col}__missing"] = x[col].isna().astype("float32")

        if fit:
            self.medians_ = {}
            for col in self.features:
                if col == "credit_utilization":
                    self.medians_[col] = 0.0
                else:
                    value = float(x[col].median())
                    self.medians_[col] = 0.0 if np.isnan(value) else value

        for col in self.features:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(self.medians_[col])

        for col in self.log_features:
            if col in x:
                x[col] = np.log1p(x[col].clip(lower=0))

        continuous = [c for c in self.features if x[c].nunique(dropna=False) > 2]
        if fit:
            self.lower_ = x[continuous].quantile(0.01).astype(float).to_dict()
            self.upper_ = x[continuous].quantile(0.99).astype(float).to_dict()
        for col in continuous:
            x[col] = x[col].clip(self.lower_[col], self.upper_[col])

        x = x.astype("float32")
        if fit:
            self.output_columns_ = x.columns.tolist()
        return x[self.output_columns_]

    def fit(self, frame: pd.DataFrame) -> "FrozenPreprocessor":
        x = self._prepare(frame, fit=True)
        scaler_cls = RobustScaler if self.scaler_kind == "robust" else StandardScaler
        self.scaler_ = scaler_cls()
        self.scaler_.fit(x)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        x = self._prepare(frame, fit=False)
        return self.scaler_.transform(x).astype("float32")

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def internal_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute comparable internal metrics without quadratic work on the full base."""
    labels = np.asarray(labels)
    n_clusters = np.unique(labels[labels >= 0]).size
    if n_clusters < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    rng = np.random.default_rng(random_state)
    valid = np.flatnonzero(labels >= 0)
    idx = rng.choice(valid, size=min(sample_size, len(valid)), replace=False)
    return {
        "silhouette": float(silhouette_score(x[idx], labels[idx])),
        "davies_bouldin": float(davies_bouldin_score(x[idx], labels[idx])),
        "calinski_harabasz": float(calinski_harabasz_score(x[idx], labels[idx])),
    }


def minimum_cluster_share(labels: Iterable[int]) -> float:
    shares = pd.Series(labels).value_counts(normalize=True)
    return float(shares.min()) if len(shares) else np.nan


def fit_rfm_cutoffs(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Freeze quartile cutoffs from the development snapshot."""
    return {
        "recency_days": frame["recency_days"].quantile([0.25, 0.5, 0.75]).tolist(),
        "frequency_90d": frame["frequency_90d"].quantile([0.25, 0.5, 0.75]).tolist(),
        "monetary_90d": frame["monetary_90d"].quantile([0.25, 0.5, 0.75]).tolist(),
    }


def apply_rfm(frame: pd.DataFrame, cutoffs: dict[str, list[float]]) -> pd.Series:
    recency = 5 - np.digitize(frame["recency_days"], cutoffs["recency_days"], right=True)
    frequency = 1 + np.digitize(frame["frequency_90d"], cutoffs["frequency_90d"], right=True)
    monetary = 1 + np.digitize(frame["monetary_90d"], cutoffs["monetary_90d"], right=True)
    total = recency + frequency + monetary
    result = np.full(len(frame), "Regular", dtype=object)
    result[total >= 11] = "Champions"
    result[(total >= 9) & (total <= 10)] = "Loyal"
    result[(total >= 7) & (total <= 8)] = "Potential"
    result[(recency <= 2) & (frequency >= 3)] = "At Risk"
    result[total <= 5] = "Hibernating"
    return pd.Series(result, index=frame.index, name="rfm_segment")


def semantic_cluster_map(profile: pd.DataFrame) -> dict[int, str]:
    """Assign unique, business-readable names from frozen Q1 cluster profiles."""
    work = profile.set_index("cluster_id")
    z = (work - work.mean()) / work.std(ddof=0).replace(0, 1)
    remaining = set(z.index.tolist())
    mapping: dict[int, str] = {}

    def pick(score: pd.Series, label: str) -> None:
        eligible = score.loc[list(remaining)]
        cluster = int(eligible.idxmax())
        mapping[cluster] = label
        remaining.remove(cluster)

    pick(z["merchant_gmv_90d"] + z["merchant_txn_count_90d"], "Comerciantes Ativos")
    pick(z["investment_balance"] + z["avg_balance_90d"] + z["margin_90d"], "Patrimônio & Valor")
    pick(z["credit_utilization"] + z["card_spend_90d"], "Crédito Intensivo")
    pick(z["app_sessions_90d"] + z["pix_count_90d"] + z["campaign_click_rate_90d"], "Digitais Engajados")
    pick(z["recency_days"] - z["frequency_90d"] - z["app_sessions_90d"], "Baixo Engajamento")
    for cluster in remaining:
        mapping[int(cluster)] = "Relacionamento Essencial"
    return mapping


def confidence_interval_mean(series: pd.Series) -> tuple[float, float]:
    n = int(series.notna().sum())
    mean = float(series.mean())
    if n <= 1:
        return mean, mean
    se = float(series.std(ddof=1) / np.sqrt(n))
    return mean - 1.96 * se, mean + 1.96 * se

