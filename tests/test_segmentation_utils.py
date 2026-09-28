import numpy as np
import pandas as pd

from src.segmentation_utils import FUTURE_OUTCOMES, FrozenPreprocessor, apply_rfm, fit_rfm_cutoffs


def _sample_frame():
    return pd.DataFrame(
        {
            "has_credit_card": [0, 1, 1, 0],
            "credit_utilization": [np.nan, 0.2, 0.8, np.nan],
            "days_since_last_login": [9999, 2, 20, 5],
            "push_open_rate_90d": [np.nan, 0.2, 0.5, 0.1],
            "email_open_rate_90d": [0.1, np.nan, 0.3, 0.0],
            "recency_days": [1, 10, 30, 90],
            "frequency_90d": [30, 15, 5, 1],
            "monetary_90d": [3000, 1000, 200, 10],
        }
    )


def test_future_outcomes_are_not_preprocessing_features():
    features = ["has_credit_card", "credit_utilization", "days_since_last_login"]
    assert set(features).isdisjoint(FUTURE_OUTCOMES)


def test_preprocessor_adds_missing_indicators_and_removes_sentinel():
    frame = _sample_frame()
    features = [
        "has_credit_card",
        "credit_utilization",
        "days_since_last_login",
        "push_open_rate_90d",
        "email_open_rate_90d",
    ]
    prep = FrozenPreprocessor(features, ["days_since_last_login"])
    transformed = prep.fit_transform(frame)
    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] == len(features) + 4
    assert np.isfinite(transformed).all()
    assert prep.medians_["credit_utilization"] == 0.0


def test_rfm_rules_are_frozen_and_reusable():
    frame = _sample_frame()
    cutoffs = fit_rfm_cutoffs(frame)
    first = apply_rfm(frame, cutoffs)
    second = apply_rfm(frame.copy(), cutoffs)
    assert first.equals(second)
    assert set(first).issubset({"Champions", "Loyal", "Potential", "At Risk", "Hibernating", "Regular"})

