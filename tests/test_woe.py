import numpy as np
import pandas as pd

from src.woe import WoEEncoder


def _make_synthetic_credit_df(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    # Strong numeric predictor (higher -> lower default probability)
    score = rng.uniform(0.05, 0.95, size=n)
    # Introduce 15% missingness with higher risk
    missing_mask = rng.random(n) < 0.15
    score_with_nan = score.copy()
    score_with_nan[missing_mask] = np.nan

    # Constant / zero-signal numeric feature (IV == 0.0 < 0.02)
    noise = rng.normal(0.0, 1.0, size=n)

    # Strong categorical predictor
    segment = rng.choice(["LowRiskTier", "MidRiskTier", "HighRiskTier"], size=n, p=[0.4, 0.4, 0.2])

    logit = np.where(missing_mask, 0.5, -3.5 * score + 0.8)
    logit += np.where(segment == "HighRiskTier", 1.2, np.where(segment == "LowRiskTier", -1.0, 0.0))
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series((rng.random(n) < prob).astype(int), name="TARGET")

    X = pd.DataFrame(
        {
            "strong_score": score_with_nan,
            "noise_num": noise,
            "segment": segment,
        }
    )
    return X, y


def test_woe_iv_selection_and_non_negative():
    X, y = _make_synthetic_credit_df()
    enc = WoEEncoder(
        num_cols=["strong_score", "noise_num"],
        cat_cols=["segment"],
        binning="tree",
        iv_threshold=0.02,
    )
    X_woe = enc.fit_transform(X, y)
    summary = enc.get_iv_summary()

    assert (summary["iv"] >= 0.0).all()
    assert "strong_score" in enc.selected_features_
    assert "segment" in enc.selected_features_
    assert "noise_num" not in enc.selected_features_
    assert list(X_woe.columns) == enc.selected_features_
    assert not X_woe.isna().any().any()
    assert np.isfinite(X_woe.to_numpy()).all()


def test_woe_missing_bin_and_unseen_category_handling():
    X, y = _make_synthetic_credit_df()
    enc = WoEEncoder(
        num_cols=["strong_score"],
        cat_cols=["segment"],
        binning="quantile",
        max_bins=5,
        iv_threshold=0.0,
    )
    enc.fit(X, y)

    woe_tbl = enc.get_woe_table("strong_score")
    assert "Missing" in woe_tbl["bin"].tolist()

    # Transform unseen category and NaN without crashing or producing NaN/inf
    X_new = pd.DataFrame(
        {
            "strong_score": [np.nan, 0.1, 0.9],
            "segment": ["UnseenSegment", None, "LowRiskTier"],
        }
    )
    X_new_woe = enc.transform(X_new)
    assert X_new_woe.shape == (3, 2)
    assert not X_new_woe.isna().any().any()
    assert np.isfinite(X_new_woe.to_numpy()).all()
    # High score should have higher WoE (safer) than low score under ln(P(Good)/P(Bad))
    assert X_new_woe.loc[2, "strong_score"] > X_new_woe.loc[1, "strong_score"]
