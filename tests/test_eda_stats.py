import numpy as np
import pandas as pd

from src.eda_stats import (
    compute_chi2_tests,
    compute_ks_tests,
    screen_raw_features,
    summarize_statistical_tests,
)


def test_ks_and_chi2_distinguish_signal_from_noise():
    rng = np.random.default_rng(123)
    n = 1000
    y = np.array([0] * 800 + [1] * 200)

    # Deterministic balanced noise category (exact 50/50 within each class -> Chi2 = 0, p = 1.0)
    cat_noise = np.tile(["X", "Y"], n // 2)

    df = pd.DataFrame(
        {
            "TARGET": y,
            "num_signal": np.concatenate([rng.normal(0.7, 0.1, 800), rng.normal(0.3, 0.1, 200)]),
            "num_noise": rng.normal(0.0, 1.0, n),
            "cat_signal": np.concatenate(
                [rng.choice(["A", "B"], 800, p=[0.85, 0.15]), rng.choice(["A", "B"], 200, p=[0.2, 0.8])]
            ),
            "cat_noise": cat_noise,
        }
    )

    ks_df = compute_ks_tests(df, ["num_signal", "num_noise"])
    assert ks_df.loc[ks_df["feature"] == "num_signal", "significant"].iloc[0]
    assert ks_df.loc[ks_df["feature"] == "num_signal", "stat"].iloc[0] > 0.7
    assert not ks_df.loc[ks_df["feature"] == "num_noise", "significant"].iloc[0]

    chi_df = compute_chi2_tests(df, ["cat_signal", "cat_noise"])
    assert chi_df.loc[chi_df["feature"] == "cat_signal", "significant"].iloc[0]
    assert chi_df.loc[chi_df["feature"] == "cat_signal", "cramers_v"].iloc[0] > 0.4
    assert not chi_df.loc[chi_df["feature"] == "cat_noise", "significant"].iloc[0]

    summary = summarize_statistical_tests(df, ["num_signal", "num_noise"], ["cat_signal", "cat_noise"])
    assert {"feature", "stat", "p_value", "p_adj", "significant"}.issubset(summary.columns)
    assert len(summary) == 4



def test_screen_raw_features_objective_funnel():
    rng = np.random.default_rng(7)
    n = 1500
    y = np.array([0] * 1200 + [1] * 300)
    strong = np.concatenate([rng.normal(0.7, 0.1, 1200), rng.normal(0.3, 0.1, 300)])
    high_miss = np.full(n, np.nan)
    high_miss[:200] = rng.normal(0, 1, 200)  # >85% missing
    near_const = np.zeros(n)
    near_const[0] = 1.0  # >99.9% zero

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": y,
            "STRONG_AVG": strong,
            "STRONG_MEDI": strong + 0.001,
            "HIGH_MISS": high_miss,
            "NEAR_CONST": near_const,
        }
    )

    res = screen_raw_features(df)
    status_map = dict(zip(res["feature"], res["screening_status"]))

    assert status_map["HIGH_MISS"].startswith("Drop: Missing")
    assert status_map["NEAR_CONST"].startswith("Drop: Near-constant")
    assert status_map["STRONG_MEDI"] == "Drop: Redundant (_MEDI/_MODE)"
    assert status_map["STRONG_AVG"].startswith("Candidate")
