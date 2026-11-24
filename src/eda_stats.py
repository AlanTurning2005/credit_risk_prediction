from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control, ks_2samp

from src.config import ID_COL, TARGET_COL
from src.preprocess import _normalize_cat_series
from src.woe import WoEEncoder


def _adjust_pvalues(p_values: Sequence[float]) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR adjustment safely."""
    arr = np.asarray(p_values, dtype=float)
    if len(arr) == 0:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    return false_discovery_control(arr, method="bh")


def compute_cramers_v(chi2_stat: float, n_obs: int, n_rows: int, n_cols: int) -> float:
    """Compute Cramér's V effect size in [0, 1] for a contingency table."""
    denom_dim = min(n_rows - 1, n_cols - 1)
    if n_obs <= 0 or denom_dim <= 0:
        return 0.0
    return float(np.sqrt(max(chi2_stat, 0.0) / (n_obs * denom_dim)))


def compute_ks_tests(
    df: pd.DataFrame,
    num_cols: Sequence[str],
    target_col: str = TARGET_COL,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run two-sample Kolmogorov-Smirnov tests comparing TARGET=0 vs TARGET=1 distributions."""
    y = df[target_col]
    mask_good = y == 0
    mask_bad = y == 1

    rows: list[dict[str, object]] = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        good_vals = s[mask_good].dropna()
        bad_vals = s[mask_bad].dropna()

        if len(good_vals) >= 2 and len(bad_vals) >= 2:
            res = ks_2samp(good_vals, bad_vals)
            stat = float(res.statistic)
            p_val = float(res.pvalue)
        else:
            stat = 0.0
            p_val = 1.0

        rows.append(
            {
                "feature": col,
                "test_type": "KS (2-sample)",
                "stat": stat,
                "p_value": p_val,
                "missing_rate_%": float(s.isna().mean() * 100.0),
                "missing_good_%": float(s[mask_good].isna().mean() * 100.0),
                "missing_bad_%": float(s[mask_bad].isna().mean() * 100.0),
                "median_good": float(good_vals.median()) if len(good_vals) > 0 else np.nan,
                "median_bad": float(bad_vals.median()) if len(bad_vals) > 0 else np.nan,
            }
        )

    cols = [
        "feature",
        "test_type",
        "stat",
        "p_value",
        "missing_rate_%",
        "missing_good_%",
        "missing_bad_%",
        "median_good",
        "median_bad",
        "p_adj",
        "significant",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["p_adj"] = _adjust_pvalues(out["p_value"].tolist())
    out["significant"] = out["p_adj"] < alpha
    return out.sort_values(["stat", "p_value"], ascending=[False, True]).reset_index(drop=True)[cols]


def compute_chi2_tests(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    target_col: str = TARGET_COL,
    alpha: float = 0.05,
    min_freq_pct: float = 0.001,
) -> pd.DataFrame:
    """Run Pearson Chi-square tests and compute Cramér's V for categorical features."""
    y = df[target_col]
    rows: list[dict[str, object]] = []

    for col in cat_cols:
        s = _normalize_cat_series(df[col])
        freqs = s.value_counts(normalize=True)
        rare_vals = freqs[freqs < min_freq_pct].index
        if len(rare_vals) > 0:
            s = s.where(~s.isin(rare_vals), "Other_Rare")

        ct = pd.crosstab(s, y)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            chi2, p_val, dof, min_expected, cramers_v = 0.0, 1.0, 0, float(ct.to_numpy().min()), 0.0
        else:
            chi2, p_val, dof, expected = chi2_contingency(ct)
            min_expected = float(expected.min())
            cramers_v = compute_cramers_v(float(chi2), int(ct.to_numpy().sum()), ct.shape[0], ct.shape[1])

        rows.append(
            {
                "feature": col,
                "test_type": "Chi-square",
                "stat": float(chi2),
                "cramers_v": cramers_v,
                "dof": int(dof),
                "min_expected_freq": min_expected,
                "p_value": float(p_val),
                "missing_rate_%": float(df[col].isna().mean() * 100.0),
            }
        )

    cols = [
        "feature",
        "test_type",
        "stat",
        "cramers_v",
        "dof",
        "min_expected_freq",
        "p_value",
        "missing_rate_%",
        "p_adj",
        "significant",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["p_adj"] = _adjust_pvalues(out["p_value"].tolist())
    out["significant"] = out["p_adj"] < alpha
    return out.sort_values(["cramers_v", "p_value"], ascending=[False, True]).reset_index(drop=True)[cols]


def summarize_statistical_tests(
    df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    target_col: str = TARGET_COL,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Combine KS and Chi-square summaries into a unified `feature | stat | p_value | significant` table."""
    ks_df = compute_ks_tests(df, num_cols, target_col=target_col, alpha=alpha)
    chi_df = compute_chi2_tests(df, cat_cols, target_col=target_col, alpha=alpha)

    parts: list[pd.DataFrame] = []
    if not ks_df.empty:
        parts.append(
            pd.DataFrame(
                {
                    "feature": ks_df["feature"],
                    "test_type": ks_df["test_type"],
                    "stat": ks_df["stat"],
                    "effect_size": ks_df["stat"],
                    "p_value": ks_df["p_value"],
                    "p_adj": ks_df["p_adj"],
                    "significant": ks_df["significant"],
                    "missing_rate_%": ks_df["missing_rate_%"],
                }
            )
        )
    if not chi_df.empty:
        parts.append(
            pd.DataFrame(
                {
                    "feature": chi_df["feature"],
                    "test_type": chi_df["test_type"],
                    "stat": chi_df["stat"],
                    "effect_size": chi_df["cramers_v"],
                    "p_value": chi_df["p_value"],
                    "p_adj": chi_df["p_adj"],
                    "significant": chi_df["significant"],
                    "missing_rate_%": chi_df["missing_rate_%"],
                }
            )
        )

    if not parts:
        return pd.DataFrame(
            columns=["feature", "test_type", "stat", "effect_size", "p_value", "p_adj", "significant", "missing_rate_%"]
        )

    combined = pd.concat(parts, ignore_index=True)
    combined["p_adj"] = _adjust_pvalues(combined["p_value"].tolist())
    combined["significant"] = combined["p_adj"] < alpha
    return combined.sort_values(["effect_size", "p_adj"], ascending=[False, True]).reset_index(drop=True)


def _classify_screening_status(
    col: str,
    missing_pct: float,
    top_share: float,
    iv: float,
    missing_cutoff: float,
    soft_missing_cutoff: float,
    constant_cutoff: float,
    iv_cutoff: float,
) -> str:
    if missing_pct > missing_cutoff:
        return f"Drop: Missing > {missing_cutoff:.0f}%"
    if top_share >= constant_cutoff:
        return f"Drop: Near-constant (>={constant_cutoff * 100:.0f}%)"
    if col.endswith(("_MEDI", "_MODE")):
        return "Drop: Redundant (_MEDI/_MODE)"
    if iv < iv_cutoff:
        return f"Drop: Low IV (<{iv_cutoff:.2f})"
    if missing_pct > soft_missing_cutoff:
        return f"Candidate (Missing {soft_missing_cutoff:.0f}-{missing_cutoff:.0f}%, High IV)"
    return f"Candidate (IV >= {iv_cutoff:.2f})"


def screen_raw_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
    missing_cutoff: float = 60.0,
    soft_missing_cutoff: float = 50.0,
    constant_cutoff: float = 0.99,
    iv_cutoff: float = 0.02,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Objectively screen all raw features by missingness, variance, redundancy, KS/Chi2, and IV."""
    feature_cols = [c for c in df.columns if c not in (id_col, target_col)]

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    stats_df = summarize_statistical_tests(df, num_cols, cat_cols, target_col=target_col, alpha=alpha)
    woe_encoder = WoEEncoder(
        num_cols=num_cols,
        cat_cols=cat_cols,
        binning="tree",
        max_bins=5,
        iv_threshold=iv_cutoff,
    )
    woe_encoder.fit(df[feature_cols], df[target_col])
    iv_map = dict(zip(woe_encoder.iv_summary_["feature"], woe_encoder.iv_summary_["iv"]))

    meta_rows: list[dict[str, object]] = []
    for col in feature_cols:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100.0)
        vc = s.fillna("__MISSING__").value_counts(normalize=True)
        top_share = float(vc.iloc[0]) if len(vc) > 0 else 1.0
        iv_val = float(iv_map.get(col, 0.0))
        meta_rows.append(
            {
                "feature": col,
                "near_constant": top_share >= constant_cutoff,
                "top_value_share_%": round(top_share * 100.0, 2),
                "iv": iv_val,
                "screening_status": _classify_screening_status(
                    col,
                    missing_pct,
                    top_share,
                    iv_val,
                    missing_cutoff,
                    soft_missing_cutoff,
                    constant_cutoff,
                    iv_cutoff,
                ),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    merged = stats_df.merge(meta_df, on="feature", how="left")
    return merged.sort_values(["iv", "effect_size"], ascending=[False, False]).reset_index(drop=True)
