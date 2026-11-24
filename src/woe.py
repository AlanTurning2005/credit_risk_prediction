from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAL_COLS, DATA_PROCESSED_DIR, NUMERICAL_COLS, TARGET_COL
from src.preprocess import _normalize_cat_series, engineer_features


def classify_iv_strength(iv: float) -> str:
    """Classify Information Value according to standard credit scoring rules."""
    if iv < 0.02:
        return "Useless (<0.02)"
    if iv < 0.10:
        return "Weak (0.02-0.10)"
    if iv < 0.30:
        return "Medium (0.10-0.30)"
    if iv < 0.50:
        return "Strong (0.30-0.50)"
    return "Suspicious (>0.50)"


def _check_monotonic(values: Sequence[float]) -> bool:
    """Return True if a numeric-bin WoE sequence (len >= 2) is monotonically non-decreasing or non-increasing."""
    if len(values) < 2:
        return True
    arr = np.asarray(values, dtype=float)
    diffs = np.diff(arr)
    return bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))


@dataclass
class _NumericBinSpec:
    inner_edges: np.ndarray
    bin_labels: list[str]
    bin_woe: np.ndarray
    missing_woe: float
    table: pd.DataFrame
    iv: float
    is_monotonic: bool


@dataclass
class _CategoricalBinSpec:
    kept_categories: set[str]
    has_other: bool
    cat_woe: dict[str, float]
    missing_woe: float
    other_woe: float
    table: pd.DataFrame
    iv: float


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence (WoE) encoder and Information Value (IV) selector.

    Sign convention:
        WoE_i = ln(P(Good | bin_i) / P(Bad | bin_i))
        Positive WoE indicates lower default risk than population average;
        Negative WoE indicates higher default risk.
    """

    def __init__(
        self,
        num_cols: Sequence[str] = tuple(NUMERICAL_COLS),
        cat_cols: Sequence[str] = tuple(CATEGORICAL_COLS),
        binning: Literal["tree", "quantile"] = "tree",
        max_bins: int = 6,
        min_bin_pct: float = 0.05,
        min_cat_pct: float = 0.01,
        smoothing: float = 0.5,
        iv_threshold: float = 0.02,
        select_by_iv: bool = True,
        random_state: int = 42,
    ) -> None:
        if binning not in ("tree", "quantile"):
            raise ValueError("binning must be 'tree' or 'quantile'.")
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.binning = binning
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.min_cat_pct = min_cat_pct
        self.smoothing = smoothing
        self.iv_threshold = iv_threshold
        self.select_by_iv = select_by_iv
        self.random_state = random_state

    def _compute_woe_table(
        self,
        bin_series: pd.Series,
        y: pd.Series,
        ordered_labels: Sequence[str],
    ) -> tuple[pd.DataFrame, float]:
        """Compute bin counts, Laplace-smoothed distributions, WoE, and IV."""
        df_tmp = pd.DataFrame({"bin": bin_series.astype(str), "target": y.to_numpy()})
        grouped = (
            df_tmp.groupby("bin", observed=False)["target"]
            .agg(total="count", bad="sum")
            .reindex(ordered_labels, fill_value=0)
        )
        grouped["good"] = grouped["total"] - grouped["bad"]

        total_good = float(grouped["good"].sum())
        total_bad = float(grouped["bad"].sum())
        k = max(len(grouped), 1)
        alpha = float(self.smoothing)

        dist_good = (grouped["good"] + alpha) / (total_good + alpha * k)
        dist_bad = (grouped["bad"] + alpha) / (total_bad + alpha * k)

        grouped["event_rate"] = np.where(grouped["total"] > 0, grouped["bad"] / grouped["total"], 0.0)
        grouped["dist_good"] = dist_good
        grouped["dist_bad"] = dist_bad
        grouped["woe"] = np.log(dist_good / dist_bad)
        grouped["iv_contrib"] = (dist_good - dist_bad) * grouped["woe"]

        table = grouped.reset_index().rename(columns={"index": "bin"})
        total_iv = float(table["iv_contrib"].sum())
        return table, total_iv

    def _find_numeric_inner_edges(self, s_valid: pd.Series, y_valid: pd.Series) -> np.ndarray:
        """Compute sorted split thresholds for non-missing values of a numeric feature."""
        n_unique = s_valid.nunique()
        if n_unique <= 1:
            return np.array([], dtype=float)

        # Preserve exact discrete levels for binary/low-cardinality flags (e.g., DAYS_EMPLOYED_ANOM)
        if n_unique <= min(self.max_bins, 5):
            uniq_sorted = np.sort(s_valid.unique().astype(float))
            return (uniq_sorted[:-1] + uniq_sorted[1:]) / 2.0

        min_leaf = max(int(len(s_valid) * self.min_bin_pct), 20)
        if min_leaf * 2 > len(s_valid):
            min_leaf = max(1, len(s_valid) // 4)

        if self.binning == "tree":
            max_depth = max(1, int(np.ceil(np.log2(max(self.max_bins, 2)))))
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                max_leaf_nodes=self.max_bins,
                min_samples_leaf=min_leaf,
                random_state=self.random_state,
            )
            tree.fit(s_valid.to_numpy(dtype=float).reshape(-1, 1), y_valid.to_numpy(dtype=int))
            thresholds = tree.tree_.threshold
            valid_thresh = np.sort(np.unique(thresholds[thresholds != -2.0]))
            if len(valid_thresh) > 0:
                return valid_thresh

        # Quantile binning (also used as fallback if tree finds 0 splits)
        quantiles = np.quantile(s_valid.to_numpy(dtype=float), np.linspace(0.0, 1.0, self.max_bins + 1))
        candidate_edges = np.unique(quantiles[:-1])
        kept_edges = [
            float(edge)
            for edge in candidate_edges
            if int((s_valid <= edge).sum()) >= min_leaf and int((s_valid > edge).sum()) >= min_leaf
        ]
        return np.asarray(sorted(kept_edges), dtype=float)

    def _assign_numeric_bin_labels(self, s: pd.Series, inner_edges: np.ndarray) -> tuple[pd.Series, list[str]]:
        """Assign interval string labels + 'Missing' label consistently."""
        n_bins = len(inner_edges) + 1
        interval_labels: list[str] = []
        bounds = [-np.inf, *inner_edges.tolist(), np.inf]
        for i in range(n_bins):
            left = "-inf" if np.isneginf(bounds[i]) else f"{bounds[i]:.4g}"
            right = "+inf" if np.isposinf(bounds[i + 1]) else f"{bounds[i + 1]:.4g}"
            interval_labels.append(f"({left}, {right}]")

        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        is_missing = np.isnan(vals)
        assigned = np.empty(len(vals), dtype=object)
        if (~is_missing).any():
            idx = np.searchsorted(inner_edges, vals[~is_missing], side="left")
            assigned[~is_missing] = [interval_labels[i] for i in idx]
        if is_missing.any():
            assigned[is_missing] = "Missing"

        ordered_labels = list(interval_labels)
        if is_missing.any():
            ordered_labels.append("Missing")
        return pd.Series(assigned, index=s.index), ordered_labels

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        y_series = pd.Series(y).reset_index(drop=True).astype(int)
        if len(X) != len(y_series):
            raise ValueError("X and y must have the same number of rows.")

        needed = set(self.num_cols) | set(self.cat_cols)
        if not needed.issubset(X.columns):
            X_work = engineer_features(X, validate=False, keep_target=False).reset_index(drop=True)
        else:
            X_work = X.reset_index(drop=True).copy()

        self.numeric_specs_: dict[str, _NumericBinSpec] = {}
        self.categorical_specs_: dict[str, _CategoricalBinSpec] = {}
        summary_rows: list[dict[str, object]] = []

        for col in self.num_cols:
            s = pd.to_numeric(X_work[col], errors="coerce")
            non_missing = s.notna()
            inner_edges = (
                self._find_numeric_inner_edges(s[non_missing], y_series[non_missing])
                if non_missing.any()
                else np.array([], dtype=float)
            )
            binned, ordered_labels = self._assign_numeric_bin_labels(s, inner_edges)
            table, iv = self._compute_woe_table(binned, y_series, ordered_labels)

            woe_lookup = dict(zip(table["bin"], table["woe"]))
            interval_labels = [lbl for lbl in ordered_labels if lbl != "Missing"]
            bin_woe = np.array([woe_lookup[lbl] for lbl in interval_labels], dtype=float)
            missing_woe = float(woe_lookup.get("Missing", 0.0))
            is_mono = _check_monotonic(bin_woe)

            self.numeric_specs_[col] = _NumericBinSpec(
                inner_edges=inner_edges,
                bin_labels=interval_labels,
                bin_woe=bin_woe,
                missing_woe=missing_woe,
                table=table,
                iv=iv,
                is_monotonic=is_mono,
            )
            summary_rows.append(
                {
                    "feature": col,
                    "feature_type": "numerical",
                    "n_bins": len(ordered_labels),
                    "iv": iv,
                    "iv_strength": classify_iv_strength(iv),
                    "is_monotonic": is_mono,
                    "selected": iv >= self.iv_threshold,
                }
            )

        for col in self.cat_cols:
            s = _normalize_cat_series(X_work[col])
            freqs = s.value_counts(normalize=True)
            kept = set(freqs[freqs >= self.min_cat_pct].index.tolist())
            # Keep Missing separate when present in training
            if "Missing" in freqs.index:
                kept.add("Missing")

            pooled = s.where(s.isin(kept), "Other")
            has_other = bool((pooled == "Other").any())
            ordered_cats = sorted(pooled.unique().tolist())

            table, iv = self._compute_woe_table(pooled, y_series, ordered_cats)
            cat_woe = dict(zip(table["bin"], table["woe"]))
            missing_woe = float(cat_woe.get("Missing", 0.0))
            other_woe = float(cat_woe.get("Other", 0.0))

            self.categorical_specs_[col] = _CategoricalBinSpec(
                kept_categories=kept,
                has_other=has_other,
                cat_woe=cat_woe,
                missing_woe=missing_woe,
                other_woe=other_woe,
                table=table,
                iv=iv,
            )
            summary_rows.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "n_bins": len(ordered_cats),
                    "iv": iv,
                    "iv_strength": classify_iv_strength(iv),
                    "is_monotonic": None,
                    "selected": iv >= self.iv_threshold,
                }
            )

        self.iv_summary_ = (
            pd.DataFrame(summary_rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        self.selected_features_ = self.iv_summary_.loc[
            self.iv_summary_["selected"], "feature"
        ].tolist()
        return self

    def transform(self, X: pd.DataFrame, select_by_iv: bool | None = None) -> pd.DataFrame:
        if not hasattr(self, "iv_summary_"):
            raise RuntimeError("WoEEncoder must be fitted before calling transform().")

        use_iv_filter = self.select_by_iv if select_by_iv is None else select_by_iv
        cols_to_output = (
            list(self.selected_features_)
            if use_iv_filter
            else list(self.num_cols) + list(self.cat_cols)
        )

        needed = set(self.num_cols) | set(self.cat_cols)
        if not needed.issubset(X.columns):
            X_work = engineer_features(X, validate=False, keep_target=False)
        else:
            X_work = X

        out = pd.DataFrame(index=X_work.index)
        for col in cols_to_output:
            if col in self.numeric_specs_:
                spec = self.numeric_specs_[col]
                vals = pd.to_numeric(X_work[col], errors="coerce").to_numpy(dtype=float)
                res = np.full(len(vals), spec.missing_woe, dtype=float)
                valid_mask = ~np.isnan(vals)
                if valid_mask.any():
                    idx = np.searchsorted(spec.inner_edges, vals[valid_mask], side="left")
                    res[valid_mask] = spec.bin_woe[idx]
                out[col] = res
            elif col in self.categorical_specs_:
                spec = self.categorical_specs_[col]
                s = _normalize_cat_series(X_work[col])
                fallback = spec.other_woe if spec.has_other else 0.0
                out[col] = s.map(spec.cat_woe).fillna(fallback).astype(float)
            else:
                raise KeyError(f"Unknown feature '{col}' in WoEEncoder.")

        return out

    def get_iv_summary(self) -> pd.DataFrame:
        if not hasattr(self, "iv_summary_"):
            raise RuntimeError("WoEEncoder must be fitted first.")
        return self.iv_summary_.copy()

    def get_woe_table(self, feature: str) -> pd.DataFrame:
        if not hasattr(self, "iv_summary_"):
            raise RuntimeError("WoEEncoder must be fitted first.")
        if feature in self.numeric_specs_:
            return self.numeric_specs_[feature].table.copy()
        if feature in self.categorical_specs_:
            return self.categorical_specs_[feature].table.copy()
        raise KeyError(f"Feature '{feature}' was not fitted by WoEEncoder.")

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "WoEEncoder":
        return joblib.load(path)


if __name__ == "__main__":
    train_csv = DATA_PROCESSED_DIR / "train.csv"
    valid_csv = DATA_PROCESSED_DIR / "valid.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Run `python -m src.preprocess` first to generate {train_csv}.")

    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)

    X_train, y_train = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL]
    X_valid = df_valid.drop(columns=[TARGET_COL])

    for strategy in ("tree", "quantile"):
        print(f"\n=== WoE & IV SUMMARY ({strategy.upper()} BINNING) ===")
        encoder = WoEEncoder(binning=strategy, iv_threshold=0.02)
        encoder.fit(X_train, y_train)
        summary = encoder.get_iv_summary()
        print(summary.to_string(index=False, formatters={"iv": lambda v: f"{v:.4f}"}))
        print(f"Selected features (IV >= 0.02): {len(encoder.selected_features_)} / {len(summary)}")

        X_valid_woe = encoder.transform(X_valid)
        assert not X_valid_woe.isna().any().any(), "Unexpected NaN in WoE-transformed valid set!"
        assert np.isfinite(X_valid_woe.to_numpy()).all(), "Unexpected inf in WoE-transformed valid set!"
        print(f"Valid transformed shape: {X_valid_woe.shape} (0 NaN, 0 inf)")

        top_feat = summary.iloc[0]["feature"]
        print(f"\nTop feature WoE breakdown ({top_feat}):")
        print(encoder.get_woe_table(str(top_feat)).to_string(index=False))
