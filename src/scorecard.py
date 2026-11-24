from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import DATA_PROCESSED_DIR, TARGET_COL
from src.woe import WoEEncoder


class WoEScorecard(BaseEstimator, ClassifierMixin):
    """Traditional Credit Risk Scorecard combining WoE binning and Logistic Regression.

    Uses WoEEncoder's convention WoE_i = ln(P(Good | bin_i) / P(Bad | bin_i)):
    higher WoE means lower default risk, so LogisticRegression coefficients (modeling
    P(Bad = 1)) are expected to be negative unless multicollinearity flips a weak feature.
    """

    def __init__(
        self,
        binning: Literal["tree", "quantile"] = "tree",
        max_bins: int = 6,
        min_bin_pct: float = 0.05,
        min_cat_pct: float = 0.01,
        smoothing: float = 0.5,
        iv_threshold: float = 0.02,
        C: float = 1.0,
        max_iter: int = 1000,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        random_state: int = 42,
    ) -> None:
        if pdo <= 0:
            raise ValueError("pdo must be positive.")
        if base_odds <= 0:
            raise ValueError("base_odds (Good:Bad) must be positive.")

        self.binning = binning
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.min_cat_pct = min_cat_pct
        self.smoothing = smoothing
        self.iv_threshold = iv_threshold
        self.C = C
        self.max_iter = max_iter
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.random_state = random_state

    def _compute_pdo_constants(
        self,
        pdo: float | None = None,
        base_score: float | None = None,
        base_odds: float | None = None,
    ) -> tuple[float, float]:
        """Return (factor, offset) for Score = offset - factor * ln(Odds_bad)."""
        use_pdo = float(self.pdo if pdo is None else pdo)
        use_base_score = float(self.base_score if base_score is None else base_score)
        use_base_odds = float(self.base_odds if base_odds is None else base_odds)
        if use_pdo <= 0 or use_base_odds <= 0:
            raise ValueError("pdo and base_odds must be positive.")

        factor = use_pdo / np.log(2.0)
        # At Score = base_score, Good:Bad = base_odds => Odds_bad = 1 / base_odds.
        offset = use_base_score - factor * np.log(use_base_odds)
        return factor, offset

    def _require_fitted(self) -> None:
        if not hasattr(self, "lr_") or not hasattr(self, "woe_encoder_"):
            raise RuntimeError("WoEScorecard must be fitted before calling this method.")

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "WoEScorecard":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        y_arr = np.asarray(y, dtype=int)
        if len(X) != len(y_arr):
            raise ValueError("X and y must have the same number of rows.")

        self.woe_encoder_ = WoEEncoder(
            binning=self.binning,
            max_bins=self.max_bins,
            min_bin_pct=self.min_bin_pct,
            min_cat_pct=self.min_cat_pct,
            smoothing=self.smoothing,
            iv_threshold=self.iv_threshold,
            select_by_iv=True,
            random_state=self.random_state,
        )
        X_woe = self.woe_encoder_.fit_transform(X, y_arr)
        self.selected_features_ = list(self.woe_encoder_.selected_features_)
        if not self.selected_features_:
            raise RuntimeError(
                f"No features passed iv_threshold={self.iv_threshold}. Lower iv_threshold."
            )

        self.lr_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.lr_.fit(X_woe, y_arr)
        self.classes_ = self.lr_.classes_
        self.intercept_ = float(self.lr_.intercept_[0])
        self.coefs_ = dict(zip(self.selected_features_, self.lr_.coef_[0].tolist()))
        self.factor_, self.offset_ = self._compute_pdo_constants()

        iv_map = dict(
            zip(
                self.woe_encoder_.iv_summary_["feature"],
                self.woe_encoder_.iv_summary_["iv"],
            )
        )
        coef_rows = []
        for feat in self.selected_features_:
            coef = float(self.coefs_[feat])
            coef_rows.append(
                {
                    "feature": feat,
                    "iv": float(iv_map[feat]),
                    "coef": coef,
                    "abs_coef": abs(coef),
                    "sign_ok": coef < 0.0,
                }
            )
        self.coef_summary_ = (
            pd.DataFrame(coef_rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        self.scorecard_table_ = self.build_scorecard_table()
        return self

    def transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        return self.woe_encoder_.transform(X, select_by_iv=True)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted log-odds of default: logit(P(Bad=1))."""
        X_woe = self.transform_woe(X)
        return self.lr_.decision_function(X_woe)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_woe = self.transform_woe(X)
        return self.lr_.predict_proba(X_woe)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_woe = self.transform_woe(X)
        return self.lr_.predict(X_woe)

    def check_sign_consistency(self) -> pd.DataFrame:
        """Return fitted feature coefficients and flag sign reversals (coef >= 0)."""
        self._require_fitted()
        return self.coef_summary_.copy()

    def build_scorecard_table(
        self,
        pdo: float | None = None,
        base_score: float | None = None,
        base_odds: float | None = None,
    ) -> pd.DataFrame:
        """Build bin-level PDO points table across all selected scorecard features."""
        self._require_fitted()
        factor, offset = self._compute_pdo_constants(pdo, base_score, base_odds)
        n_features = len(self.selected_features_)
        iv_map = dict(
            zip(
                self.woe_encoder_.iv_summary_["feature"],
                self.woe_encoder_.iv_summary_["iv"],
            )
        )

        frames: list[pd.DataFrame] = []
        for feat in self.selected_features_:
            coef = float(self.coefs_[feat])
            woe_tbl = self.woe_encoder_.get_woe_table(feat)
            pts_exact = -(coef * woe_tbl["woe"] + self.intercept_ / n_features) * factor + (
                offset / n_features
            )
            part = pd.DataFrame(
                {
                    "feature": feat,
                    "bin": woe_tbl["bin"],
                    "total": woe_tbl["total"],
                    "bad": woe_tbl["bad"],
                    "event_rate": woe_tbl["event_rate"],
                    "woe": woe_tbl["woe"],
                    "iv": float(iv_map[feat]),
                    "coef": coef,
                    "points_exact": pts_exact,
                    "points": np.round(pts_exact).astype(int),
                }
            )
            frames.append(part)

        return pd.concat(frames, ignore_index=True)

    def score_exact(self, X: pd.DataFrame) -> np.ndarray:
        """Compute continuous PDO credit score consistent with Score = Offset - Factor * logit(PD)."""
        self._require_fitted()
        logits = self.decision_function(X)
        return self.offset_ - self.factor_ * logits

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute rounded integer PDO credit score."""
        return np.round(self.score_exact(X)).astype(int)

    def score_breakdown(self, row: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Return per-feature WoE and PDO point contribution for a single applicant."""
        self._require_fitted()
        row_df = row.to_frame().T if isinstance(row, pd.Series) else row
        if len(row_df) != 1:
            raise ValueError("score_breakdown expects a single-row DataFrame or Series.")

        X_woe = self.transform_woe(row_df)
        n_features = len(self.selected_features_)
        rows = []
        for feat in self.selected_features_:
            woe_val = float(X_woe[feat].iloc[0])
            coef = float(self.coefs_[feat])
            pts_exact = -(coef * woe_val + self.intercept_ / n_features) * self.factor_ + (
                self.offset_ / n_features
            )
            rows.append(
                {
                    "feature": feat,
                    "woe": woe_val,
                    "coef": coef,
                    "points_exact": pts_exact,
                    "points": int(round(pts_exact)),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("points_exact", ascending=True)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "WoEScorecard":
        return joblib.load(path)


if __name__ == "__main__":
    train_csv = DATA_PROCESSED_DIR / "train.csv"
    valid_csv = DATA_PROCESSED_DIR / "valid.csv"
    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError("Missing train.csv or valid.csv in data/processed/.")

    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)

    X_train, y_train = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL]
    X_valid, y_valid = df_valid.drop(columns=[TARGET_COL]), df_valid[TARGET_COL]

    scorecard = WoEScorecard(binning="tree", iv_threshold=0.02)
    scorecard.fit(X_train, y_train)

    coef_df = scorecard.check_sign_consistency()
    reversed_feats = coef_df.loc[~coef_df["sign_ok"], "feature"].tolist()
    print(f"Selected features (IV >= 0.02): {len(scorecard.selected_features_)}")
    print(f"Logistic intercept: {scorecard.intercept_:.4f}")
    print("\n=== COEFFICIENT SIGN CHECK ===")
    print(coef_df.to_string(index=False, formatters={"iv": "{:.4f}".format, "coef": "{:.4f}".format, "abs_coef": "{:.4f}".format}))
    if reversed_feats:
        print(f"\n[Note] Sign-reversed features (coef >= 0, likely collinearity): {reversed_feats}")

    tbl = scorecard.build_scorecard_table()
    print(f"\n=== PDO SCORECARD TABLE PREVIEW (Total bins: {len(tbl)}) ===")
    print(
        tbl.head(15).to_string(
            index=False,
            columns=["feature", "bin", "total", "event_rate", "woe", "coef", "points"],
            formatters={
                "event_rate": "{:.2%}".format,
                "woe": "{:.3f}".format,
                "coef": "{:.3f}".format,
            },
        )
    )

    valid_probs = scorecard.predict_proba(X_valid)[:, 1]
    valid_scores = scorecard.score(X_valid)
    auc = roc_auc_score(y_valid, valid_probs)
    brier = brier_score_loss(y_valid, valid_probs)
    print(
        f"\nValid check -> ROC-AUC: {auc:.4f} | Gini: {2*auc - 1:.4f} | "
        f"Brier: {brier:.4f} | Score min/median/max: "
        f"{int(valid_scores.min())}/{int(np.median(valid_scores))}/{int(valid_scores.max())}"
    )

    # Verify exact algebraic consistency on a single row
    sample_row = X_valid.iloc[[0]]
    exact_total = float(scorecard.score_exact(sample_row)[0])
    breakdown_sum = float(scorecard.score_breakdown(sample_row)["points_exact"].sum())
    assert abs(exact_total - breakdown_sum) < 1e-9, (
        f"Score breakdown mismatch: {exact_total} vs {breakdown_sum}"
    )
    print(f"Algebraic breakdown check passed (diff={abs(exact_total - breakdown_sum):.2e}).")
