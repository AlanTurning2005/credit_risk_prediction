from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_RAW_SELECTED_COLS,
    BOUNDED_01_COLS,
    CATEGORICAL_COLS,
    CSV_COLUMNS,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DAYS_EMPLOYED_ANOMALY_VALUE,
    ID_COL,
    MIN_ONE_COLS,
    NON_NEGATIVE_COLS,
    NUMERICAL_COLS,
    RAW_CATEGORICAL_COLS,
    RAW_NUMERICAL_COLS,
    SPLIT_CONFIG,
    STRICTLY_POSITIVE_COLS,
    TARGET_COL,
)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series safely, returning NaN where denominator <= 0 or either operand is NaN."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    valid = num.notna() & den.notna() & (den > 0)
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def _normalize_cat_series(series: pd.Series) -> pd.Series:
    """Normalize categorical series: NaN / empty strings -> 'Missing'."""
    cleaned = series.astype("object").where(series.notna(), "Missing").astype(str).str.strip()
    return cleaned.replace({"": "Missing", "nan": "Missing", "None": "Missing"})


def validate_raw_input(
    df: pd.DataFrame,
    require_all_raw: bool = False,
    require_target: bool = False,
) -> None:
    """Validate input schema and domain constraints before feature engineering.

    Accepts either raw Home Credit schema (`DAYS_BIRTH`, `DAYS_EMPLOYED`) or
    serving/processed age-years schema (`AGE`, `YEARS_EMPLOYED`) unless
    `require_all_raw=True`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row.")

    cols = set(df.columns)
    common_num = [c for c in RAW_NUMERICAL_COLS if c not in ("DAYS_BIRTH", "DAYS_EMPLOYED")]
    required_common = set(common_num) | set(RAW_CATEGORICAL_COLS)

    if require_all_raw:
        required = set(RAW_NUMERICAL_COLS) | set(RAW_CATEGORICAL_COLS)
    else:
        has_raw_time = {"DAYS_BIRTH", "DAYS_EMPLOYED"}.issubset(cols)
        has_eng_time = {"AGE", "YEARS_EMPLOYED"}.issubset(cols)
        if has_raw_time or has_eng_time:
            required = required_common
        else:
            required = required_common | {"DAYS_BIRTH", "DAYS_EMPLOYED"}

    if require_target:
        required.add(TARGET_COL)

    missing_cols = sorted(required - cols)
    if missing_cols:
        raise ValueError(f"Missing required input columns: {missing_cols}")

    # Strictly positive numeric fields
    for col in STRICTLY_POSITIVE_COLS:
        if col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s.notna() & (s <= 0)).any():
                raise ValueError(f"Column '{col}' contains non-positive values (<= 0).")

    # Non-negative numeric fields
    for col in NON_NEGATIVE_COLS:
        if col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s.notna() & (s < 0)).any():
                raise ValueError(f"Column '{col}' contains negative values (< 0).")

    for col in MIN_ONE_COLS:
        if col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s.notna() & (s < 1)).any():
                raise ValueError(f"Column '{col}' contains values < 1.")

    for col in BOUNDED_01_COLS:
        if col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s.notna() & ((s < 0.0) | (s > 1.0))).any():
                raise ValueError(f"Column '{col}' contains values outside [0, 1].")

    if "AGE" in cols:
        age = pd.to_numeric(df["AGE"], errors="coerce")
        if (age.notna() & ((age < 18.0) | (age > 120.0))).any():
            raise ValueError("Column 'AGE' contains values outside valid adult range [18, 120].")

    if "DAYS_BIRTH" in cols:
        days_birth = pd.to_numeric(df["DAYS_BIRTH"], errors="coerce")
        if (days_birth.notna() & ((days_birth > -18 * 365) | (days_birth < -120 * 365))).any():
            raise ValueError("Column 'DAYS_BIRTH' implies age outside valid adult range [18, 120].")

    if "DAYS_EMPLOYED" in cols:
        emp = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
        invalid_pos = emp.notna() & (emp > 0) & (emp != DAYS_EMPLOYED_ANOMALY_VALUE)
        if invalid_pos.any():
            raise ValueError(
                f"Column 'DAYS_EMPLOYED' contains invalid positive values other than {DAYS_EMPLOYED_ANOMALY_VALUE}."
            )

    if require_target:
        target = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if target.isna().any() or not set(target.unique()).issubset({0, 1}):
            raise ValueError(f"Target column '{TARGET_COL}' must only contain 0 and 1 without NaN.")


def engineer_features(
    df: pd.DataFrame,
    validate: bool = True,
    keep_target: bool = True,
) -> pd.DataFrame:
    """Clean anomalies and compute deterministic credit risk domain features.

    Preserves NaN in numerical features (e.g., EXT_SOURCE_1/3, YEARS_EMPLOYED) so
    downstream transformers (TabularPreprocessor or WoEEncoder) can fit on train only.
    """
    if validate:
        validate_raw_input(df, require_all_raw=False, require_target=False)

    out = df.copy()

    if "AGE" not in out.columns:
        out["AGE"] = -pd.to_numeric(out["DAYS_BIRTH"], errors="coerce") / 365.0
    else:
        out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")

    if "DAYS_EMPLOYED" in out.columns:
        emp_days = pd.to_numeric(out["DAYS_EMPLOYED"], errors="coerce")
        is_anom = emp_days == DAYS_EMPLOYED_ANOMALY_VALUE
        out["DAYS_EMPLOYED_ANOM"] = is_anom.astype(int)
        clean_days = emp_days.where(~is_anom & (emp_days <= 0), np.nan)
        years_emp = np.where(clean_days.isna(), np.nan, np.maximum(-clean_days / 365.0, 0.0))
        out["YEARS_EMPLOYED"] = pd.Series(years_emp, index=out.index, dtype="float64")
    else:
        years_emp = pd.to_numeric(out["YEARS_EMPLOYED"], errors="coerce")
        # Avoid IEEE -0.0 when YEARS_EMPLOYED == 0.0
        out["YEARS_EMPLOYED"] = years_emp.where(years_emp != 0.0, 0.0)
        if "DAYS_EMPLOYED_ANOM" not in out.columns:
            out["DAYS_EMPLOYED_ANOM"] = 0
        else:
            out["DAYS_EMPLOYED_ANOM"] = pd.to_numeric(out["DAYS_EMPLOYED_ANOM"], errors="coerce").fillna(0).astype(int)

    # Derived credit domain features (all safe against division by zero / NaN)
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    for col in ext_cols + ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["EXT_SOURCES_MEAN"] = out[ext_cols].mean(axis=1)
    out["PAYMENT_RATE"] = safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["INCOME_CREDIT_PERC"] = safe_divide(out["AMT_INCOME_TOTAL"], out["AMT_CREDIT"])
    out["ANNUITY_INCOME_PERC"] = safe_divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])

    for col in CATEGORICAL_COLS:
        if col in out.columns:
            # Preserve raw NaN vs explicit string consistently as object/NaN or normalized in preprocessor
            out[col] = out[col].astype("object")

    selected_cols = list(NUMERICAL_COLS) + list(CATEGORICAL_COLS)
    if keep_target and TARGET_COL in out.columns:
        selected_cols.append(TARGET_COL)

    return out[selected_cols]


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Fit-on-train, transform-on-serving tabular preprocessor.

    - Learns per-column numerical medians strictly from X_train.
    - Normalizes categorical NaN/empty values to 'Missing' and pools rare categories
      below `min_cat_freq` into 'Other'.
    - Optionally adds binary missingness indicators (`{col}_IS_MISSING`).
    """

    def __init__(
        self,
        num_cols: Sequence[str] = tuple(NUMERICAL_COLS),
        cat_cols: Sequence[str] = tuple(CATEGORICAL_COLS),
        min_cat_freq: float = 0.005,
        add_missing_indicators: bool = False,
        validate_on_transform: bool = True,
    ) -> None:
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.min_cat_freq = min_cat_freq
        self.add_missing_indicators = add_missing_indicators
        self.validate_on_transform = validate_on_transform

    def _ensure_engineered(self, X: pd.DataFrame, validate: bool) -> pd.DataFrame:
        needed = set(self.num_cols) | set(self.cat_cols)
        is_credit_schema = set(NUMERICAL_COLS).issubset(needed) and set(CATEGORICAL_COLS).issubset(needed)
        if not needed.issubset(X.columns):
            if is_credit_schema:
                return engineer_features(X, validate=validate, keep_target=False)
            missing = sorted(needed - set(X.columns))
            raise ValueError(f"Missing required columns for TabularPreprocessor: {missing}")
        if validate and is_credit_schema:
            validate_raw_input(X, require_all_raw=False, require_target=False)
        return X.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TabularPreprocessor":
        df_eng = self._ensure_engineered(X, validate=True)

        self.medians_: dict[str, float] = {}
        self.missing_indicator_cols_: list[str] = []
        for col in self.num_cols:
            s = pd.to_numeric(df_eng[col], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            self.medians_[col] = med
            if self.add_missing_indicators and s.isna().any():
                self.missing_indicator_cols_.append(col)

        self.valid_categories_: dict[str, set[str]] = {}
        for col in self.cat_cols:
            s = _normalize_cat_series(df_eng[col])
            freqs = s.value_counts(normalize=True)
            kept = set(freqs[freqs >= self.min_cat_freq].index.tolist())
            kept.add("Missing")
            self.valid_categories_[col] = kept

        indicator_names = [f"{col}_IS_MISSING" for col in self.missing_indicator_cols_]
        self.feature_names_out_ = list(self.num_cols) + indicator_names + list(self.cat_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform().")

        df_eng = self._ensure_engineered(X, validate=self.validate_on_transform)
        out = pd.DataFrame(index=df_eng.index)

        for col in self.num_cols:
            s = pd.to_numeric(df_eng[col], errors="coerce")
            if self.add_missing_indicators and col in self.missing_indicator_cols_:
                out[f"{col}_IS_MISSING"] = s.isna().astype(int)
            out[col] = s.fillna(self.medians_[col]).astype("float64")

        for col in self.cat_cols:
            s = _normalize_cat_series(df_eng[col])
            allowed = self.valid_categories_[col]
            out[col] = s.where(s.isin(allowed), "Other")

        return out[self.feature_names_out_]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        return joblib.load(path)


def prepare_processed_splits(
    raw_csv_path: str | Path = DATA_RAW_DIR / "application_train.csv",
    output_dir: str | Path = DATA_PROCESSED_DIR,
    test_size: float = SPLIT_CONFIG["test_size"],
    valid_size: float = SPLIT_CONFIG["valid_size"],
    random_state: int = SPLIT_CONFIG["random_state"],
) -> dict[str, tuple[int, int]]:
    """Build reproducible stratified train/valid/test CSVs from raw application_train.csv."""
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df_raw = pd.read_csv(raw_path, usecols=ALL_RAW_SELECTED_COLS)
    validate_raw_input(df_raw, require_all_raw=True, require_target=True)
    df_clean = engineer_features(df_raw, validate=False, keep_target=True)[CSV_COLUMNS]

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_size, random_state=random_state, stratify=y_temp
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes: dict[str, tuple[int, int]] = {}
    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ):
        df_split = X_split.copy()
        df_split[TARGET_COL] = y_split
        df_split = df_split[CSV_COLUMNS]
        df_split.to_csv(out_dir / f"{split_name}.csv", index=False)
        shapes[split_name] = df_split.shape

    return shapes


if __name__ == "__main__":
    print("--- PREPARING CLEAN PROCESSED SPLITS ---")
    split_shapes = prepare_processed_splits()
    for name, shape in split_shapes.items():
        print(f"Saved {name}.csv -> shape={shape}")

    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_PROCESSED_DIR / "valid.csv")

    prep = TabularPreprocessor(add_missing_indicators=False)
    X_train_t = prep.fit_transform(train_df.drop(columns=[TARGET_COL]))
    X_valid_t = prep.transform(valid_df.drop(columns=[TARGET_COL]))
    print(f"TabularPreprocessor check -> train nulls: {int(X_train_t.isna().sum().sum())}, valid nulls: {int(X_valid_t.isna().sum().sum())}")
