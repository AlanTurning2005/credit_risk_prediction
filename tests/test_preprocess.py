import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_COLS, NUMERICAL_COLS
from src.preprocess import TabularPreprocessor, engineer_features, safe_divide, validate_raw_input


def _make_sample_raw_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.5, np.nan, 0.7, 0.3][:n],
            "EXT_SOURCE_2": [0.6, 0.4, 0.8, 0.2][:n],
            "EXT_SOURCE_3": [0.55, 0.35, np.nan, 0.45][:n],
            "AMT_INCOME_TOTAL": [180000.0, 120000.0, 250000.0, 90000.0][:n],
            "AMT_CREDIT": [400000.0, 300000.0, 600000.0, 200000.0][:n],
            "AMT_ANNUITY": [20000.0, 15000.0, 30000.0, 10000.0][:n],
            "AMT_GOODS_PRICE": [360000.0, 280000.0, 550000.0, 180000.0][:n],
            "DAYS_BIRTH": [-12000, -15000, -10000, -20000][:n],
            "DAYS_EMPLOYED": [-1500, 365243, 0, -3650][:n],
            "CNT_CHILDREN": [0, 1, 2, 0][:n],
            "CNT_FAM_MEMBERS": [1.0, 3.0, 4.0, 2.0][:n],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans", "Cash loans"][:n],
            "NAME_INCOME_TYPE": ["Working", "Pensioner", "Commercial associate", "Working"][:n],
            "CODE_GENDER": ["M", "F", "F", "M"][:n],
            "NAME_EDUCATION_TYPE": ["Higher education", "Secondary / secondary special", "Higher education", "Lower secondary"][:n],
            "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Married", "Widow"][:n],
            "OCCUPATION_TYPE": ["Managers", np.nan, "Core staff", "Laborers"][:n],
            "ORGANIZATION_TYPE": ["Business Entity Type 3", "XNA", "Self-employed", "Trade: type 7"][:n],
            "NAME_HOUSING_TYPE": ["House / apartment", "House / apartment", "Rented apartment", "House / apartment"][:n],
            "FLAG_OWN_CAR": ["Y", "N", "Y", "N"][:n],
            "FLAG_OWN_REALTY": ["Y", "Y", "N", "Y"][:n],
        }
    )


def test_missing_required_columns_raises():
    df = _make_sample_raw_df().drop(columns=["AMT_CREDIT", "CODE_GENDER"])
    with pytest.raises(ValueError, match="Missing required input columns"):
        validate_raw_input(df)


def test_empty_dataframe_raises():
    df = _make_sample_raw_df().iloc[0:0]
    with pytest.raises(ValueError, match="at least one row"):
        validate_raw_input(df)


@pytest.mark.parametrize(
    "col,bad_val",
    [
        ("AMT_INCOME_TOTAL", -5000.0),
        ("AMT_INCOME_TOTAL", 0.0),
        ("AMT_CREDIT", -100.0),
        ("AMT_ANNUITY", 0.0),
        ("CNT_CHILDREN", -1),
        ("CNT_FAM_MEMBERS", 0),
        ("EXT_SOURCE_2", 1.5),
        ("EXT_SOURCE_3", -0.1),
    ],
)
def test_invalid_numeric_values_raise(col, bad_val):
    df = _make_sample_raw_df()
    df.loc[0, col] = bad_val
    with pytest.raises(ValueError):
        validate_raw_input(df)


def test_underage_days_birth_raises():
    df = _make_sample_raw_df()
    df.loc[0, "DAYS_BIRTH"] = -16 * 365
    with pytest.raises(ValueError, match="DAYS_BIRTH"):
        validate_raw_input(df)


def test_days_employed_anomaly_365243_and_zero_sign():
    df = _make_sample_raw_df(4)
    eng = engineer_features(df, validate=True)

    assert list(eng.columns) == NUMERICAL_COLS + CATEGORICAL_COLS
    # Row 1 has DAYS_EMPLOYED == 365243 -> ANOM=1 and YEARS_EMPLOYED=NaN before imputation
    assert eng.loc[1, "DAYS_EMPLOYED_ANOM"] == 1
    assert np.isnan(eng.loc[1, "YEARS_EMPLOYED"])
    # Row 2 has DAYS_EMPLOYED == 0 -> exact +0.0 (no negative zero -0.0)
    assert eng.loc[2, "DAYS_EMPLOYED_ANOM"] == 0
    assert eng.loc[2, "YEARS_EMPLOYED"] == 0.0
    assert not np.signbit(eng.loc[2, "YEARS_EMPLOYED"])


def test_safe_divide_guards_zero_and_nan():
    num = pd.Series([100.0, 200.0, np.nan, 50.0])
    den = pd.Series([20.0, 0.0, 10.0, -5.0])
    res = safe_divide(num, den)
    assert res.iloc[0] == pytest.approx(5.0)
    assert np.isnan(res.iloc[1])
    assert np.isnan(res.iloc[2])
    assert np.isnan(res.iloc[3])


def test_tabular_preprocessor_no_leakage_and_unseen_category():
    df_train = _make_sample_raw_df(4)
    # Force an extra known category so min_cat_freq works cleanly
    df_train = pd.concat([df_train] * 5, ignore_index=True)

    prep = TabularPreprocessor(min_cat_freq=0.1, add_missing_indicators=True)
    X_train_t = prep.fit_transform(df_train)

    assert X_train_t.isna().sum().sum() == 0
    assert "EXT_SOURCE_1_IS_MISSING" in X_train_t.columns
    assert "YEARS_EMPLOYED_IS_MISSING" in X_train_t.columns

    # Valid/test row with extreme values and unseen category must NOT shift train medians
    df_test = _make_sample_raw_df(1)
    df_test.loc[0, "EXT_SOURCE_1"] = np.nan
    df_test.loc[0, "OCCUPATION_TYPE"] = "Astronaut_Unseen"
    X_test_t = prep.transform(df_test)

    assert X_test_t.loc[0, "EXT_SOURCE_1"] == pytest.approx(prep.medians_["EXT_SOURCE_1"])
    assert X_test_t.loc[0, "EXT_SOURCE_1_IS_MISSING"] == 1
    assert X_test_t.loc[0, "OCCUPATION_TYPE"] == "Other"
