import numpy as np
import pytest
import pandas as pd
from pathlib import Path

from src.scorecard import WoEScorecard

def test_pdo_constants_caculation():
    """Kiem tra factor, offset co khop voi cong thuc khong"""


    card = WoEScorecard(pdo= 20.0, base_score= 600.0, base_odds= 50.0)
    factor, offset = card._compute_pdo_constants()
    expected_factor = 20.0 / np.log(2.0)
    expected_offset = 600.0 - expected_factor * np.log(50.0)
    assert np.isclose(factor, expected_factor)
    assert np.isclose(offset, expected_offset)
    assert np.isclose(factor, 28.8539, atol=1e-3)
    assert np.isclose(offset, 487.123, atol=1e-3)


def test_pdo_points_to_double_odds_property():
    """Khi ln(odds_bad) giảm ln(2) (odds tốt:xấu nhân đôi), điểm phải tăng đúng PDO."""
    card = WoEScorecard(pdo=20.0, base_score=600.0, base_odds=50.0)
    factor, offset = card._compute_pdo_constants()

    ln_odds_1 = 0.5
    score_1 = offset - factor * ln_odds_1

    # Odds xấu giảm một nửa (tương đương odds tốt gấp đôi)
    ln_odds_2 = ln_odds_1 - np.log(2.0)
    score_2 = offset - factor * ln_odds_2

    assert np.isclose(score_2 - score_1, 20.0)


def test_invalid_pdo_or_base_odds_raises():
    """Hàm khởi tạo phải ném ValueError khi pdo hoặc base_odds <= 0."""
    with pytest.raises(ValueError, match="pdo must be positive"):
        WoEScorecard(pdo=0.0)

    with pytest.raises(ValueError, match="pdo must be positive"):
        WoEScorecard(pdo=-10.0)

    with pytest.raises(ValueError, match="base_odds.*must be positive"):
        WoEScorecard(base_odds=0.0)

    with pytest.raises(ValueError, match="base_odds.*must be positive"):
        WoEScorecard(base_odds=-5.0)


import pandas as pd


from src.config import CATEGORICAL_COLS, NUMERICAL_COLS


def _make_synthetic_data(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Tạo dữ liệu theo đúng schema 26 cột của dự án."""
    rng = np.random.default_rng(seed)
    strong = rng.uniform(0.1, 0.9, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    gender = rng.choice(["M", "F"], size=n, p=[0.6, 0.4])

    logit = -3.5 * strong + (gender == "F") * 1.2
    prob_bad = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series((rng.random(n) < prob_bad).astype(int), name="TARGET")

    data = {col: 1.0 for col in NUMERICAL_COLS}
    data.update({col: "Constant" for col in CATEGORICAL_COLS})
    data["EXT_SOURCE_1"] = strong
    data["EXT_SOURCE_2"] = noise
    data["CODE_GENDER"] = gender

    return pd.DataFrame(data), y


def test_scorecard_fit_feature_selection_and_coefficients():
    """Kiểm tra: lọc đúng biến theo IV, hệ số biến mạnh âm, và xác suất dự báo hợp lệ."""
    X, y = _make_synthetic_data()
    card = WoEScorecard(binning="tree", iv_threshold=0.02, random_state=42)
    card.fit(X, y)

    # 1. Kiểm tra lọc biến qua IV: EXT_SOURCE_1 và CODE_GENDER được chọn, EXT_SOURCE_2 bị loại
    assert "EXT_SOURCE_1" in card.selected_features_
    assert "CODE_GENDER" in card.selected_features_
    assert "EXT_SOURCE_2" not in card.selected_features_

    # 2. Kiểm tra dấu hệ số hồi quy (kỳ vọng âm)
    assert card.coefs_["EXT_SOURCE_1"] < 0.0
    assert card.coef_summary_.loc[card.coef_summary_["feature"] == "EXT_SOURCE_1", "sign_ok"].iloc[0]

    # 3. Kiểm tra predict_proba: shape (n, 2) và tổng xác suất mỗi hàng bằng 1.0
    probs = card.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_scorecard_algebraic_properties_and_monotonicity():
    """Kiểm tra: điểm khớp công thức logit, bảo toàn điểm thành phần, và đơn điệu giảm theo PD."""
    X, y = _make_synthetic_data()
    card = WoEScorecard(binning="tree", iv_threshold=0.02, random_state=42).fit(X, y)

    logits = card.decision_function(X)
    exact_scores = card.score_exact(X)

    # 1. Điểm chính xác phải bằng offset - factor * logit(PD)
    expected_scores = card.offset_ - card.factor_ * logits
    assert np.allclose(exact_scores, expected_scores)

    # 2. Bảo toàn điểm: tổng điểm từng đặc trưng trong score_breakdown bằng score_exact
    sample_row = X.iloc[[0]]
    breakdown_sum = card.score_breakdown(sample_row)["points_exact"].sum()
    assert np.isclose(breakdown_sum, card.score_exact(sample_row)[0])

    # 3. Điểm integer là làm tròn của exact score; PD tăng dần thì Score không tăng
    assert np.array_equal(card.score(X), np.round(exact_scores).astype(int))
    pd_bad = card.predict_proba(X)[:, 1]
    sorted_scores = exact_scores[np.argsort(pd_bad)]
    assert np.all(np.diff(sorted_scores) <= 0.0)
    assert np.all(np.diff(sorted_scores) <= 0.0)



def test_scorecard_error_handling():
    """Kiểm tra xử lý ngoại lệ: chưa fit, sai kiểu dữ liệu, lệch số dòng, và không có biến đạt IV."""
    X, y = _make_synthetic_data(n=200)
    card = WoEScorecard()

    # 1. Gọi khi chưa fit -> RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted before calling"):
        card.score(X)

    # 2. X không phải pandas DataFrame -> TypeError
    with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
        card.fit(X.to_numpy(), y)

    # 3. X và y khác số dòng -> ValueError
    with pytest.raises(ValueError, match="same number of rows"):
        card.fit(X, y.iloc[:100])

    # 4. iv_threshold quá cao không biến nào qua được -> RuntimeError
    with pytest.raises(RuntimeError, match="No features passed iv_threshold"):
        WoEScorecard(iv_threshold=10.0).fit(X, y)



def test_scorecard_save_and_load_consistency(tmp_path: Path):
    """Mô hình sau khi save và load bằng joblib phải cho ra dự báo và điểm giống hệt trước khi lưu."""
    X, y = _make_synthetic_data()
    card = WoEScorecard(binning="tree", iv_threshold=0.02, random_state=42).fit(X, y)

    orig_probs = card.predict_proba(X)
    orig_scores = card.score(X)

    # Lưu vào file tạm trong tmp_path
    save_file = tmp_path / "scorecard_test.joblib"
    card.save(save_file)
    assert save_file.exists()

    # Nạp lại mô hình và đối chiếu
    loaded_card = WoEScorecard.load(save_file)
    assert np.allclose(loaded_card.predict_proba(X), orig_probs)
    assert np.array_equal(loaded_card.score(X), orig_scores)
    assert loaded_card.selected_features_ == card.selected_features_