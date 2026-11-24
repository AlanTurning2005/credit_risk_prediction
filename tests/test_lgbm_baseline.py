import numpy as np
import pytest
from src.lgbm_baseline import calculate_metrics
import pandas as pd
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS
from src.lgbm_baseline import build_preprocessor


def test_metrics_perfect_prediction():
    """Dự báo hoàn hảo: Brier=0, AUC=1, Gini=1, KS=1, PR-AUC=1."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.0, 0.0, 1.0, 1.0])
    res = calculate_metrics(y_true, y_pred)
    assert res["brier"] == pytest.approx(0.0)
    assert res["auc"] == pytest.approx(1.0)
    assert res["gini"] == pytest.approx(1.0)
    assert res["ks"] == pytest.approx(1.0)
    assert res["pr_auc"] == pytest.approx(1.0)

def test_metrics_constant_prediction_brier():
    """Dự báo hằng số p=0.25: Brier = p*(1-p) = 0.1875, PR-AUC = 0.25."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0.25, 0.25, 0.25, 0.25])
    res = calculate_metrics(y_true, y_pred)
    assert res["brier"] == pytest.approx(0.1875)
    assert res["pr_auc"] == pytest.approx(0.25)



def test_preprocessor_pipeline_no_nan():
    """Kiểm tra ColumnTransformer điền khuyết median và OneHot sạch 100% NaN."""
    # Tạo DataFrame giả lập 2 dòng có giá trị khuyết NaN
    data = {col: [1.0, np.nan] for col in NUMERICAL_COLS}
    data.update({col: ["Category_A", np.nan] for col in CATEGORICAL_COLS})
    df_sample = pd.DataFrame(data)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df_sample)

    assert not np.isnan(transformed).any(), "Dữ liệu sau biến đổi vẫn còn chứa NaN!"
    assert transformed.shape[0] == 2

