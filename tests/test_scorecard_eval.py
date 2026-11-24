import numpy as np
import pytest
from src.scorecard_eval import calculate_metrics

def test_metrics_perfect_and_random():
    """Kiểm tra hai trường hợp biên: phân loại hoàn hảo và dự báo ngẫu nhiên."""
    # Case 1: Hoàn hảo
    res_perfect = calculate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert res_perfect["auc"] == pytest.approx(1.0)
    assert res_perfect["gini"] == pytest.approx(1.0)
    assert res_perfect["ks"] == pytest.approx(1.0)

    # Case 2: Dự báo ngẫu nhiên / không phân biệt
    res_rand = calculate_metrics(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert res_rand["auc"] == pytest.approx(0.5)
    assert res_rand["gini"] == pytest.approx(0.0)
    assert res_rand["ks"] == pytest.approx(0.0)


def test_metrics_hand_calculated():
    """Kiểm tra ca tính tay: 3/4 cặp đúng thứ tự, KS đạt cực đại tại 0.5."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    res = calculate_metrics(y_true, y_pred)
    
    assert res["auc"] == pytest.approx(0.75)
    assert res["gini"] == pytest.approx(0.50)
    assert res["ks"] == pytest.approx(0.50)