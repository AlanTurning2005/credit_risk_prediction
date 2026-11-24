import pytest
from fastapi.testclient import TestClient
from src.api import app, app_state

@pytest.fixture(scope ="module")
def client():
    "intinialize TestClient to trigger lifespan "
    with TestClient(app) as c: 
        yield c 

def test_health_endpoint(client):
    respone = client.get("/health")
    assert respone.status_code == 200
    data = respone.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_predict_endpoint_contact_and_determinism(client):
    payload = {}
    res1 = client.post("/predict", json= payload)
    assert res1.status_code == 200
    data1 = res1.json()

    for key in ("decision", "action_required", "credit_score",
                "risk_tier", "probability_of_default", "threshold_used"):
        assert key in data1
    assert 0.0 <= data1["probability_of_default"] <= 1.0
    assert 300 <= data1["credit_score"] <= 850

    res2 = client.post("/predict", json = payload)
    assert data1["probability_of_default"] == res2.json()["probability_of_default"]
    assert data1['credit_score'] == res2.json()["credit_score"]

def test_predict_invalid_data_type_returns_422(client):
    """Gửi sai kiểu dữ liệu (chuỗi cho trường số) phải bị Pydantic chặn với mã 422."""
    payload = {"AMT_INCOME_TOTAL": "khong_phai_so"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    err_detail = response.json()["detail"]
    assert any(err["loc"][-1] == "AMT_INCOME_TOTAL" for err in err_detail)

def test_predict_out_of_range_values_behavior(client):
    """Ghi nhận hành vi thực tế: Giá trị phi lý (AGE=-5, EXT=5) vẫn trả về 200 do thiếu ràng buộc."""
    payload = {"AGE": -5.0, "EXT_SOURCE_1": 5.0}
    response = client.post("/predict", json=payload)
    # Hiện tại Pydantic schema chưa có ge/le nên API vẫn chấp nhận và trả về 200
    assert response.status_code == 200
    assert "credit_score" in response.json()




class DummyModel:
    """Mô hình giả trả về xác suất PD cố định để test các nhánh logic."""
    def __init__(self, fixed_pd: float):
        self.fixed_pd = fixed_pd

    def __call__(self, *args, **kwargs):
        import tensorflow as tf
        return tf.constant([[self.fixed_pd]], dtype=tf.float32)


@pytest.mark.parametrize(
    "fixed_pd, expected_score, expected_tier, expected_decision",
    [
        (0.50, 575, "HIGH_RISK", "REJECTED"),
        (0.20, 670, "LOW_RISK", "APPROVED"),
        (0.40, 603, "MEDIUM_RISK", "PENDING_VERIFICATION"),
        (0.01, 850, "VERY_LOW_RISK", "APPROVED"),
        (0.90, 424, "HIGH_RISK", "REJECTED"),
    ],
)
def test_predict_score_and_risk_tiers_with_dummy_model(client, fixed_pd, expected_score, expected_tier, expected_decision):
    """Kiểm tra logic quy đổi PD sang điểm và phân nhóm rủi ro bằng mô hình giả."""
    original_model = app_state["model"]
    try:
        app_state["model"] = DummyModel(fixed_pd)
        res = client.post("/predict", json={})
        data = res.json()
        assert data["credit_score"] == expected_score
        assert data["risk_tier"] == expected_tier
        assert data["decision"] == expected_decision
    finally:
        app_state["model"] = original_model


import numpy as np
from src.threshold_optimization import pd_to_credit_score

def test_score_formula_consistency_with_threshold_optimization(client):
    """Đảm bảo công thức tính điểm trong api.py khớp hoàn toàn với pd_to_credit_score."""
    original_model = app_state["model"]
    pd_grid = [0.001, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 0.999]

    try:
        for pd_val in pd_grid:
            app_state["model"] = DummyModel(pd_val)
            api_score = client.post("/predict", json={}).json()["credit_score"]
            expected_score = int(pd_to_credit_score(np.array([pd_val]))[0])
            assert api_score == expected_score, f"Lệch điểm tại PD={pd_val}: API={api_score}, hàm={expected_score}"
    finally:
        app_state["model"] = original_model




def test_optimal_threshold_does_not_affect_decision(client):
    """Chứng minh: optimal_threshold chỉ để hiển thị (threshold_used), không đổi decision."""
    original_model = app_state["model"]
    original_config = app_state["config"].copy()

    try:
        # Cố định PD = 0.20 (tương ứng credit_score = 670 -> decision: APPROVED)
        app_state["model"] = DummyModel(0.20)

        for dummy_th in (0.01, 0.99):
            app_state["config"]["optimal_threshold"] = dummy_th
            res = client.post("/predict", json={}).json()
            # Dù ngưỡng = 0.01 (rất khắt khe), decision vẫn là APPROVED vì logic dựa vào credit_score >= 670
            assert res["decision"] == "APPROVED"
            assert res["threshold_used"] == dummy_th
    finally:
        app_state["model"] = original_model
        app_state["config"] = original_config