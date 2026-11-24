""""
Đưa mô hình vào phục vụ thực tế 
Gửi hồ sơ vay và nhận quyết định duyệt vay thông qua giao thức HTTP/ RESTful API
Kiểm tra và từ chối các yêu cầu gửi sai định dạng


"""
import sys
from pathlib import Path
# Đảm bảo đường dẫn tuyệt đối cho model và config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI
import tensorflow as tf
import uvicorn
import json
import numpy as np
import time
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
CONFIG_PATH = PROJECT_ROOT / "config" / "threshold_config.json"

class LoanApplicationRequest(BaseModel):
    EXT_SOURCE_1: float = Field(0.5, description="Điểm tín dụng bên ngoài 1")
    EXT_SOURCE_2: float = Field(0.5, description="Điểm tín dụng bên ngoài 2")
    EXT_SOURCE_3: float = Field(0.5, description="Điểm tín dụng bên ngoài 3")
    AMT_INCOME_TOTAL: float = Field(150000.0, description="Tổng thu nhập năm")
    AMT_CREDIT: float = Field(500000.0, description="Số tiền vay")
    AMT_ANNUITY: float = Field(25000.0, description="Số tiền phải trả mỗi kỳ")
    AMT_GOODS_PRICE: float = Field(450000.0, description="Giá trị hàng hóa vay mua")
    CNT_CHILDREN: float = Field(0.0, description="Số lượng con cái")
    CNT_FAM_MEMBERS: float = Field(2.0, description="Số lượng thành viên gia đình")
    AGE: float = Field(35.0, description="Tuổi khách hàng")
    YEARS_EMPLOYED: float = Field(5.0, description="Số năm làm việc")
    DAYS_EMPLOYED_ANOM: float = Field(0.0, description="Cờ bất thường ngày làm việc")
    EXT_SOURCES_MEAN: float = Field(0.5, description="Trung bình 3 nguồn ngoài")
    PAYMENT_RATE: float = Field(0.05, description="Tỷ lệ trả nợ / khoản vay")
    INCOME_CREDIT_PERC: float = Field(0.3, description="Tỷ lệ thu nhập / khoản vay")
    ANNUITY_INCOME_PERC: float = Field(0.16, description="Tỷ lệ tiền góp / thu nhập")
    
    # 10 Biến danh mục (String)
    NAME_CONTRACT_TYPE: str = Field("Cash loans", description="Loại hợp đồng")
    NAME_INCOME_TYPE: str = Field("Working", description="Nguồn thu nhập")
    CODE_GENDER: str = Field("M", description="Giới tính (M/F)")
    NAME_EDUCATION_TYPE: str = Field("Secondary / secondary special", description="Học vấn")
    NAME_FAMILY_STATUS: str = Field("Married", description="Tình trạng hôn nhân")
    OCCUPATION_TYPE: str = Field("Laborers", description="Nghề nghiệp")
    ORGANIZATION_TYPE: str = Field("Business Entity Type 3", description="Loại tổ chức")
    NAME_HOUSING_TYPE: str = Field("House / apartment", description="Loại nhà ở")
    FLAG_OWN_CAR: str = Field("Y", description="Có sở hữu ô tô không (Y/N)")
    FLAG_OWN_REALTY: str = Field("Y", description="Có sở hữu bất động sản không (Y/N)")


app_state = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Load trained model to memory")
    app_state["model"] = tf.keras.models.load_model("models/best_model.keras")

    print("LOADING OPTIMAL THRESHOLD")
    with open("config/threshold_config.json", "r", encoding= "utf-8") as f:
        app_state["config"] = json.load(f)

    print('Ready to use')
    yield
    app_state.clear()

app = FastAPI(
    title = "Home Credit Default Risk Prediction API",
    description='Hệ thống chấm điểm tín dụng và phê duyệt khoản vay tự động',
    version = '1.0.0',
    lifespan = lifespan
)

@app.post('/predict')
async def predict_credit_risk(application: LoanApplicationRequest):
    start_time = time.time()

    input_data = application.model_dump()
    tensor_inputs = {
        col: tf.constant([val], dtype = tf.float32 if isinstance(val,(int, float)) else tf.string) for col, val in input_data.items()
    }

    model = app_state["model"]
    pd_pred = float(model(tensor_inputs, training=False).numpy()[0][0])

    config = app_state['config']
    optimal_th = config['optimal_threshold']

    eps = 1e-4
    clipped_pd = np.clip(pd_pred, eps, 1.0 - eps)
    log_odds = np.log((1.0 - clipped_pd) / clipped_pd)
    norm_odds = np.clip((log_odds + 4.0) / 8.0, 0.0, 1.0)
    credit_score = int(round(300 + norm_odds * 550))    

    if credit_score >= 750:
        risk_tier = "VERY_LOW_RISK"
        action = "AUTO_APPROVE"
        decision = "APPROVED"
    elif credit_score >= 670:
        risk_tier = "LOW_RISK"
        action = "APPROVE"
        decision = "APPROVED"
    elif credit_score >= 580:
        risk_tier = "MEDIUM_RISK"
        action = "MANUAL_REVIEW"
        decision = "PENDING_VERIFICATION"
    else:
        risk_tier = "HIGH_RISK"
        action = "AUTO_REJECT"
        decision = "REJECTED"
    latency_ms = round((time.time() - start_time) * 1000, 2)
    return {
        "decision": decision,
        "action_required": action,
        "credit_score": credit_score,
        "risk_tier": risk_tier,
        "probability_of_default": round(pd_pred, 4),
        "threshold_used": optimal_th,
        "inference_latency_ms": latency_ms
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in app_state,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)