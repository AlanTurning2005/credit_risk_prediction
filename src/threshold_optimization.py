import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset import get_dataset
from src.config import BATCH_SIZE


def pd_to_credit_score(pd_series, min_score=300, max_score=850):
    """
    Quy đổi Xác suất nợ xấu (PD) sang thang điểm FICO tiêu chuẩn ngân hàng (300 - 850).
    - PD = 0.0 (an toàn tuyệt đối)  -> Điểm = 850
    - PD = 1.0 (rủi ro vỡ nợ 100%) -> Điểm = 300
    Sử dụng log-odds có giới hạn để tạo độ dốc phân hóa tốt.
    """
    eps = 1e-4
    clipped_pd = np.clip(pd_series, eps, 1.0 - eps)
    # log-odds: odds của khách hàng trả tốt so với nợ xấu
    log_odds = np.log((1.0 - clipped_pd) / clipped_pd)
    
    # Chuẩn hóa log-odds từ khoảng [-4, +4] về thang [min_score, max_score]
    norm_odds = (log_odds + 4.0) / 8.0
    norm_odds = np.clip(norm_odds, 0.0, 1.0)
    score = min_score + norm_odds * (max_score - min_score)
    return np.round(score).astype(int)


def optimize_threshold(
    model_path="models/best_model.keras",
    test_csv_path="data/processed/test.csv",
    profit_margin=0.15,       # Lợi nhuận biên trên 1 khoản vay tốt (15% lãi suất và phí)
    loss_given_default=0.75,  # Tỷ lệ tổn thất khi vỡ nợ (LGD: mất 75% vốn gốc sau khi thu hồi)
    output_config_path="config/threshold_config.json"
):
    """
    Xác định ngưỡng cắt tối ưu (Optimal Cut-off Threshold) dựa trên Ma trận Chi phí - Lợi nhuận.
    Ngân hàng duyệt vay khi PD < Threshold (Duyệt) và từ chối khi PD >= Threshold.
    """
    print("=" * 70)
    print("   TỐI ƯU HÓA NGƯỠNG CẮT PHÊ DUYỆT TÍN DỤNG (COST-BENEFIT MATRIX)")
    print("=" * 70)

    # 1. Nạp nhãn thực tế
    print("Nạp nhãn thực tế từ test.csv")
    df_test = pd.read_csv(test_csv_path, usecols=["TARGET"])
    y_true = df_test["TARGET"].to_numpy()
    total_customers = len(y_true)

    # 2. Dự báo xác suất vỡ nợ (PD)
    print(f"Nạp mô hình {model_path} và dự báo PD")
    model = tf.keras.models.load_model(model_path)
    test_ds = get_dataset(test_csv_path, batch_size=BATCH_SIZE, is_training=False)
    y_pred = model.predict(test_ds, verbose=0).ravel()

    # 3. Quét dải ngưỡng từ 0.05 đến 0.95 với bước nhảy 0.01
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    print("\nMô phỏng kết quả kinh doanh qua các mức ngưỡng cắt")
    for th in thresholds:
        # Quyết định: Phê duyệt nếu PD < th, Từ chối nếu PD >= th
        approved = (y_pred < th)
        rejected = ~approved

        # Trong số khách được duyệt (Approved):
        approved_goods = np.sum(approved & (y_true == 0))  # Khách tốt trả nợ đầy đủ
        approved_bads = np.sum(approved & (y_true == 1))   # Khách nợ xấu bị lọt vào

        # Trong số khách bị từ chối (Rejected):
        rejected_goods = np.sum(rejected & (y_true == 0))  # Khách tốt bị từ chối nhầm (Mất cơ hội)
        rejected_bads = np.sum(rejected & (y_true == 1))   # Khách nợ xấu chặn thành công

        # Tính toán tài chính giả định (chuẩn hóa trên 1 đơn vị tiền vay):
        # - Mỗi khách tốt được duyệt: mang lại +profit_margin lợi nhuận
        # - Mỗi khách nợ xấu lọt vào: gây thiệt hại -loss_given_default vốn gốc
        total_profit = approved_goods * profit_margin
        total_loss = approved_bads * loss_given_default
        net_profit = total_profit - total_loss

        total_approved = approved_goods + approved_bads
        approval_rate = total_approved / total_customers
        bad_rate_in_portfolio = (approved_bads / total_approved) if total_approved > 0 else 0.0

        results.append({
            "threshold": round(float(th), 2),
            "approval_rate_%": round(approval_rate * 100, 2),
            "bad_rate_%": round(bad_rate_in_portfolio * 100, 2),
            "approved_goods": int(approved_goods),
            "approved_bads": int(approved_bads),
            "rejected_customers": int(np.sum(rejected)),
            "net_profit_units": round(float(net_profit), 2)
        })

    df_results = pd.DataFrame(results)

    #Tìm điểm cực đại của Lợi nhuận ròng (Net Profit)
    best_row = df_results.loc[df_results["net_profit_units"].idxmax()]
    best_threshold = best_row["threshold"]

    # So sánh với ngưỡng mặc định 0.50
    row_default_05 = df_results.loc[df_results["threshold"] == 0.50].iloc[0]

    print("\n" + "=" * 70)
    print("                    KẾT QUẢ TỐI ƯU HÓA NGƯỠNG")
    print("=" * 70)
    print(f"Tham số giả định thị trường tín dụng:")
    print(f" - Lợi nhuận biên trên khoản vay tốt (Interest Margin): {profit_margin:.1%}")
    print(f" - Tổn thất khi khách hàng vỡ nợ (LGD):                 {loss_given_default:.1%}")
    print("-" * 70)
    print(f"★ NGƯỠNG CẮT TỐI ƯU TÌM ĐƯỢC:           Threshold = {best_threshold:.2f}")
    print(f"   + Tỷ lệ phê duyệt hồ sơ (Approval Rate): {best_row['approval_rate_%']}%")
    print(f"   + Tỷ lệ nợ xấu trong danh mục duyệt:    {best_row['bad_rate_%']}%")
    print(f"   + Lợi nhuận ròng tối đa đạt được:       {best_row['net_profit_units']:,} đơn vị")
    print("-" * 70)
    print(f"So với Ngưỡng truyền thống 0.50:")
    print(f"   + Tỷ lệ phê duyệt tại ngưỡng 0.50:       {row_default_05['approval_rate_%']}%")
    print(f"   + Tỷ lệ nợ xấu danh mục tại 0.50:       {row_default_05['bad_rate_%']}%")
    print(f"   + Lợi nhuận ròng tại 0.50:              {row_default_05['net_profit_units']:,} đơn vị")
    diff_profit = best_row["net_profit_units"] - row_default_05["net_profit_units"]
    print(f"   ==> Việc tối ưu ngưỡng giúp gia tăng:   +{diff_profit:,.2f} đơn vị lợi nhuận!")
    print("=" * 70)

    #Lưu kết quả cấu hình vào file JSON để FastAPI sử dụng
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    threshold_config = {
        "optimal_threshold": float(best_threshold),
        "profit_margin": profit_margin,
        "loss_given_default": loss_given_default,
        "expected_approval_rate_%": float(best_row["approval_rate_%"]),
        "expected_portfolio_bad_rate_%": float(best_row["bad_rate_%"]),
        "credit_score_scale": {"min": 300, "max": 850},
        "risk_tiers": {
            "VERY_LOW_RISK": {"score_min": 750, "action": "AUTO_APPROVE"},
            "LOW_RISK": {"score_min": 670, "action": "APPROVE"},
            "MEDIUM_RISK": {"score_min": 580, "action": "MANUAL_REVIEW"},
            "HIGH_RISK": {"score_min": 300, "action": "REJECT"}
        }
    }

    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=4, ensure_ascii=False)

    print(f"\nSaved config : {output_config_path}")
    return threshold_config


if __name__ == "__main__":
    optimize_threshold()
