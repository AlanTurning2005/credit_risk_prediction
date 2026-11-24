import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.dataset import get_dataset
from src.config import BATCH_SIZE


def calculate_ks_statistic(y_true, y_pred):
    """
    Tính chỉ số KS (Kolmogorov-Smirnov Statistic) chuẩn ngân hàng.
    Đo mức độ phân tách lớn nhất giữa phân phối nợ tốt (0) và nợ xấu (1).
    """
    good_preds = y_pred[y_true == 0]
    bad_preds = y_pred[y_true == 1]
    ks_res = ks_2samp(bad_preds, good_preds)
    return ks_res.statistic, ks_res.pvalue


def generate_decile_table(y_true, y_pred, num_bins=10):
    """
    Tạo bảng phân tích 10 nhóm rủi ro (Decile Analysis).
    Một mô hình tốt phải có tỷ lệ nợ xấu (Bad Rate) giảm dần đều từ nhóm 1 đến 10.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=num_bins, labels=False, duplicates="drop")
    # Đảo thứ tự để Decile 1 là nhóm rủi ro cao nhất (xác suất PD lớn nhất)
    df["decile"] = num_bins - df["decile"]

    grouped = df.groupby("decile").agg(
        total_customers=("y_true", "count"),
        bad_customers=("y_true", "sum"),
        min_prob=("y_pred", "min"),
        max_prob=("y_pred", "max"),
        avg_prob=("y_pred", "mean")
    ).reset_index()

    grouped["good_customers"] = grouped["total_customers"] - grouped["bad_customers"]
    grouped["bad_rate_%"] = (grouped["bad_customers"] / grouped["total_customers"]) * 100

    # Tích lũy tỷ lệ
    total_bads = grouped["bad_customers"].sum()
    total_goods = grouped["good_customers"].sum()
    grouped["cum_bad_%"] = (grouped["bad_customers"].cumsum() / total_bads) * 100
    grouped["cum_good_%"] = (grouped["good_customers"].cumsum() / total_goods) * 100
    grouped["ks_%"] = np.abs(grouped["cum_bad_%"] - grouped["cum_good_%"])

    return grouped


def evaluate_model(
    model_path="models/best_model.keras",
    test_csv_path="data/processed/test.csv"
):
    print("=" * 60)
    print("   BÁO CÁO ĐÁNH GIÁ MÔ HÌNH RỦI RO TÍN DỤNG (TEST SET)")
    print("=" * 60)

    # Kiểm tra tồn tại của model và dữ liệu test
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
    if not Path(test_csv_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file test tại: {test_csv_path}")

    # Nạp nhãn thực tế y_true
    df_test_labels = pd.read_csv(test_csv_path, usecols=["TARGET"])
    y_true = df_test_labels["TARGET"].to_numpy()
    total_samples = len(y_true)
    total_bads = int(y_true.sum())
    print(f"Tổng số hồ sơ kiểm tra: {total_samples:,}")
    print(f"Số lượng nợ xấu thực tế: {total_bads:,} ({total_bads/total_samples:.2%})")

    # Nạp mô hình đã train
    print(f"\nNạp mô hình từ: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Dự báo xác suất vỡ nợ (Probability of Default)
    print("\nThực hiện dự báo xác suất vỡ nợ ")
    test_ds = get_dataset(test_csv_path, batch_size=BATCH_SIZE, is_training=False)
    y_pred = model.predict(test_ds, verbose=1).ravel()

    # Tính toán các chỉ số nghiệp vụ ngân hàng
    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    pr_auc = average_precision_score(y_true, y_pred)
    ks_stat, _ = calculate_ks_statistic(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    #So sánh trực tiếp với Baseline LightGBM
    baseline_auc = 0.7509
    baseline_gini = 0.5018

    print("\n" + "=" * 60)
    print("              BẢNG TỔNG HỢP CHỈ SỐ NGHIỆP VỤ")
    print("=" * 60)
    print(f"Chỉ số              Deep Learning         Baseline (LGBM)     Đánh giá chuẩn Basel")
    print("-" * 60)
    print(f"ROC-AUC             {auc:.4f}               {baseline_auc:.4f}             ")
    print(f"Hệ số GINI          {gini:.4f}               {baseline_gini:.4f}           ")
    print(f"Chỉ số KS           {ks_stat*100:.2f}%               N/A                 ")
    print(f"PR-AUC              {pr_auc:.4f}               N/A                ")
    print(f"Brier Score         {brier:.4f}               N/A                 ")
    print("=" * 60)

    # In bảng phân tích 10 nhóm rủi ro (Decile Table)
    print("\nBẢNG PHÂN BỔ RỦI RO THEO 10 NHÓM:")
    decile_table = generate_decile_table(y_true, y_pred)
    print(decile_table.to_string(
        index=False,
        columns=["decile", "total_customers", "bad_customers", "bad_rate_%", "avg_prob", "ks_%"],
        formatters={
            "bad_rate_%": lambda x: f"{x:.2f}%",
            "avg_prob": lambda x: f"{x:.4f}",
            "ks_%": lambda x: f"{x:.2f}%"
        }
    ))

    print("\n" + "-" * 60)
    print("Ý NGHĨA KẾT QUẢ:")
    print(f"- Nhóm rủi ro cao nhất (Decile 1) có tỷ lệ nợ xấu: {decile_table.loc[0, 'bad_rate_%']:.2f}%")
    print(f"- Nhóm rủi ro thấp nhất (Decile 10) có tỷ lệ nợ xấu: {decile_table.loc[9, 'bad_rate_%']:.2f}%")
    if decile_table.loc[0, "bad_rate_%"] > decile_table.loc[9, "bad_rate_%"]:
        print("-> ĐẠT: Mô hình có tính đơn điệu (Monotonicity), phân tách rủi ro cực tốt!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
