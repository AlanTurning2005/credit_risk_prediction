import numpy as np
import json
from pathlib import Path
from src.threshold_optimization import optimize_threshold



def net_profit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    profit_margin: float = 0.15,
    lgd: float = 0.75,
) -> float:
    """Tính lợi nhuận ròng: khách tốt duyệt * margin - khách xấu duyệt * LGD."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    approved = y_pred_arr < threshold
    approved_goods = np.sum(approved & (y_true_arr == 0))
    approved_bads = np.sum(approved & (y_true_arr == 1))

    return float(approved_goods * profit_margin - approved_bads * lgd)


def reproduce_threshold_test(output_path: str = "docs/results/threshold_repro_test.json") -> dict:
    """Tái lập tối ưu ngưỡng trên test.csv và lưu an toàn vào docs/results/."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = optimize_threshold(
        test_csv_path="data/processed/test.csv",
        output_config_path=str(out_file),
    )
    return cfg

def compare_configs(
    repro_path: str = "docs/results/threshold_repro_test.json",
    original_path: str = "config/threshold_config.json"
) -> bool:
    """So sánh nội dung giữa file tái lập và file cấu hình gốc (chỉ đọc)."""
    with open(repro_path, "r", encoding="utf-8") as f1, open(original_path, "r", encoding="utf-8") as f2:
        repro_cfg = json.load(f1)
        orig_cfg = json.load(f2)

    is_identical = (repro_cfg == orig_cfg)
    print(f"Cấu hình tái lập so với gốc: {'HOÀN TOÀN TRÙNG KHỚP' if is_identical else 'CÓ ĐIỂM KHÁC BIỆT'}")
    print(f" - Optimal Threshold: {repro_cfg.get('optimal_threshold')} (gốc: {orig_cfg.get('optimal_threshold')})")
    print(f" - Expected Approval Rate: {repro_cfg.get('expected_approval_rate_%')}% (gốc: {orig_cfg.get('expected_approval_rate_%')}%)")
    return is_identical

def reproduce_threshold_valid(output_path: str = "docs/results/threshold_from_valid.json") -> dict:
    """Tối ưu ngưỡng trên valid.csv (chuẩn phương pháp luận, không rò rỉ test)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = optimize_threshold(
        test_csv_path="data/processed/valid.csv",
        output_config_path=str(out_file),
    )
    return cfg



def main():
    print("\n--- B2: TÁI LẬP NGƯỠNG TRÊN TEST (KIỂM TRA GỐC) ---")
    cfg_test = reproduce_threshold_test()
    compare_configs()

    print("\n--- M1: TỐI ƯU NGƯỠNG TRÊN VALID (CHUẨN ML) ---")
    cfg_valid = reproduce_threshold_valid()
    print(f"Ngưỡng tối ưu chọn từ VALID: T* = {cfg_valid['optimal_threshold']:.2f}")
    print(f"Ngưỡng tối ưu chọn từ TEST : T* = {cfg_test['optimal_threshold']:.2f}")

if __name__ == "__main__":
    main()