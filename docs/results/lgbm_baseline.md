# Kết quả Tái lập Mô hình Baseline LightGBM & Bảng So sánh 3 Mô hình

- **Ngày thực hiện:** 2026-11-23
- **Môi trường:** Python 3.12, LightGBM 4.7.0, Scikit-Learn 1.8.0
- **Lệnh thực thi:** `python -m src.lgbm_baseline`
- **Cấu hình mô hình:** `n_estimators=100`, `learning_rate=0.04`, `random_state=23102026`, `n_jobs=-1`
- **Quy trình tiền xử lý:** 
  - Biến số (16 cột): `SimpleImputer(strategy="median")`
  - Biến danh mục (10 cột): `SimpleImputer(strategy="constant", fill_value="Missing")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
  - Chỉ `fit` trên tập Train, tập Valid và Test chỉ `transform` (chống rò rỉ dữ liệu).

---

## 1. Kết quả Đánh giá LightGBM trên 3 Tập Dữ liệu

| Tập dữ liệu | Số dòng | ROC-AUC | Hệ số GINI | Chỉ số KS (%) | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 249,083 | **0.7719** | **0.5438** | 40.53% | 0.2658 | 0.0670 |
| **Valid** | 27,676 | **0.7573** | **0.5146** | 38.06% | 0.2437 | 0.0679 |
| **Test** | 30,752 | **0.7509** | **0.5018** | 38.12% | 0.2324 | **0.0683** |

### Đối chiếu tính tái lập:
- Khớp 100% với `notebooks/02_baseline_model.ipynb` (Train AUC 0.7719, Valid AUC 0.7573, Test AUC 0.7509).
- Khớp 100% với các hằng số tham chiếu trong `src/evaluate.py` và `README.md` (AUC 0.7509, Gini 0.5018).

---

## 2. Bảng So sánh Tổng hợp Cả 3 Mô hình trên Cùng Tập Test (30,752 dòng)

| Chỉ số / Đặc tính | WoEScorecard | Keras Neural Net | LightGBM Baseline | Nguồn số liệu |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.7374 | 0.7472 | **0.7509** | `docs/results/{scorecard_metrics, keras_repro, lgbm_baseline}.md` |
| **Hệ số GINI** | 0.4747 | 0.4944 | **0.5018** | $2 \times \text{AUC} - 1$ |
| **Chỉ số KS** | 36.24% | 37.60% | **38.12%** | Tính từ phân phối điểm/xác suất |
| **PR-AUC** | - | 0.2191 | **0.2324** | Tính trên đường cong Precision-Recall |
| **Brier Score** | - | 0.2079 | **0.0683** | Đo mức độ hiệu chuẩn xác suất ($\hat{p} \text{ vs } y$) |
| **Khả năng giải thích** | Rất cao (Bảng điểm cộng/trừ) | Hộp đen | Cần SHAP/Feature Importance | Đặc tính thuật toán |
| **Trạng thái mã nguồn** | Đã tái lập (`scorecard_eval.py`) | Đã tái lập (`repro_keras.py`) | Đã tái lập (`lgbm_baseline.py`) | Toàn bộ đều có mã nguồn và test |

---

## 3. Nhận xét Phân tích Kỹ thuật

1. **Về Khả năng Phân loại (Discrimination):**
   - LightGBM đạt hiệu năng cao nhất trên cả 3 chỉ số AUC ($0.7509$), KS ($38.12\%$), và PR-AUC ($0.2324$).
   - Tuy nhiên, chênh lệch AUC giữa LightGBM và Keras chỉ là $+0.0037$ ($0.7509 - 0.7472$), nhỏ hơn sai số chuẩn của AUC trên tập test ($SE \approx 0.006$). Do đó chưa đủ bằng chứng thống kê để kết luận LightGBM vượt trội hơn hẳn Keras.
2. **Về Hiệu chuẩn Xác suất (Calibration):**
   - LightGBM có Brier Score xuất sắc ($0.0683$), thấp hơn mức ngây thơ $p(1-p) = 0.0742$. Xác suất dự báo của LightGBM bám sát tần suất thực tế (~8%), đủ điều kiện dùng làm Xác suất Vỡ nợ (PD).
   - Keras có Brier Score tệ ($0.2079$) do kỹ thuật gán trọng số phạt lớp (`class_weight`), không thể dùng trực tiếp làm PD ngân hàng nếu chưa qua hiệu chuẩn hậu xử lý.
3. **Về Mức độ Quá khớp (Overfitting):**
   - Khoảng cách Train-Test của LightGBM là $\Delta \text{AUC} = 0.7719 - 0.7509 = 0.0210$ (mức suy thoái bình thường của mô hình boosting).
   - WoEScorecard có mức suy thoái nhỏ nhất ($\Delta \text{AUC} = 0.7450 - 0.7374 = 0.0076$) do mô hình tuyến tính có bậc tự do thấp, tính ổn định cao.