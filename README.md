# Home Credit Default Risk Prediction & Real-Time Scoring System
### *Hệ thống Đánh giá Rủi ro Tín dụng, Dự báo Nợ xấu & Phê duyệt Khoản vay Tự động*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-57%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. Giới thiệu Dự án (Project Overview)
Dự án xây dựng một pipeline học máy đầu-cuối cho bài toán **Dự báo Xác suất Vỡ nợ (Probability of Default - PD)** và **Chấm điểm Tín dụng (thang điểm 300 - 850)**, sử dụng bộ dữ liệu từ cuộc thi **[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)** (307,511 hồ sơ, 24,825 ca nợ xấu, tỷ lệ 8.07%).

Dự án giải quyết bài toán theo cách làm của ngành Ngân hàng & Fintech:
1. **Nạp dữ liệu theo luồng:** dùng `tf.data.Dataset` đọc CSV theo từng batch, không nạp cả bảng vào bộ nhớ.
2. **Đối chuẩn 3 mô hình:** so sánh **Scorecard truyền thống (Weight of Evidence + Logistic Regression)**, **mạng nơ-ron Keras (Deep MLP)** và **Gradient Boosting (LightGBM)** trên cùng một tập test độc lập (30,752 hồ sơ).
3. **Đóng gói tiền xử lý vào mô hình Keras:** các lớp `Normalization` và `StringLookup` nằm trong đồ thị mô hình, giảm nguy cơ lệch giữa huấn luyện và phục vụ (Train/Serving Skew) đối với các đặc trưng thô.
4. **Kiểm định theo chỉ số ngân hàng:** **ROC-AUC**, **Gini**, **KS**, **bảng Decile**, **PR-AUC** và **Brier Score** (đo mức độ hiệu chuẩn xác suất).
5. **Tối ưu quyết định theo chi phí:** tìm ngưỡng cắt theo ma trận chi phí - lợi nhuận thay cho ngưỡng mặc định 0.5.
6. **Phục vụ qua FastAPI:** API ánh xạ xác suất PD sang thang điểm tín dụng và phân tầng rủi ro, có bộ kiểm thử tự động (57 test).

---

## 2. Kiến trúc Hệ thống (System Architecture)
```text
[Dữ liệu thô: application_train.csv]
       │
       ▼
[Chia phân tầng theo TARGET: Train 81% / Valid 9% / Test 10%, seed cố định]
       │
       ├──────────────────────────────┬──────────────────────────────┐
       ▼                              ▼                              ▼
[WoE Scorecard]               [Keras Deep MLP]               [LightGBM Baseline]
• WoE & IV (Tree Binning)     • tf.data đọc CSV theo luồng   • Median + One-hot Preprocessor
• Logistic Regression         • Normalization + StringLookup • Boosting, log-loss không trọng số
• Điểm cộng/trừ theo PDO      • class_weight khi huấn luyện
       │                              │                              │
       └──────────────────────────────┼──────────────────────────────┘
                                      ▼
                      [Đánh giá trên cùng Test Set: 30,752 hồ sơ]
                      • AUC, Gini, KS, PR-AUC, Brier
                                      │
                                      ▼
                      [Tối ưu ngưỡng theo chi phí - lợi nhuận]
                                      │
                                      ▼
                      [FastAPI: chấm điểm & quyết định]
```

---

## 3. Kết quả Thực nghiệm & Đối chuẩn Mô hình (Benchmark)
Tất cả mô hình được đánh giá trên cùng tập test: **30,752 hồ sơ, 2,483 ca nợ xấu (8.07%)**.

### 3.1. Bảng so sánh 3 mô hình
| Chỉ số | WoEScorecard | Keras Neural Net | LightGBM Baseline | Lệnh tái lập |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.7374 | 0.7472 | **0.7509** | `python -m src.scorecard_eval`, `python src/evaluate.py`, `python -m src.lgbm_baseline` |
| **Gini** ($2 \times \text{AUC} - 1$) | 0.4747 | 0.4944 | **0.5018** | tính từ AUC |
| **KS** | 36.24% | 37.60% | **38.12%** | như trên |
| **PR-AUC** | chưa đo | 0.2191 | **0.2324** | như trên |
| **Brier Score** | chưa đo | 0.2079 | **0.0683** | như trên (mô hình hằng số: 0.0742) |
| **Tính giải thích** | Rất cao (bảng điểm) | Hộp đen | Cần SHAP / độ quan trọng đặc trưng | đặc tính thuật toán |

Mô hình hằng số luôn dự đoán tỷ lệ nợ xấu 0.0807 có Brier $0.0807 \times 0.9193 \approx 0.0742$.

### 3.2. Phân tích Kỹ thuật
- **Khả năng phân loại:** LightGBM cao nhất về AUC, KS và PR-AUC; Keras theo sau; scorecard thấp nhất nhưng giải thích được từng điểm cộng/trừ. Chênh lệch giữa các mô hình nhỏ (khoảng 0.004 đến 0.013 AUC); sai số chuẩn của một lần đo AUC trên tập test khoảng 0.006, nên cần kiểm định ghép cặp trước khi kết luận (xem mục 3.5).
- **Hiệu chuẩn xác suất:** LightGBM tối ưu log-loss không trọng số nên Brier (0.0683) thấp hơn mô hình hằng số (0.0742). Keras dùng `class_weight` khi huấn luyện nên Brier (0.2079) **cao hơn** mô hình hằng số: xác suất của Keras chỉ nên dùng để xếp hạng rủi ro, chưa phải PD, cho đến khi qua bước hiệu chuẩn (Platt scaling hoặc isotonic regression).
- **Ứng viên làm PD:** LightGBM là ứng viên phù hợp nhất về mặt hiệu chuẩn, nhưng cần thêm kiểm tra (độ ổn định theo thời gian, khả năng giải thích) trước khi dùng trong phục vụ.

### 3.3. Bảng phân tích 10 nhóm rủi ro (Decile Analysis - Keras)
Kết quả từ `python src/evaluate.py`: tỷ lệ nợ xấu thực tế giảm đơn điệu từ nhóm rủi ro cao nhất về nhóm an toàn nhất:
```text
Decile 1 (10% rủi ro cao nhất):  Tỷ lệ nợ xấu = 25.39%  ▲ (gấp khoảng 18 lần Decile 10)
Decile 2:                       Tỷ lệ nợ xấu = 15.12%  │
Decile 3:                       Tỷ lệ nợ xấu = 10.96%  │
Decile 4:                       Tỷ lệ nợ xấu =  8.49%  │
Decile 5:                       Tỷ lệ nợ xấu =  5.53%  │
Decile 6:                       Tỷ lệ nợ xấu =  5.30%  │
Decile 7:                       Tỷ lệ nợ xấu =  3.35%  │
Decile 8:                       Tỷ lệ nợ xấu =  2.60%  │
Decile 9:                       Tỷ lệ nợ xấu =  2.57%  │
Decile 10 (10% an toàn nhất):    Tỷ lệ nợ xấu =  1.43%  ▼
```

### 3.4. Tối ưu Ngưỡng cắt theo Chi phí - Lợi nhuận
Giả định: thu lãi 0.15 trên mỗi khoản vay tốt được duyệt, mất 0.75 vốn trên mỗi khoản vay xấu được duyệt (tỷ lệ 5:1). Kết quả từ `python -m src.repro_keras`:
- **Ngưỡng mặc định (0.50):** duyệt 63.35%, nợ xấu trong danh mục 3.69%, lợi nhuận **2,274.9 đơn vị**.
- **Ngưỡng tối ưu T\* = 0.70** (quét trên chính tập test): duyệt 87.12%, nợ xấu trong danh mục 5.79%, lợi nhuận **2,623.8 đơn vị** (+15.3% so với ngưỡng 0.50). Vì ngưỡng được chọn trên tập dùng để báo cáo nên con số này hơi lạc quan.
- **Chọn ngưỡng trên tập Validation:** T\* = 0.71 (duyệt 88.16%, nợ xấu 5.83%, lợi nhuận 2,379.3 đơn vị **trên valid**; không so sánh trực tiếp với số của tập test). Quy trình đúng là chọn ngưỡng trên valid rồi đo trên test.

### 3.5. Phân tích bổ sung (chưa có script trong repo)
Các số dưới đây được tính bằng script phân tích tạm thời **chưa nằm trong repo**, nên người đọc chưa tái lập trực tiếp được; hãy coi là kết quả tham khảo. Cùng mô hình, cùng tập test như trên.
- **Bootstrap ghép cặp 300 lần trên test (chênh lệch AUC, khoảng tin cậy 95%):** Keras − scorecard = +0.0098 (0.0050 đến 0.0143), có ý nghĩa; LightGBM − Keras = +0.0036 (0.0000 đến 0.0074), sát ranh giới nên chưa kết luận chắc; LightGBM − scorecard = +0.0134 (0.0095 đến 0.0175), có ý nghĩa.
- **Hiệu chuẩn:** xác suất dự báo trung bình trên test là 0.0810 với LightGBM và 0.4239 với Keras, trong khi tỷ lệ nợ xấu thực tế là 0.0807. Theo 10 nhóm, trung bình dự báo của LightGBM đi từ 0.0195 đến 0.2555 và tỷ lệ xấu thực tế từ 0.0150 đến 0.2601.
- **Rò rỉ nhãn và đối chứng âm (scorecard):** fit WoE trên train + valid rồi đo trên valid làm AUC tăng từ 0.7395 lên 0.7433 (+0.0038; số này có trong `python -m src.scorecard_eval`). Xáo trộn nhãn train với 5 seed khác (1 đến 5) cho AUC 0.5217, 0.4615, 0.4973, 0.4932, 0.5236 (trung bình 0.4995, độ lệch chuẩn 0.0226); seed 42 trong `src.scorecard_eval` cho 0.4385. Kết luận có điều kiện: không thấy kênh rò rỉ ngầm.
- **Độ ổn định của ngưỡng:** áp T\* = 0.71 (chọn từ valid) lên test cho lợi nhuận khoảng 2,622.8, chênh khoảng 1 đơn vị (0.04%) so với 2,623.8 của T\* = 0.70; lấy mẫu lại 200 lần trên test, T\* tối ưu dao động ±0.02 (0.68 đến 0.74).
- **Cơ cấu quyết định của API trên test:** theo các mốc điểm cứng 750/670/580, API duyệt tự động 15.6% hồ sơ (tỷ lệ nợ xấu 1.73%), xem xét thủ công 46.0%, từ chối 38.4%; ngưỡng T\* = 0.70 duyệt 87.12% hồ sơ.
- **Tái lập dữ liệu:** `python -m src.preprocess` sinh lại ba tệp `train/valid/test.csv` trùng từng byte (sha256) với dữ liệu dùng để đo các kết quả trên.

---

## 4. Cấu trúc Thư mục Dự án
```text
credit_risk_prediction/
├── config/
│   └── threshold_config.json      # Ngưỡng tối ưu và cấu hình phân tầng rủi ro (API đọc tệp này)
├── data/                          # Dữ liệu (bị .gitignore, xem mục 5)
│   ├── raw/                       # Dữ liệu gốc Kaggle
│   └── processed/                 # train.csv, valid.csv, test.csv
├── docs/
│   ├── technical_report.md        # Báo cáo kỹ thuật
│   ├── project_highlights.md      # Các ý chính của dự án
│   ├── credit_risk_domain_knowledge.md
│   ├── results/                   # Kết quả thực nghiệm (scorecard, Keras, LightGBM)
│   └── assets/                    # Bản lưu giao diện Swagger UI
├── models/
│   └── best_model.keras           # Mô hình Keras đã huấn luyện
├── notebooks/
│   ├── 01_explore_data.ipynb      # EDA, sàng lọc biến, WoE/IV
│   ├── 02_baseline_model.ipynb    # Baseline LightGBM
│   └── eda_report.html
├── src/
│   ├── config.py                  # Danh sách cột, hằng số
│   ├── preprocess.py              # Làm sạch, tạo biến, chia train/valid/test
│   ├── eda_stats.py               # Thống kê sàng lọc biến
│   ├── woe.py                     # WoE / IV (WoEEncoder)
│   ├── scorecard.py               # WoEScorecard (logistic + thang điểm PDO)
│   ├── scorecard_eval.py          # Đánh giá scorecard, thí nghiệm rò rỉ và xáo trộn nhãn
│   ├── dataset.py                 # Pipeline tf.data
│   ├── keras_preprocessing.py     # Normalization & StringLookup cho Keras
│   ├── model.py                   # Kiến trúc Deep MLP
│   ├── train.py                   # Huấn luyện Keras (class_weight, early stopping)
│   ├── evaluate.py                # Đánh giá Keras: AUC, Gini, KS, PR-AUC, Brier, Decile
│   ├── lgbm_baseline.py           # Baseline LightGBM
│   ├── threshold_optimization.py  # Tìm ngưỡng theo chi phí - lợi nhuận
│   ├── repro_keras.py             # Tái lập ngưỡng, ghi ra docs/results (không đụng config/)
│   └── api.py                     # FastAPI
├── tests/                         # 57 test: api, evaluate, threshold_optimization, scorecard,
│                                  # scorecard_eval, lgbm_baseline, repro_keras, eda_stats,
│                                  # preprocess, woe
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 5. Hướng dẫn Cài đặt & Tái lập (Quickstart)
> **Lưu ý:** mọi lệnh cần chạy từ **thư mục gốc của dự án** (API và test dùng đường dẫn tương đối tới `models/` và `config/`).

### Bước 1: Lấy mã nguồn và cài môi trường (Python 3.12)
Tải mã nguồn (`git clone` hoặc tải ZIP từ GitHub), vào thư mục dự án rồi:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy bộ kiểm thử (không cần dữ liệu)
```bash
python -m pytest tests -q
# Kết quả mong đợi: 57 passed
```

### Bước 3: Chuẩn bị dữ liệu
Dữ liệu không nằm trong repo. Tải `application_train.csv` từ cuộc thi [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (cần chấp nhận điều khoản của cuộc thi) vào `data/raw/`, rồi sinh các tập train/valid/test:
```bash
python -m src.preprocess
```
Lệnh này tạo `data/processed/train.csv`, `valid.csv`, `test.csv` (chia phân tầng 81/9/10, seed 23102026).

### Bước 4: Tái lập kết quả từng mô hình
```bash
# 1. WoE Scorecard (kèm thí nghiệm rò rỉ nhãn và xáo trộn nhãn):
python -m src.scorecard_eval
# 2. Keras (mô hình đã huấn luyện có sẵn trong models/):
python src/evaluate.py
# 3. LightGBM baseline:
python -m src.lgbm_baseline
# 4. Ngưỡng chi phí - lợi nhuận (chỉ ghi vào docs/results/, không ghi đè config/):
python -m src.repro_keras
```
> Không chạy trực tiếp `python src/threshold_optimization.py` với đối số mặc định: nó ghi đè `config/threshold_config.json` mà API đang đọc.

### Bước 5: Khởi chạy API
```bash
python src/api.py
```
Truy cập Swagger UI tại `http://127.0.0.1:8000/docs`.

#### Ví dụ Request và Response thật
Client **phải tự tính và gửi** bốn biến dẫn xuất (`EXT_SOURCES_MEAN`, `PAYMENT_RATE`, `INCOME_CREDIT_PERC`, `ANNUITY_INCOME_PERC`); nếu bỏ qua, API dùng giá trị mặc định và điểm có thể sai đáng kể.

**Request:** `POST /predict`
```json
{
  "EXT_SOURCE_1": 0.65,
  "EXT_SOURCE_2": 0.58,
  "EXT_SOURCE_3": 0.72,
  "AMT_INCOME_TOTAL": 250000.0,
  "AMT_CREDIT": 450000.0,
  "AMT_ANNUITY": 22000.0,
  "AMT_GOODS_PRICE": 400000.0,
  "AGE": 32.0,
  "YEARS_EMPLOYED": 6.0,
  "EXT_SOURCES_MEAN": 0.65,
  "PAYMENT_RATE": 0.04889,
  "INCOME_CREDIT_PERC": 0.55556,
  "ANNUITY_INCOME_PERC": 0.088,
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "M",
  "NAME_EDUCATION_TYPE": "Higher education",
  "OCCUPATION_TYPE": "Managers"
}
```
**Response** (chạy qua `TestClient`; độ trễ phụ thuộc máy và lần gọi, lần gọi đầu thường chậm hơn):
```json
{
  "decision": "APPROVED",
  "action_required": "APPROVE",
  "credit_score": 684,
  "risk_tier": "LOW_RISK",
  "probability_of_default": 0.1703,
  "threshold_used": 0.7,
  "inference_latency_ms": 95.68
}
```

---

## 6. Hạn chế đã biết & Hướng phát triển
1. **Keras chưa hiệu chuẩn xác suất.** Do dùng `class_weight`, Brier Score là 0.2079 (cao hơn mô hình hằng số 0.0742). Trường `probability_of_default` của API hiện chỉ mang ý nghĩa xếp hạng rủi ro, chưa phải PD theo chuẩn Basel nếu chưa hiệu chuẩn.
2. **API chưa dùng ngưỡng tối ưu để quyết định.** `optimal_threshold` chỉ được trả về trong `threshold_used`; quyết định dựa trên các mốc điểm cứng 750/670/580 (xem 3.5: khác nhiều so với ngưỡng T\* = 0.70).
3. **Bốn biến dẫn xuất do client tự tính.** Nếu client bỏ sót hoặc tính sai, API lấy giá trị mặc định (ví dụ trên: điểm giảm từ 684 xuống 630 khi bỏ qua).
4. **Chưa kiểm tra giá trị đầu vào theo miền nghiệp vụ.** Ví dụ `AGE = -5` vẫn nhận mã 200.
5. **Ngưỡng T\* = 0.70 trong 3.4 được chọn trên tập test**; chọn trên valid cho 0.71 (khác biệt nhỏ, xem 3.5).
6. **Dữ liệu không có chiều thời gian**, nên chưa kiểm tra được độ ổn định theo thời gian (ví dụ PSI).
7. **`evaluate.py` còn hằng số LightGBM ghi cứng** (AUC 0.7509, Gini 0.5018); số có thể kiểm chứng bằng `python -m src.lgbm_baseline`.
8. **Các số ở mục 3.5 chưa có script trong repo.**

**Hướng phát triển:**
- Hiệu chuẩn Keras (Platt scaling hoặc isotonic regression) hoặc dùng LightGBM làm mô hình tính PD chính.
- API tự tính bốn biến dẫn xuất, thêm ràng buộc miền giá trị và dùng ngưỡng tối ưu để quyết định.
- Sửa `WoEScorecard`: nhận danh sách cột làm tham số; đổi tên phương thức `score` để không xung đột với giao diện scikit-learn.
- Viết script tái lập các phân tích ở mục 3.5.
