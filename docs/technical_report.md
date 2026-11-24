# Báo cáo Kỹ thuật: Dự báo Rủi ro Tín dụng & Đối chuẩn Ba Mô hình
**Dự án:** Home Credit Default Risk Prediction & Real-Time Scoring System  
**Thời gian:** Tháng 11/2026  
**Môi trường:** Python 3.12, TensorFlow 2.21 (yêu cầu tối thiểu 2.15), LightGBM 4.7, scikit-learn 1.8, pandas 3.0, FastAPI 0.162

---

## Tóm tắt
Báo cáo trình bày việc xây dựng và đối chuẩn ba mô hình dự báo xác suất vỡ nợ (Probability of Default - PD) trên bộ dữ liệu Home Credit Default Risk (307,511 hồ sơ):
1. **WoE Scorecard:** mô hình bảng điểm truyền thống của ngành tín dụng, giải thích được từng điểm cộng/trừ.
2. **Deep MLP (Keras):** mạng nơ-ron có các lớp tiền xử lý (`Normalization`, `StringLookup`) nằm trong đồ thị mô hình.
3. **LightGBM:** gradient boosting tối ưu log-loss không trọng số.

**Kết quả chính (cùng tập test 30,752 hồ sơ):**
- LightGBM cao nhất (AUC 0.7509, Gini 0.5018, KS 38.12%), Keras (AUC 0.7472) và Scorecard (AUC 0.7374) theo sau.
- Về hiệu chuẩn xác suất, LightGBM đạt Brier 0.0683, thấp hơn mô hình hằng số (0.0742); Keras đạt 0.2079 (cao hơn mô hình hằng số) do huấn luyện bằng `class_weight`, nên đầu ra của Keras chỉ phù hợp để xếp hạng rủi ro cho đến khi được hiệu chuẩn.
- Ngưỡng cắt tối ưu theo ma trận chi phí - lợi nhuận là T\* = 0.70 (chọn trên test), lợi nhuận tăng 15.3% so với ngưỡng 0.50; chọn trên tập valid cho 0.71.
- Phân tích bổ sung (chưa có script trong repo, xem mục 4.2): bootstrap ghép cặp cho thấy Keras và LightGBM đều hơn Scorecard có ý nghĩa, còn LightGBM so với Keras sát ranh giới.

---

## 1. Dữ liệu & Chiến lược Chia tập

### 1.1. Phân phối dữ liệu
Bộ dữ liệu `application_train.csv` có 307,511 hồ sơ và 122 cột; sau sàng lọc, dự án dùng 26 đặc trưng (16 số, 10 danh mục). Biến mục tiêu `TARGET` = 1 là khách gặp khó khăn trả nợ.
- Nhóm tốt (`TARGET = 0`): 282,686 hồ sơ (91.93%).
- Nhóm xấu (`TARGET = 1`): 24,825 hồ sơ (8.07%).
- Tỷ lệ mất cân bằng khoảng **11.4 : 1**.

### 1.2. Chia phân tầng
Dữ liệu được chia phân tầng theo `TARGET` với seed cố định (23102026) thành ba tập riêng, tạo bởi `python -m src.preprocess`:

| Tập | Số dòng | Tỷ lệ | Số ca xấu | Tỷ lệ xấu |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 249,083 | 81.0% | 20,108 | 8.073% |
| **Validation** | 27,676 | 9.0% | 2,234 | 8.072% |
| **Test** | 30,752 | 10.0% | 2,483 | 8.074% |
| **Tổng** | **307,511** | **100%** | **24,825** | **8.073%** |

**Nguyên tắc cách ly:** các bước làm sạch theo từng dòng (ví dụ đổi `DAYS_BIRTH` sang `AGE`, tạo biến tỷ lệ) không dùng thống kê toàn tập. Mọi bước có **học tham số** (trung vị để điền khuyết, bảng mã hóa danh mục, ngưỡng cắt bin và WoE, trung bình/phương sai của lớp `Normalization`) chỉ được `fit` trên tập Train; Valid và Test chỉ `transform`.

---

## 2. Mô hình 1: WoE Scorecard

### 2.1. Weight of Evidence và Information Value
Mỗi biến được chia bin (cây quyết định đơn biến, tối đa 6 bin, bin "Missing" riêng cho giá trị khuyết). Với bin $i$ trong tổng $k$ bin:

$$\text{WoE}_i = \ln\left(\frac{\text{dist\_good}_i}{\text{dist\_bad}_i}\right), \quad \text{dist\_good}_i = \frac{\text{Good}_i + \alpha}{\text{Total Good} + \alpha k},\ \text{dist\_bad}_i = \frac{\text{Bad}_i + \alpha}{\text{Total Bad} + \alpha k}$$

với $\alpha = 0.5$ là hệ số làm trơn Laplace. Lưu ý: $\text{dist\_good}_i$ là tỷ lệ **khách tốt rơi vào bin $i$ trên tổng khách tốt**, không phải xác suất có điều kiện $P(\text{Good} \mid \text{bin}_i)$.

$$\text{IV} = \sum_{i=1}^k (\text{dist\_good}_i - \text{dist\_bad}_i) \times \text{WoE}_i$$

Dự án dùng quy tắc kinh nghiệm phổ biến trong chấm điểm tín dụng (không phải chuẩn quy định): $\text{IV} < 0.02$ không có khả năng dự báo; $0.02$ đến $0.1$ yếu; $0.1$ đến $0.3$ trung bình; $0.3$ đến $0.5$ mạnh; trên $0.5$ cần kiểm tra rò rỉ. Với ngưỡng $\text{IV} \ge 0.02$, 17 đặc trưng được giữ lại (ví dụ `PAYMENT_RATE` đạt IV 0.1315 với Tree Binning).

### 2.2. Hồi quy logistic và thang điểm
Hồi quy logistic dự báo $\text{logit}\,P(\text{Bad})$ trên 17 biến WoE. Vì WoE cao nghĩa là an toàn hơn, hệ số **được kỳ vọng âm**. Khi chạy thật, 14/17 hệ số âm và 3 hệ số dương (`AGE`, `AMT_GOODS_PRICE`, `DAYS_EMPLOYED_ANOM`); mã đánh dấu các hệ số dương (`sign_ok = False` trong `check_sign_consistency`) vì có thể do đa cộng tuyến. Xem `python -m src.scorecard`.

Điểm tín dụng được tuyến tính hóa theo quy ước phổ biến trong sách về scorecard: điểm gốc 600 tại odds tốt:xấu 50:1, PDO = 20 (cứ odds gấp đôi thì điểm tăng 20):

$$\text{Factor} = \frac{\text{PDO}}{\ln 2} \approx 28.8539, \qquad \text{Offset} = 600 - \text{Factor} \times \ln 50 \approx 487.123$$
$$\text{Score} = \text{Offset} - \text{Factor} \times \ln\left(\frac{P(\text{Bad})}{P(\text{Good})}\right) = \text{Offset} + \text{Factor} \times \ln\left(\frac{P(\text{Good})}{P(\text{Bad})}\right)$$

Hệ số `Offset` và intercept được chia đều cho $n$ đặc trưng nên điểm của từng đặc trưng cộng lại đúng bằng điểm tổng (`score_breakdown`). Nhờ đó có thể chỉ ra các đặc trưng làm giảm điểm nhiều nhất cho từng hồ sơ.

---

## 3. Mô hình 2: Mạng nơ-ron (Keras)

### 3.1. Nạp dữ liệu bằng tf.data
`tf.data.Dataset.list_files` đọc CSV, `tf.io.decode_csv` giải mã từng dòng, rồi `shuffle` (bộ đệm 5,000 dòng khi huấn luyện), `batch(1024)` và `prefetch(tf.data.AUTOTUNE)`. Cách này đọc theo luồng, không nạp toàn bộ bảng vào bộ nhớ.

### 3.2. Tiền xử lý nằm trong mô hình
- 16 biến số: lớp `tf.keras.layers.Normalization` (chuẩn hóa z-score, `adapt` trên Train).
- 10 biến danh mục: `tf.keras.layers.StringLookup(output_mode="one_hot")` (từ điển `adapt` trên Train).

Nhờ vậy đầu vào thô đi qua đúng biến đổi lúc huấn luyện và lúc phục vụ, đối với các đặc trưng thô. Bốn biến dẫn xuất (`EXT_SOURCES_MEAN`, `PAYMENT_RATE`, `INCOME_CREDIT_PERC`, `ANNUITY_INCOME_PERC`) **không** nằm trong mô hình mà do phía gọi tính (xem mục 6.2).

### 3.3. Kiến trúc và huấn luyện
- Mạng: `Dense(128, ReLU)` → `BatchNormalization` → `Dropout(0.5)` → `Dense(64, ReLU)` → `BatchNormalization` → `Dropout(0.3)` → `Dense(32, ReLU)` → `BatchNormalization` → `Dropout(0.2)` → `Dense(1, Sigmoid)`.
- Tối ưu: Adam; hàm mất mát `BinaryCrossentropy`, kèm `class_weight` tính từ tỷ lệ lớp của tập Train (nhóm nợ xấu có trọng số lớn hơn); tối đa 20 epoch, `EarlyStopping` (patience = 5) và `ModelCheckpoint`.

---

## 4. Đo lường Đối chuẩn

### 4.1. Kết quả trên tập Test (30,752 hồ sơ, 2,483 ca xấu)

| Chỉ số | WoEScorecard | Keras | LightGBM | Lệnh tái lập |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.7374 | 0.7472 | **0.7509** | `python -m src.scorecard_eval`, `python src/evaluate.py`, `python -m src.lgbm_baseline` |
| **Gini** | 0.4747 | 0.4944 | **0.5018** | $2 \times \text{AUC} - 1$ |
| **KS** | 36.24% | 37.60% | **38.12%** | như trên |
| **PR-AUC** | chưa đo | 0.2191 | **0.2324** | như trên |
| **Brier** | chưa đo | 0.2079 | **0.0683** | như trên (mô hình hằng số: 0.0742) |
| **Tính giải thích** | Rất cao | Hộp đen | Cần SHAP / độ quan trọng đặc trưng | |

PR-AUC của một mô hình ngẫu nhiên bằng tỷ lệ xấu (0.0807), nên 0.2191 và 0.2324 gấp khoảng 2.7 và 2.9 lần mức ngẫu nhiên.

### 4.2. Phân tích bổ sung (chưa có script trong repo)
Các số dưới đây được tính bằng script phân tích tạm thời chưa nằm trong repo; hãy coi là kết quả tham khảo.

**Bootstrap ghép cặp 300 lần trên test** (seed 42; sai số chuẩn của một lần đo AUC khoảng 0.006, nên so không ghép cặp quá thận trọng khi các mô hình được đo trên cùng một tập):

| Chênh lệch AUC | Giá trị | Khoảng tin cậy 95% | Kết luận |
| :--- | :---: | :---: | :--- |
| Keras − Scorecard | +0.0098 | (0.0050, 0.0143) | có ý nghĩa |
| LightGBM − Keras | +0.0036 | (0.0000, 0.0074) | sát ranh giới, chưa kết luận chắc |
| LightGBM − Scorecard | +0.0134 | (0.0095, 0.0175) | có ý nghĩa |

### 4.3. Thí nghiệm rò rỉ nhãn và đối chứng âm (Scorecard)
- **Rò rỉ nhãn:** khi cố tình `fit` WoE và hồi quy trên Train + Valid (276,759 hồ sơ) rồi đo trên **Valid**, AUC tăng từ 0.7395 lên 0.7433 (+0.0038). Mức tăng nhỏ vì mô hình ít bậc tự do và Train đã lớn, nhưng cho thấy việc dùng nhãn của tập đánh giá làm số đo bị lạc quan. Xem `python -m src.scorecard_eval`.
- **Xáo trộn nhãn:** huấn luyện trên nhãn Train bị hoán vị ngẫu nhiên rồi đo trên Valid. Seed 42 trong `python -m src.scorecard_eval` cho AUC 0.4385; chạy thêm 5 seed khác (phân tích bổ sung) cho 0.5217, 0.4615, 0.4973, 0.4932, 0.5236, trung bình 0.4995, độ lệch chuẩn 0.0226. Mô hình học nhãn ngẫu nhiên là một hàm ngẫu nhiên của đặc trưng nên AUC dao động quanh 0.5. Kết luận có điều kiện: không thấy kênh rò rỉ ngầm; thí nghiệm này không chứng minh tuyệt đối.

---

## 5. Hiệu chuẩn Xác suất & Tối ưu Ngưỡng

### 5.1. Hiệu chuẩn và Brier Score
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
Mô hình hằng số luôn dự đoán $\bar{y} = 0.0807$ có Brier $\bar{y}(1-\bar{y}) \approx 0.0742$.
- **LightGBM** tối ưu log-loss không trọng số: Brier 0.0683 (thấp hơn mô hình hằng số). Kiểm tra bổ sung: xác suất dự báo trung bình trên test là 0.0810, sát tỷ lệ xấu thực tế 0.0807.
- **Keras** dùng `class_weight`: Brier 0.2079 (cao hơn mô hình hằng số); xác suất dự báo trung bình trên test là 0.4239 (kiểm tra bổ sung), nên đầu ra chỉ có ý nghĩa xếp hạng cho đến khi được hiệu chuẩn (Platt scaling hoặc isotonic regression).

### 5.2. Ma trận chi phí - lợi nhuận
Với giả định thu lãi 0.15 trên mỗi khoản vay tốt được duyệt và mất 0.75 vốn trên mỗi khoản vay xấu được duyệt (tỷ lệ 5:1), lợi nhuận ròng ở ngưỡng $T$ (duyệt khi $\hat{p} < T$) là:
$$\text{net\_profit}(T) = 0.15 \times \text{(khách tốt được duyệt)} - 0.75 \times \text{(khách xấu được duyệt)}$$
Từ chối thì lợi nhuận bằng 0. Kết quả từ `python -m src.repro_keras`:
- **Ngưỡng 0.50:** duyệt 63.35%, nợ xấu trong danh mục 3.69%, lợi nhuận 2,274.9 đơn vị.
- **T\* = 0.70** (quét trên tập test): duyệt 87.12%, nợ xấu 5.79%, lợi nhuận 2,623.8 đơn vị (+15.3% so với 0.50). Vì ngưỡng được chọn trên chính tập dùng để báo cáo nên con số hơi lạc quan.
- **T\* chọn trên Valid:** 0.71 (duyệt 88.16%, nợ xấu 5.83%, lợi nhuận 2,379.3 đơn vị **trên valid**, không so sánh trực tiếp với số của test).
- **Độ ổn định** (phân tích bổ sung): áp 0.71 lên test cho lợi nhuận khoảng 2,622.8, chênh khoảng 1 đơn vị (0.04%) so với 2,623.8; lấy mẫu lại 200 lần trên test, T\* tối ưu dao động ±0.02 (0.68 đến 0.74). Vì vậy 0.70 và 0.71 không khác nhau có ý nghĩa.

---

## 6. Phục vụ qua API & Hạn chế Đã biết

### 6.1. FastAPI
`src/api.py` nạp mô hình và `config/threshold_config.json` khi khởi động (`lifespan`).
- `POST /predict`: nhận hồ sơ, tính PD bằng mô hình Keras, đổi sang điểm $300 + \text{clip}\left(\frac{\ln\frac{1-\text{PD}}{\text{PD}} + 4}{8}, 0, 1\right) \times 550$, rồi phân nhóm theo điểm: $\ge 750$ rất thấp, $\ge 670$ thấp, $\ge 580$ trung bình (xem xét thủ công), còn lại cao (từ chối).
- `GET /health`: trạng thái dịch vụ.
- 11 test riêng cho API (hợp đồng, kiểm tra kiểu dữ liệu trả mã 422, giá trị phi lý, mô hình giả để kiểm các nhánh điểm, nhất quán công thức với `pd_to_credit_score`, vai trò của ngưỡng). Toàn bộ dự án có 57 test.

### 6.2. Hạn chế đã biết
1. **Keras chưa hiệu chuẩn:** `probability_of_default` của API chỉ có ý nghĩa xếp hạng.
2. **API chưa dùng ngưỡng tối ưu để quyết định:** `optimal_threshold` chỉ được trả về trong `threshold_used`; quyết định dựa trên các mốc điểm cứng 750/670/580. Trên test (phân tích bổ sung): duyệt tự động 15.6% hồ sơ, xem xét thủ công 46.0%, từ chối 38.4%; ngưỡng T\* = 0.70 duyệt 87.12%. Các mốc điểm được đặt như cho xác suất đã hiệu chuẩn, còn PD của Keras bị đẩy lên.
3. **Bốn biến dẫn xuất do client tính:** nếu client bỏ sót, API dùng giá trị mặc định (ví dụ trong README: điểm giảm từ 684 xuống 630).
4. **Thiếu kiểm tra miền giá trị** (`AGE = -5` vẫn trả 200); API và test phải chạy từ thư mục gốc vì đường dẫn trong `lifespan` là tương đối.
5. **Dữ liệu không có chiều thời gian:** chưa kiểm tra độ ổn định theo thời gian (ví dụ PSI).
6. **`evaluate.py`** còn hằng số LightGBM ghi cứng; số này được tái lập bằng `python -m src.lgbm_baseline`.
7. **Phân tích bổ sung ở mục 4.2, 4.3, 5.1, 5.2 và 6.2 chưa có script trong repo.**

---

## 7. Tái lập Kết quả
Chạy từ thư mục gốc dự án.

```bash
# 0. Cài đặt và kiểm thử (không cần dữ liệu)
pip install -r requirements.txt
python -m pytest tests -q                 # 57 passed

# 1. Dữ liệu: đặt application_train.csv vào data/raw/ rồi sinh train/valid/test
python -m src.preprocess

# 2. Scorecard (kèm thí nghiệm rò rỉ và xáo trộn nhãn)
python -m src.scorecard_eval
python -m src.scorecard                   # kiểm tra dấu hệ số và bảng điểm

# 3. Keras (dùng models/best_model.keras có sẵn)
python src/evaluate.py

# 4. LightGBM
python -m src.lgbm_baseline

# 5. Ngưỡng chi phí (ghi vào docs/results/, không đụng config/)
python -m src.repro_keras
```
Không chạy `python src/threshold_optimization.py` với đối số mặc định: nó ghi đè `config/threshold_config.json` mà API đọc.
