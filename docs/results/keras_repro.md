# Tái lập số liệu mô hình Keras và kiểm tra chọn ngưỡng

Ngày: 2026-11-22. Suy luận xác định (không huấn luyện lại, không có ngẫu nhiên).
Mô hình: `models/best_model.keras` (sha256 `85d806bdaf39235525be38af7a8aa8bd150c6b10526c7c7c8a2203a01a748eba`).
Dữ liệu: `data/processed/test.csv` (30,752 hồ sơ, 2,483 nợ xấu, 8.07%) và `valid.csv` (27,676 hồ sơ).
Lệnh: `python src/evaluate.py` và `python -m src.repro_keras`.

Ký hiệu nguồn: **[B]** = output của lệnh nêu ở đầu tài liệu; **[S]** = scorecard, lệnh `python -m src.scorecard_eval` (seed 42); **[P]** = phân tích bổ sung bằng script ngoài repo (chưa có trong repo, tham khảo).

## 1. Keras chạy lại so với README (tập test) [B]

| Chỉ số | README | Chạy lại | Kết luận |
| --- | --- | --- | --- |
| ROC-AUC | 0.7472 | 0.7472 | khớp |
| Gini | 0.4944 | 0.4944 | khớp |
| KS | 37.60% | 37.60% | khớp |
| PR-AUC | 0.2191 | 0.2191 | khớp |
| Brier | 0.2079 | 0.2079 | khớp |

Bảng decile cũng khớp README (Decile 1: tỷ lệ xấu 25.39%; Decile 10: 1.43%). Chạy lại độc lập [P]: test AUC 0.7472, Brier 0.2079; valid AUC 0.7510, Brier 0.2066.

## 2. Keras và scorecard trên cùng tập test (30,752 hồ sơ)

| Mô hình | AUC | Gini | KS | Nguồn |
| --- | --- | --- | --- | --- |
| WoEScorecard (fit trên train, seed 42) | 0.7374 | 0.4747 | 36.24% | [S] |
| Keras | 0.7472 | 0.4944 | 37.60% | [B] |
| LightGBM | 0.7509 | 0.5018 | 38.12% | `python -m src.lgbm_baseline`: AUC 0.7509, Gini 0.5018, KS 38.12%, PR-AUC 0.2324, Brier 0.0683 (khớp notebook 02 và README); `evaluate.py` còn hằng số ghi cứng AUC/Gini của LightGBM |

Chênh lệch AUC Keras so với scorecard: 0.0098. Sai số chuẩn của một lần đo AUC ở cỡ mẫu này khoảng 0.006 [P, ước tính trên valid], nên so với sai số đó chênh lệch chỉ cỡ 1.6 lần. Tuy nhiên, so sánh theo sai số chuẩn không ghép cặp là quá thận trọng khi các mô hình được đo trên cùng tập test. Bootstrap ghép cặp 300 lần trên test [P] cho: Keras − scorecard = +0.0098, khoảng tin cậy 95% (0.0050, 0.0143), tức khác 0 có ý nghĩa; LightGBM − Keras = +0.0036, khoảng (0.0000, 0.0074), sát ranh giới; LightGBM − scorecard = +0.0134, khoảng (0.0095, 0.0175). PR-AUC và Brier của scorecard chưa được đo.

## 3. Hiệu chuẩn xác suất của Keras

| Đại lượng | Giá trị | Nguồn |
| --- | --- | --- |
| Tỷ lệ xấu thực tế (test) | 0.0807 | [B] |
| Xác suất dự báo trung bình (test) | 0.4239 | [P] |
| Brier của Keras | 0.2079 | [B] |
| Brier của mô hình hằng số (luôn dự đoán 0.0807) | 0.0742 | [P] |

Cột `avg_prob` so với `bad_rate_%` trong bảng decile cho cùng kết luận [B]: Decile 1 có `avg_prob` 0.7886 nhưng tỷ lệ xấu 25.39%; Decile 4 có 0.5133 và 8.49%.

**Kết luận:** mô hình xếp hạng khách hàng tốt (AUC 0.7472) nhưng xác suất đầu ra không phải tần suất vỡ nợ, nhiều khả năng do huấn luyện bằng class weight. Không dùng trực tiếp làm PD (ví dụ tính EL = PD × LGD × EAD, hoặc đổi sang điểm bằng `pd_to_credit_score`) khi chưa hiệu chuẩn (Platt scaling hoặc isotonic trên tập valid). Đây là hạn chế đã biết của dự án.

## 4. Ngưỡng quyết định

Tham số giả định: biên lợi nhuận 0.15, LGD 0.75 (tỷ lệ 5:1, nên duyệt thêm một khách ở biên khi tỷ lệ tốt/xấu của khách đó lớn hơn 5).

| Chọn T* từ | T* | Tỷ lệ duyệt | Tỷ lệ xấu trong danh mục duyệt | Lợi nhuận ròng (đơn vị tiền gốc đã chuẩn hóa) | Nguồn |
| --- | --- | --- | --- | --- | --- |
| test (cách README) | 0.70 | 87.12% | 5.79% | 2,623.8 (trên test) | [B] |
| test, tham chiếu ngưỡng 0.50 | 0.50 | 63.35% | 3.69% | 2,274.9 (trên test) | [B] |
| valid | 0.71 | 88.16% | 5.83% | 2,379.3 (trên valid) | [B] |
| valid, tham chiếu ngưỡng 0.50 | 0.50 | 63.7% | 3.77% | 2,046.9 (trên valid) | [B] |

Lợi nhuận trên valid và test **không so sánh được với nhau** (hai tập khác cỡ). So sánh đúng là lợi nhuận **trên test** tại hai ngưỡng [P]: 2,623.8 tại 0.70 (chọn từ test) và 2,622.8 tại 0.71 (chọn từ valid), chênh khoảng 1 đơn vị (0.04%). Đường lợi nhuận gần phẳng quanh 0.69–0.72, và T* tối ưu dao động ±0.02 khi lấy mẫu lại (bootstrap 200 lần trên test: trung bình 0.712, khoảng 0.68–0.74) [P]. Vì vậy 0.70 và 0.71 không khác nhau có ý nghĩa.

Mức tăng +348.9 đơn vị (+15.3%) so với ngưỡng 0.50 trong README được tính trên chính tập test dùng để chọn ngưỡng; ở dữ liệu này mức lạc quan do việc đó rất nhỏ, nhưng quy trình đúng vẫn là chọn trên valid rồi đo trên test.

## 5. Tệp cấu hình thật không bị đổi

JSON tái lập `docs/results/threshold_repro_test.json` bằng `config/threshold_config.json` (so sánh dict: trùng khớp). sha256 của `config/threshold_config.json` trước và sau khi chạy đều là `642019A9C7FCF8E0C914DCBA20939D9D14A3E34F4EFDB2ADC22765126E0D8B7F`; `git status config` sạch [B].

## 6. Hạn chế

- LightGBM nay có mã trong `src/lgbm_baseline.py` (có test); số liệu khớp notebook 02 và README.
- So sánh Keras và scorecard chưa có khoảng tin cậy ghép cặp.
- Lợi nhuận tính bằng đơn vị tiền gốc đã chuẩn hóa với biên lợi nhuận và LGD giả định; kết quả chưa được kiểm tra độ nhạy theo hai tham số này.
- Mô hình Keras chưa hiệu chuẩn nên điểm tín dụng suy ra từ nó chỉ có ý nghĩa xếp hạng.
