# Đánh giá Hiệu năng WoEScorecard và Thí nghiệm Rò rỉ Nhãn

> **Ghi chú hiệu đính (2026-11-22, cập nhật 2026-11-24).**
> 1. **Xáo trộn nhãn:** kết quả seed 42 (AUC 0.4385) chỉ là một lần chạy. Chạy thêm 5 seed khác (1 đến 5, cùng hàm, `iv_threshold=0.0`) bằng script phân tích ngoài repo: AUC 0.5217, 0.4615, 0.4973, 0.4932, 0.5236, trung bình 0.4995, độ lệch chuẩn 0.0226. Kết luận đúng: kỳ vọng ≈ 0.5, nên **không thấy kênh rò rỉ ngầm**; không phải "chứng minh pipeline hoàn toàn sạch". AUC dưới 0.5 trong một lần chạy là dao động của mô hình ngẫu nhiên, không phải dấu hiệu lỗi quy trình; rò rỉ sẽ làm AUC **cao hơn** 0.5.
> 2. **Mức phồng +0.0038** nhỏ hơn sai số chuẩn của một lần đo AUC trên valid (≈ 0.0062 với 2,234 khách xấu trên 27,676 dòng). Chưa có khoảng tin cậy ghép cặp cho phép so này nên chưa kết luận chắc là "phồng". Câu "đủ gây lệch PD" chưa có phân tích nào hỗ trợ.
> 3. **Số Keras và LightGBM** ở mục so sánh cuối tài liệu lấy từ `docs/results/keras_repro.md` (`python src/evaluate.py`) và `docs/results/lgbm_baseline.md` (`python -m src.lgbm_baseline`), cùng tập `data/processed/test.csv`.

Ngày thực hiện: 2026-11-22
Môi trường & Seed: Python 3.12, random_state = 42
Lệnh thực thi: python -m src.scorecard_eval

## 1. Kích thước dữ liệu phân chia (Dataset Splits)
Train: 249,083 dòng (khoảng 81%)
Valid: 27,676 dòng (khoảng 9%)
Test: 30,752 dòng (khoảng 10%)

## Các phần thực hiện thí nghiệm.

+ fit trên tập train, transform trên valid/test (clean)
+ fit trên tập train,valid và transform trên test (leak data)
+ Xáo trộn nhãn dữ liệu grouth truth của y

Tiến hành đo các metrics AUC, Gini, KS trên 3 thí nghiệm trên. 

## 2. Kết quả trên tập clean fit

| Tập dữ liệu | Số dòng | AUC | Gini | KS (%) |
| --- | --- | --- | --- | --- |
| Train | 249,083 | 0.7450 | 0.4899 | 36.38% |
| Valid | 27,676 | 0.7395 | 0.4790 | 36.20% |
| Test | 30,752 | 0.7374 | 0.4747 | 36.24% |

## 3. Kết quả trên tập leaked

Mô phỏng lỗi thực hành: gộp train + valid (276,759 dòng) để fit toàn bộ pipeline, sau đó đo lại trên valid:

Valid (Leakage): AUC = 0.7433 | Gini = 0.4865 | KS = 36.52%
Chênh lệch AUC (ΔAUC): +0.0038 (+0.38% AUC)
Nhận xét:
Số đo trên valid bị thổi phồng giả tạo do các ngưỡng cắt bin và trọng số hồi quy đã nhìn thấy trước nhãn của tập đánh giá.
Mức chênh lệch không quá đột biến (+0.0038) vì kích thước tập train đã quá lớn (~250k dòng) và mô hình scorecard tuyến tính có bậc tự do thấp, ít khả năng "học vẹt". Tuy nhiên trong nghiệp vụ ngân hàng, sai số này đủ để gây lệch tỷ lệ PD khi thẩm định rủi ro danh mục lớn.

## 4. Thí nghiệm Xáo trộn nhãn (Label Shuffling / Negative Control)

Xáo trộn ngẫu nhiên nhãn của tập train, fit scorecard và đánh giá trên valid:

Valid (Shuffled): AUC = 0.4385 | Gini = -0.1230 | KS = 8.46%
Nhận xét:
AUC dao động quanh mức  0.5 (kém hơn đoán mò ngẫu nhiên do một số trọng số bị lệch pha ngẫu nhiên với dữ liệu thật, dẫn đến Gini âm).
Kết quả này chứng minh pipeline đánh giá độc lập hoàn toàn trong sạch, không hề có kênh rò rỉ ngầm từ tập valid sang train.


## Nghiệm thu kết quả:

=== NAP DU LIEU ===
So dong train: 249083
So dong valid: 27676
So dong test: 30752

=== BƯỚC 3: FIT SACH ===
[TRAIN] AUC: 0.7450 | Gini: 0.4899 | KS: 36.38%
[VALID] AUC: 0.7395 | Gini: 0.4790 | KS: 36.20%
[TEST] AUC: 0.7374 | Gini: 0.4747 | KS: 36.24%

=== BƯỚC 4: THÍ NGHIỆM RÒ RỈ ===
[VALID LEAK] AUC: 0.7433 | Gini: 0.4865 | KS: 36.52%
Chenh lech AUC (leak - clean): +0.0038

=== BƯỚC 5: XÁO TRỘN NHÃN ===
[VALID SHUFFLED] AUC: 0.4385 | Gini: -0.1230 | KS: 8.46%


So sánh tương quan trên các phương pháp
Scorecard (WoE + Logistic Regression): 
AUC=0.7374 | KS=36.24%
Keras (README): AUC=0.7472| KS=37.60%
LightGBM (README): AUC=0.7509