# Các ý chính của dự án

Tài liệu liệt kê những điều đáng nói nhất về dự án, mỗi ý kèm bằng chứng để kiểm tra. Số liệu lấy từ `docs/results/*.md` và `docs/technical_report.md`. Ký hiệu: **[R]** = có lệnh trong repo để tái lập; **[P]** = phân tích bổ sung bằng script ngoài repo (chưa có trong repo, tham khảo).

## 1. Bài toán và dữ liệu
- Dự báo vỡ nợ trên Home Credit Default Risk (Kaggle): `application_train.csv`, 307,511 hồ sơ, 24,825 nợ xấu (8.07%), mất cân bằng khoảng 11.4 : 1. [R]
- 26 đặc trưng đầu vào (16 số, 10 danh mục), gồm các biến tạo thêm như tuổi, số năm làm việc và bốn biến tỷ lệ (ví dụ `PAYMENT_RATE`). [R: `python -m src.preprocess`]
- Chia phân tầng 81 / 9 / 10 (249,083 / 27,676 / 30,752), seed cố định 23102026; mọi bước có học tham số chỉ `fit` trên train. [R]

## 2. Ba mô hình, một tập test (30,752 hồ sơ)
| Mô hình | AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|
| WoE Scorecard (logistic trên WoE, điểm PDO) | 0.7374 | 0.4747 | 36.24% | chưa đo | chưa đo |
| Keras (MLP, tiền xử lý nằm trong mô hình) | 0.7472 | 0.4944 | 37.60% | 0.2191 | 0.2079 |
| LightGBM (baseline) | 0.7509 | 0.5018 | 38.12% | 0.2324 | 0.0683 |

Lệnh: `python -m src.scorecard_eval`, `python src/evaluate.py`, `python -m src.lgbm_baseline`. [R]

## 3. So sánh mô hình có kiểm định
- Bootstrap ghép cặp 300 lần trên test [P]: Keras − scorecard = +0.0098, khoảng 95% (0.0050, 0.0143), có ý nghĩa; LightGBM − Keras = +0.0036, khoảng (0.0000, 0.0074), sát ranh giới nên chưa kết luận chắc; LightGBM − scorecard = +0.0134, khoảng (0.0095, 0.0175), có ý nghĩa.
- Bài học: so chênh lệch AUC với sai số chuẩn không ghép cặp (khoảng 0.006) quá thận trọng khi các mô hình được đo trên cùng một tập test.

## 4. Chống rò rỉ và kiểm chứng quy trình
- Thí nghiệm rò rỉ nhãn: fit WoE trên train + valid rồi đo trên valid làm AUC tăng từ 0.7395 lên 0.7433 (+0.0038). [R]
- Xáo trộn nhãn (đối chứng âm): seed 42 cho AUC 0.4385 [R]; 5 seed khác cho trung bình 0.4995, độ lệch chuẩn 0.0226 [P]. Kết luận có điều kiện: không thấy kênh rò rỉ ngầm; một lần chạy không chứng minh được điều gì.
- `python -m src.preprocess` sinh lại ba tệp CSV trùng sha256 với dữ liệu dùng để đo. [P]

## 5. Hiệu chuẩn xác suất
- LightGBM tối ưu log-loss không trọng số nên xác suất bám tần suất thật: Brier 0.0683 thấp hơn mô hình hằng số 0.0742 [R]; trung bình dự báo 0.0810 so với tỷ lệ xấu 0.0807 [P].
- Keras dùng `class_weight` nên xác suất bị đẩy lên: Brier 0.2079 tệ hơn cả mô hình hằng số [R]; trung bình dự báo 0.4239 [P]. Đầu ra của nó là điểm xếp hạng, chưa phải PD.

## 6. Ngưỡng quyết định theo chi phí
- Ma trận chi phí: lãi 0.15 trên khách tốt được duyệt, mất 0.75 trên khách xấu được duyệt (tỷ lệ 5:1). T\* = 0.70 (chọn trên test): duyệt 87.12%, nợ xấu 5.79%, lợi nhuận 2,623.8 so với 2,274.9 ở ngưỡng 0.50 (+15.3%, tính trên chính tập chọn ngưỡng). [R: `python -m src.repro_keras`]
- Chọn trên valid cho T\* = 0.71; lợi nhuận trên test chênh khoảng 1 đơn vị (0.04%) và T\* dao động ±0.02 khi lấy mẫu lại [P]. Nguyên tắc vẫn là chọn trên valid, đo trên test.

## 7. API chấm điểm
- FastAPI nhận hồ sơ, trả PD, điểm 300 đến 850, nhóm rủi ro và quyết định; 11 test cho API (hợp đồng, 422, giá trị phi lý, mô hình giả, nhất quán công thức, vai trò của ngưỡng); toàn dự án 57 test. [R]
- Ví dụ thật: PD 0.1703, điểm 684, `LOW_RISK`, `APPROVED` (khi gửi đủ bốn biến dẫn xuất).

## 8. Phát hiện nhờ kiểm thử và kiểm chứng
1. `WoEScorecard` không truyền danh sách cột cho `WoEEncoder`, nên ghép chặt với 26 cột trong `config.py`.
2. `score(X)` của scorecard ghi đè `ClassifierMixin.score(X, y)` của scikit-learn: `cross_val_score` ném `TypeError`.
3. Ngưỡng tối ưu **không** được API dùng để quyết định (chỉ trả về); các mốc điểm cứng 750/670/580 duyệt tự động khoảng 15.6% hồ sơ test so với 87.1% của T\* = 0.70 [P].
4. Bốn biến dẫn xuất do client gửi; bỏ qua thì điểm của ví dụ tụt từ 684 xuống 630.
5. `evaluate.py` ghi cứng hằng số LightGBM; `optimize_threshold` mặc định đọc `test.csv` và ghi đè `config/threshold_config.json`.

## 9. Hạn chế đã biết
Keras chưa hiệu chuẩn; API chưa dùng ngưỡng tối ưu, chưa có ràng buộc miền giá trị (`AGE = -5` vẫn trả 200) và phải chạy từ thư mục gốc; chưa kiểm tra ổn định theo thời gian (dữ liệu không có chiều thời gian); scorecard chưa có PR-AUC và Brier; các số [P] chưa có script trong repo.

## 10. Kiến thức nên nắm để trả lời phỏng vấn
WoE và IV (công thức `ln(dist_good/dist_bad)` trên tổng khách tốt/xấu; thang IV là quy tắc kinh nghiệm), điểm PDO (factor = PDO/ln 2, offset), AUC/Gini/KS, PR-AUC so với tỷ lệ xấu, Brier và hiệu chuẩn, rò rỉ nhãn so với rò rỉ phân phối, chọn ngưỡng trên tập nào, bootstrap ghép cặp, đánh đổi scorecard (giải thích được) với LightGBM (hiệu chuẩn tốt, mạnh hơn) và Keras (cần hiệu chuẩn).

## 11. Hướng phát triển
Hiệu chuẩn Keras (Platt hoặc isotonic) hoặc chuyển PD sang LightGBM; API tự tính biến dẫn xuất, thêm ràng buộc miền và dùng ngưỡng tối ưu; sửa `scorecard.py`; viết script tái lập các phân tích [P]; kiểm tra ổn định theo thời gian nếu có dữ liệu.
