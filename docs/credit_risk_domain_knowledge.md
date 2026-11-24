# 🏦 CẨM NANG NGHIỆP VỤ RỦI RO TÍN DỤNG (CREDIT RISK DOMAIN KNOWLEDGE)
### *Dành cho Kỹ sư ML/AI & Data Scientist phỏng vấn ngành Ngân hàng & Fintech*

---

## 📌 MỤC LỤC
1. [Tổng quan về Nghiệp vụ Tín dụng & Chu trình Phê duyệt](#1-tổng-quan-về-nghiệp-vụ-tín-dụng--chu-trình-phê-duyệt)
2. [Bộ ba tham số chuẩn Basel II / Basel III (PD, LGD, EAD)](#2-bộ-ba-tham-số-chuẩn-basel-ii--basel-iii-pd-lgd-ead)
3. [Giải mã Bộ chỉ số Đo lường Mô hình Ngân hàng (Validation Metrics)](#3-giải-mã-bộ-chỉ-số-đo-lường-mô-hình-ngân-hàng-validation-metrics)
4. [Tối ưu hóa Ngưỡng cắt & Ma trận Chi phí - Lợi nhuận (Cost-Benefit Matrix)](#4-tối-ưu-hóa-ngưỡng-cắt--ma-trận-chi-phí---lợi-nhuận-cost-benefit-matrix)
5. [Hệ thống Điểm tín dụng (Credit Score & FICO Scale)](#5-hệ-thống-điểm-tín-dụng-credit-score--fico-scale)
6. [Bộ câu hỏi phỏng vấn thực chiến (Interview Cheatsheet)](#6-bộ-câu-hỏi-phỏng-vấn-thực-chiến-interview-cheatsheet)

---

## 1. Tổng quan về Nghiệp vụ Tín dụng & Chu trình Phê duyệt

### 1.1. Bản chất kinh doanh của Ngân hàng
Ngân hàng kinh doanh dựa trên **chênh lệch lãi suất (Net Interest Margin - NIM)**:
* Huy động vốn từ người gửi tiền với lãi suất thấp (ví dụ: $5\%/\text{năm}$).
* Cho các cá nhân/doanh nghiệp vay lại với lãi suất cao hơn (ví dụ: $15\% - 20\%/\text{năm}$).

> **Nghịch lý tín dụng:** Nếu cho vay quá dễ dãi $\to$ Doanh số cao nhưng **nợ xấu bùng nổ**, mất vốn gốc. Nếu siết quá chặt $\to$ Không có nợ xấu nhưng **mất hết khách hàng** vào tay đối thủ.  
> 👉 **Mô hình Machine Learning (Credit Scoring Model)** sinh ra để tìm ra điểm cân bằng hoàn hảo giữa Doanh số và Rủi ro.

### 1.2. Chu trình xét duyệt tín dụng (Credit Underwriting Lifecycle)
1. **Thu thập hồ sơ:** Khách hàng nộp thông tin (Thu nhập, độ tuổi, công việc, số tiền muốn vay, v.v.).
2. **Lọc hồ sơ thô (Knock-out rules):** Kiểm tra các điều kiện tiên quyết (ví dụ: tuổi $< 18$, đang có nợ nhóm 3-5 trên CIC/bureau $\to$ Từ chối tự động).
3. **Chấm điểm tín dụng (Credit Scoring):** Đưa hồ sơ qua mô hình Machine Learning/Deep Learning để tính ra **Xác suất vỡ nợ (Probability of Default - PD)**.
4. **Ra quyết định (Decision Engine):** So sánh PD với **Ngưỡng cắt (Cut-off Threshold)**:
   * **Auto-Approve (Duyệt ngay):** Rủi ro cực thấp, giải ngân tự động sau vài phút.
   * **Manual Review (Chuyển thẩm định viên):** Rủi ro trung bình, cần con người gọi điện xác minh.
   * **Auto-Reject (Từ chối ngay):** Rủi ro cao vượt ngưỡng an toàn.

---

## 2. Bộ ba tham số chuẩn Basel II / Basel III (PD, LGD, EAD)

Hiệp ước **Basel II & Basel III** là bộ quy chuẩn quản trị rủi ro ngân hàng toàn cầu mà Ngân hàng Nhà nước Việt Nam (Thông tư 41/2016/TT-NHNN) bắt buộc các ngân hàng phải áp dụng.

Công thức tính **Tổn thất dự kiến (Expected Loss - EL)** của một khoản vay:
$$\mathbf{EL = PD \times LGD \times EAD}$$

| Thuật ngữ | Tên tiếng Anh | Định nghĩa nghiệp vụ | Ý nghĩa trong dự án của chúng ta |
| :--- | :--- | :--- | :--- |
| **PD** | **Probability of Default** | Xác suất một khách hàng **mất khả năng trả nợ (quá hạn $\ge 90$ ngày)** trong vòng 12 tháng tới. | **Đây chính là output $y_{pred} \in [0, 1]$ của mô hình Deep Learning!** |
| **LGD** | **Loss Given Default** | Tỷ lệ tổn thất thực tế nếu khách hàng vỡ nợ (sau khi đã thanh lý tài sản đảm bảo hoặc bán nợ). | Với vay tín chấp không tài sản đảm bảo, LGD thường rất cao: **$60\% - 80\%$**. |
| **EAD** | **Exposure at Default** | Tổng số dư nợ gốc và lãi tại thời điểm khách hàng tuyên bố vỡ nợ. | Bằng giá trị khoản vay được phê duyệt ban đầu (`AMT_CREDIT`). |

> **Tại sao ngân hàng quan tâm đến EL?**  
> Theo luật, ngân hàng phải **trích lập quỹ dự phòng rủi ro** tương ứng với $EL$. Quỹ này bị "đóng băng" trong két, không được đem đi đầu tư. Mô hình ML dự báo PD càng chuẩn xác thì ngân hàng càng tối ưu được lượng vốn dự phòng này.

---

## 3. Giải mã Bộ chỉ số Đo lường Mô hình Ngân hàng (Validation Metrics)

Khi nộp báo cáo mô hình lên Hội đồng Quản trị Rủi ro (Risk Committee), họ **không bao giờ nhìn vào Accuracy**. Họ yêu cầu 4 chỉ số bắt buộc sau:

### 3.1. ROC-AUC (Area Under the ROC Curve)
* **Khái niệm:** Đo lường xác suất mà mô hình xếp hạng một khách hàng xấu ngẫu nhiên có điểm rủi ro cao hơn một khách hàng tốt ngẫu nhiên.
* **Chuẩn ngân hàng:**
  * $\text{AUC} < 0.65$: Mô hình yếu, không được phê duyệt triển khai.
  * $0.70 \le \text{AUC} < 0.75$: **Mô hình đạt chuẩn sản xuất (Mô hình của chúng ta đạt 0.7472)**.
  * $\text{AUC} \ge 0.80$: Rất hiếm với dữ liệu tín chấp cá nhân, cần kiểm tra rò rỉ dữ liệu (Data Leakage).

### 3.2. Hệ số GINI ($Gini = 2 \times AUC - 1$)
* **Nguồn gốc:** Mượn từ hệ số Gini trong kinh tế học (đo mức độ bất bình đẳng thu nhập).
* **Ứng dụng tín dụng:** Đo mức độ phân hóa giữa nhóm trả nợ tốt và nhóm nợ xấu.
* **Quy đổi:**
  $$\text{AUC} = 0.50 \iff \text{Gini} = 0.00 \text{ (Đoán ngẫu nhiên)}$$
  $$\text{AUC} = 0.7472 \iff \text{Gini} = \mathbf{0.4944} \text{ (Đạt chuẩn Basel rất tốt, chuẩn là } \ge 0.40\text{)}$$

### 3.3. Chỉ số KS (Kolmogorov-Smirnov Statistic)
* **Khái niệm:** Khoảng cách lớn nhất giữa **Đường cong tích lũy khách hàng Tốt** và **Đường cong tích lũy khách hàng Xấu**.
  $$\text{KS} = \max |F_{\text{bad}}(t) - F_{\text{good}}(t)|$$
* **Tại sao ngân hàng rất thích KS?**
  Chỉ số KS cho biết **khả năng tách bạch 2 nhóm khách hàng** mạnh đến mức nào. Điểm cực đại của KS thường chính là gợi ý tuyệt vời cho vị trí đặt ngưỡng cắt (Cut-off).
* **Chuẩn ngân hàng:**
  * $\text{KS} < 20\%$: Rất kém.
  * $20\% - 30\%$: Chấp nhận được.
  * $30\% - 40\%$: **Tốt (Mô hình của chúng ta đạt 37.60%)**.
  * $> 40\%$: Xuất sắc.

### 3.4. Bảng Decile (Decile Analysis) & Tính Đơn Điệu (Monotonicity)
Chia toàn bộ danh mục khách hàng sau khi dự báo thành **10 phần bằng nhau (10 Deciles)** theo thứ tự rủi ro giảm dần:

```text
Decile 1 (Top 10% rủi ro cao nhất)  ──> Tỷ lệ nợ xấu: 25.39%  ▲
Decile 2                            ──> Tỷ lệ nợ xấu: 15.12%  │ (Giảm dần đều)
Decile 3                            ──> Tỷ lệ nợ xấu: 10.96%  │
...                                                            │
Decile 10 (Top 10% an toàn nhất)    ──> Tỷ lệ nợ xấu:  1.43%  ▼
```

> **Quy tắc vàng (Monotonicity Rule):**  
> Bắt buộc tỷ lệ nợ xấu thực tế (`bad_rate_%`) phải **giảm dần đều từ Decile 1 đến Decile 10**. Nếu Decile 4 tự nhiên có tỷ lệ nợ xấu vọt lên cao hơn Decile 3, mô hình sẽ bị **bác bỏ ngay lập tức** vì không có tính ổn định và tính xếp hạng rủi ro bị méo mó.

---

## 4. Tối ưu hóa Ngưỡng cắt & Ma trận Chi phí - Lợi nhuận (Cost-Benefit Matrix)

### 4.1. Tại sao ngưỡng mặc định 0.5 là một sai lầm?
Hầu hết các thư viện Machine Learning mặc định gán nhãn: $PD \ge 0.5 \to$ Dự đoán nợ xấu.  
Tuy nhiên, trong dữ liệu thực tế:
* Tỷ lệ nợ xấu chỉ chiếm $\approx 8\%$.
* Hậu quả tài chính của 2 loại sai lầm là **hoàn toàn bất đối xứng**:
  * **False Positive (Type I Error):** Đoán nhầm khách hàng tốt thành nợ xấu $\to$ Ngân hàng chỉ mất **tiền lãi** (khoảng $+15\%$).
  * **False Negative (Type II Error):** Đoán nhầm khách nợ xấu thành khách tốt $\to$ Ngân hàng duyệt cho vay và mất gần như **toàn bộ vốn gốc** ($-75\%$ đến $-100\%$).

### 4.2. Ma trận Chi phí - Lợi nhuận (Cost-Benefit Matrix)

| Thực tế \ Mô hình | Mô hình bảo DUYỆT ($PD < T$) | Mô hình bảo TỪ CHỐI ($PD \ge T$) |
| :--- | :--- | :--- |
| **Khách TỐT ($Y=0$)** | **Thu lãi:** $+15\%$ giá trị khoản vay | **Mất cơ hội:** $0$ lợi nhuận |
| **Khách XẤU ($Y=1$)** | **Thất thoát vốn:** $-75\%$ vốn gốc | **Chặn tổn thất:** Tránh được rủi ro |

Công thức Lợi nhuận ròng (Net Profit):
$$\mathbf{\text{Net Profit}} = (N_{\text{Approved Goods}} \times 15\%) - (N_{\text{Approved Bads}} \times 75\%)$$

### 4.3. Kết quả thực tế từ file `src/threshold_optimization.py`:
* **Tại ngưỡng 0.50:** Ngân hàng quá thận trọng, chỉ duyệt $63.35\%$ hồ sơ, bỏ lỡ rất nhiều khách hàng tốt $\to$ Lợi nhuận: **2,274 đơn vị**.
* **Tại ngưỡng tối ưu 0.70:** Phê duyệt $87.12\%$ hồ sơ, kiểm soát nợ xấu ở mức an toàn $5.79\%$ $\to$ Lợi nhuận: **2,623 đơn vị**.
* 👉 **Việc tối ưu ngưỡng mang lại thêm $+15.3\%$ lợi nhuận ròng** mà không cần phải đổi mô hình!

---

## 5. Hệ thống Điểm tín dụng (Credit Score & FICO Scale)

Khách hàng và nhân viên ngân hàng không hiểu con số xác suất $PD = 0.082$. Họ cần một con số trực quan: **Điểm tín dụng (Credit Score)** theo thang chuẩn **FICO (300 - 850 điểm)**.

### 5.1. Công thức Log-Odds (Points to Double Odds - PDO)
Mối quan hệ giữa xác suất vỡ nợ $PD$ và Điểm số là quan hệ nghịch biến phi tuyến tính (Log-Odds):
$$\text{Odds} = \frac{1 - PD}{PD} = \frac{P(\text{Trả nợ tốt})}{P(\text{Nợ xấu})}$$
$$\text{Credit Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

### 5.2. Bảng phân tầng rủi ro (Risk Tiers) trong hệ thống:

| Phân khúc (Tier) | Thang điểm FICO | Hành động tự động (System Action) | Mức độ rủi ro |
| :--- | :--- | :--- | :--- |
| **Super Prime (Xuất sắc)** | **$750 - 850$** | **AUTO-APPROVE** (Duyệt vay tức thì, ưu đãi lãi suất) | Cực kỳ an toàn ($PD < 3\%$) |
| **Prime (Chuẩn)** | **$670 - 749$** | **APPROVE** (Duyệt vay tiêu chuẩn) | An toàn ($PD \approx 3\% - 8\%$) |
| **Near Prime (Cận chuẩn)** | **$580 - 669$** | **MANUAL REVIEW** (Yêu cầu bổ sung sao kê lương/thẩm định) | Trung bình ($PD \approx 8\% - 15\%$) |
| **Subprime (Dưới chuẩn)** | **$300 - 579$** | **AUTO-REJECT** (Từ chối tự động để bảo toàn vốn) | Rủi ro cao ($PD > 15\%$) |

---

## 6. Bộ câu hỏi phỏng vấn thực chiến (Interview Cheatsheet)

Dưới đây là 5 câu hỏi phỏng vấn kinh điển dành cho vị trí **ML/AI Engineer trong ngành Ngân hàng / Fintech**:

#### ❓ Câu 1: "Tại sao em chọn metric ROC-AUC thay vì Accuracy hay F1-Score?"
> **Trả lời:** "Dữ liệu tín dụng có tính chất mất cân bằng nghiêm trọng (imbalanced), chỉ khoảng 8% khách hàng nợ xấu. Nếu dùng Accuracy, mô hình chỉ cần dự báo tất cả khách hàng đều tốt là đã đạt 92% nhưng vô dụng trong kinh doanh.  
> Em không chọn F1-Score làm metric chính vì F1 phụ thuộc vào một ngưỡng cố định (threshold-dependent). Trong quản trị rủi ro tín dụng, ROC-AUC đo lường khả năng **xếp hạng rủi ro (rank-ordering capability)** tổng quát trên toàn bộ các ngưỡng cắt khả dĩ, phù hợp với chuẩn mực Basel II."

#### ❓ Câu 2: "Hệ số Gini và chỉ số KS trong mô hình của em có ý nghĩa gì?"
> **Trả lời:** "Hệ số Gini của mô hình đạt **0.4944** ($Gini = 2 \times AUC - 1$), vượt xa mức yêu cầu tối thiểu của ngân hàng là 0.40.  
> Chỉ số KS đạt **37.60%**, nghĩa là ở điểm phân tách tối ưu, khoảng cách giữa tỷ lệ tích lũy khách hàng tốt và khách hàng xấu là 37.6%. Con số này nằm trong khoảng 30% - 40%, chứng minh mô hình có khả năng phân loại khách hàng rủi ro cực kỳ sắc bén."

#### ❓ Câu 3: "Em xử lý bài toán Imbalanced Data trong Deep Learning như thế nào?"
> **Trả lời:** "Em đã áp dụng cơ chế `class_weight` trong hàm mất mát `BinaryCrossentropy`. Dựa trên công thức heuristic cân bằng của scikit-learn ($w_j = \frac{N}{2 \times N_j}$), nhãn nợ xấu (Class 1) được gán trọng số phạt gấp **11.38 lần** so với nhãn trả nợ tốt (Class 0). Điều này ép mạng nơ-ron phải tối ưu kỹ các đặc trưng phi tuyến của nhóm nợ xấu thay vì thiên vị nhóm đa số."

#### ❓ Câu 4: "Sau khi train xong, làm sao em xác định được ngưỡng quyết định cho vay hay từ chối?"
> **Trả lời:** "Em không dùng ngưỡng mặc định 0.5. Em xây dựng **Ma trận Chi phí - Lợi nhuận (Cost-Benefit Matrix)** mô phỏng môi trường kinh doanh thực tế: mỗi khoản vay tốt mang lại 15% biên lãi, còn mỗi khoản nợ xấu lọt vào gây mất 75% vốn gốc (LGD). Bằng cách quét qua dải ngưỡng từ 0.05 đến 0.95, em tìm ra ngưỡng tối ưu là **0.70**, giúp tăng thêm 15.3% lợi nhuận ròng và phê duyệt được 87.1% khách hàng với tỷ lệ nợ xấu danh mục chỉ 5.79%."

#### ❓ Câu 5: "Làm thế nào để đưa mô hình này vào phục vụ thực tế (Production Serving) mà không bị lỗi Train/Serving Skew?"
> **Trả lời:** "Em đóng gói toàn bộ các bước tiền xử lý (Normalization biến số và StringLookup One-Hot biến danh mục) **trực tiếp vào trong kiến trúc Keras Functional API** dưới dạng Keras Preprocessing Layers.  
> Nhờ đó, file `best_model.keras` là một mô hình End-to-End hoàn chỉnh: nhận thẳng dictionary thô (chứa cả số thực và chuỗi string) và xuất ra xác suất vỡ nợ PD. Khi triển khai API bằng FastAPI, hệ thống không cần duy trì bất kỳ pipeline tiền xử lý rời nào bên ngoài, loại bỏ hoàn toàn rủi ro Train/Serving Skew và đạt độ trễ suy luận dưới 15ms."

---

*Tài liệu này được biên soạn độc quyền cho dự án Home Credit Default Risk Prediction.*
