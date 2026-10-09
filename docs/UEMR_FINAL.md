# UEMR — Unified Event-Aware Multi-Vector Retrieval

**Biểu diễn đa vector âm thanh–hình ảnh theo sự kiện cho truy hồi video dài, kết hợp OmniRetriever và UniAV**

> **Bản chốt cuối cùng.** File này thay thế `omni_uniav_multivector_thesis_idea.md`, `final.md` và `claude/uemr_method.md`.
>
> Nguyên tắc xuyên suốt: **Omni trả lời "cái gì"** (không gian ngữ nghĩa audio-video-text), **UniAV trả lời "khi nào"** (sự kiện nằm ở đâu và ngữ cảnh xung quanh). **Cách chia video là biến độc lập chính của nghiên cứu.**

---

## Mục lục

1. Bài toán và câu chuyện nghiên cứu
2. Research questions
3. Kiến trúc tổng quan
4. Các thành phần (A → D)
5. Các cách biểu diễn video cần so sánh
6. Huấn luyện
7. Dữ liệu, protocol và metrics
8. Thí nghiệm
9. Đóng góp và những gì không claim
10. Phạm vi: khóa luận và tạp chí
11. Thứ tự triển khai
12. Siêu tham số khởi điểm
13. Pseudocode
14. Checklist trước khi chạy

---

## 1. Bài toán và câu chuyện nghiên cứu

Cho text query $q$ và gallery gồm $N$ video dài $\{V_1,\dots,V_N\}$, mỗi video chứa nhiều sự kiện. Nhiệm vụ là xếp hạng video sao cho video chứa sự kiện được mô tả đứng đầu. Ngoài ra, hệ thống còn trả về khoảng thời gian của sự kiện đó.

**Câu chuyện nghiên cứu:**

```text
Video dài chứa nhiều sự kiện, nhưng query chỉ mô tả một sự kiện
        ↓
Một vector global làm loãng thông tin cục bộ
        ↓
Multi-vector là hướng hợp lý, nhưng MeVTR đã làm (K-Medoids trên frame CLIP)
        ↓
Câu hỏi chưa được trả lời: lợi ích đến từ VIỆC CÓ NHIỀU VECTOR,
hay từ CẤU TRÚC SỰ KIỆN THEO THỜI GIAN?
        ↓
Cố định backbone (Omni + UniAV), chỉ thay đổi cách chia video
        ↓
Thêm ngữ cảnh thời gian (Context Fusion) để phân biệt các sự kiện trông giống nhau
        ↓
Đo có kiểm soát: giá trị của ranh giới sự kiện ngữ nghĩa và của ngữ cảnh
```

**Khác biệt với MeVTR:**

| | MeVTR (Me-Retriever) | UEMR |
|---|---|---|
| Đơn vị | Frame đại diện (medoid) | Đoạn sự kiện có $[t_s,t_e]$ |
| Cách chọn | K-Medoids theo nội dung | Detector sự kiện (UniAV) |
| Modality | Visual (CLIP) | Audio + Visual (Omni, UniAV) |
| Ngữ cảnh | Không có | Context Fusion (cục bộ + giữa các event) |
| Timestamp | Không có | Có, miễn phí từ argmax |

---

## 2. Research questions

- **RQ1 — Multi vs Single.** Với cùng backbone, biểu diễn đa vector có tốt hơn một vector không?
- **RQ2 — Ranh giới ngữ nghĩa.** Chia theo sự kiện (UniAV / GT) có tốt hơn chia đều, chia ngẫu nhiên và K-Medoids khi **cùng số vector** không?
- **RQ3 — Ngữ cảnh.** Bổ sung ngữ cảnh audio-visual theo thời gian từ UniAV có giúp phân biệt các sự kiện giống nhau không? Thứ hạng giữa các cách chia có thay đổi khi có ngữ cảnh không?
- **RQ4 — Chất lượng detector.** Ranh giới phải chính xác đến mức nào thì mới có lợi? UniAV còn thua oracle GT bao nhiêu?
- **RQ5 (tạp chí).** Một index sự kiện duy nhất có giải được đồng thời video retrieval và moment localization không?

---

## 3. Kiến trúc tổng quan

```text
                     Untrimmed video + audio
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
  (A) UniAV ❄ (fine-tuned)                 Partition strategy
      ├─ Z = {z_1..z_T}   ngữ cảnh AV      (Global / Uniform / Random /
      └─ proposals (t_s,t_e,conf) ───────►  K-Medoids / UniAV-Pred / GT)
          │                                         │
          │                                  các đoạn S_1..S_K
          │                                         ▼
          │                           (B) Omni video+audio ❄ (fine-tuned)
          │                                         │
          │                                e_1 .. e_K  (1 vector/đoạn)
          │                                         │
          └──── Z quanh mỗi đoạn ────────►  (C) Context Fusion 🔥
                                                    │
                                          c_1 .. c_K
                                                    │
                                   C(V) = {c_1..c_K}  (+ c_g tuỳ chọn)
                                                    │
  Text ──► Omni text ❄ ──► q ────────────────► (D) Late-interaction scoring
                                                    │
                                   S(q,V)  +  [t_s,t_e] của đoạn khớp nhất
```

❄ = frozen (đã fine-tune trước đó)  🔥 = trainable, chỉ vài triệu tham số

**Không dùng ONE-PEACE text encoder.** Text và video đều đi qua Omni, nên đã nằm sẵn trong cùng một không gian.

---

## 4. Các thành phần

### (A) UniAV — tìm sự kiện và cung cấp ngữ cảnh

UniAV (fine-tuned) chạy **offline một lần** cho mỗi video và sinh ra:

1. **Chuỗi feature $Z=\{z_1,\dots,z_T\}$** ở **một pyramid level cố định** cho mọi thí nghiệm (chọn level có stride khoảng 1–2 giây). Lưu kèm hàm mapping giây → index lấy từ config của UniAV. **Không được giả định** $z_t$ ứng với giây thứ $t$.
2. **Event proposals** $P=\{(t^s_i,t^e_i,\text{conf}_i)\}$ từ localization head sau Soft-NMS.

Hậu xử lý proposals, áp dụng giống nhau cho mọi video:

- giữ các proposal có $\text{conf}\ge\theta$, tối đa $K_{\max}$ proposal;
- nếu sau khi lọc còn ít hơn 1 proposal, giữ proposal có confidence cao nhất;
- gộp các proposal có tIoU > 0.7;
- kéo dài các đoạn ngắn hơn $d_{\min}$ để Omni có đủ frame;
- sắp xếp theo thời gian.

$K_{pred}$ là số proposal còn lại. Đây là số vector dùng để kiểm soát các baseline (xem mục 5).

### (B) Omni — mã hoá đoạn và query vào cùng không gian

Với mỗi đoạn $S_i=[t^s_i,t^e_i]$ (sinh ra từ bất kỳ cách chia nào), cắt clip video $V_i$ và audio $A_i$ tương ứng rồi mã hoá **jointly**:

$$
e_i=f_{Omni}(V_i,A_i)
$$

Text query:

$$
q=f^{T}_{Omni}(Q)
$$

Vector global (dùng cho Global baseline và vector $c_g$ tuỳ chọn):

$$
e_g=f_{Omni}(V,A)
$$

Mọi embedding đều **được cache offline**.

### (C) Context Fusion — bổ sung ngữ cảnh cho từng đoạn

**Vấn đề.** $e_i$ chỉ nhìn thấy nội dung *bên trong* đoạn. Hai lần "khuấy" ở bước 3 (sau khi cho hành) và bước 7 (sau khi đổ sốt cà chua) cho ra hai vector gần như giống nhau. Tuy vậy, query *"stir the tomato sauce"* chỉ đúng với bước 7. Thông tin để phân biệt nằm **ngoài** clip. Ngoài ra, nếu detector cắt lệch ranh giới thì đoạn sẽ mất phần đầu hoặc phần cuối của hành động.

**Giải pháp.** Module này sửa $e_i$ bằng ngữ cảnh, qua 3 bước.

**Bước 1: ngữ cảnh cục bộ từ UniAV (cross-attention).** Lấy các feature của UniAV trong cửa sổ mở rộng $[t^s_i-\delta,\ t^e_i+\delta]$, cộng với positional encoding tương đối so với đoạn (bên trong, trước hay sau đoạn). Khi đó $e_i$ đóng vai **query**, còn $z_t$ đóng vai **key/value**:

$$
r_i=\text{MHA}\big(W_Q e_i,\ W_K(Z_i^{\pm\delta}+p),\ W_V(Z_i^{\pm\delta}+p)\big)
$$

Hai ma trận $W_K$ và $W_V$ chiếu $Z$ (không gian của UniAV) sang không gian của Omni, nên hai backbone không cần chung không gian.

**Bước 2: ngữ cảnh giữa các đoạn (self-attention).** Xếp $K$ đoạn của cùng video theo thứ tự thời gian rồi cho qua một lớp Transformer encoder:

$$
\{u_1,\dots,u_K\}=\text{SelfAttn}\big(\{e_i+r_i+p_i\}_{i=1}^K\big)
$$

Nhờ đó mỗi đoạn biết vị trí của nó trong quy trình. Ý tưởng này giống ColBERT: mỗi token được encode trong ngữ cảnh của cả passage trước khi tính MaxSim. Ở đây token là đoạn, passage là video.

**Bước 3: residual có gate.**

$$
c_i=\operatorname{Normalize}\big(e_i+\alpha\,W_u u_i\big),\qquad \alpha\ \text{learnable, khởi tạo}=0
$$

- Khi $\alpha=0$ lúc bắt đầu, mô hình đúng bằng Omni thuần. Nếu ngữ cảnh có ích, quá trình train sẽ tự tăng $\alpha$.
- Công thức này cộng thêm vào $e_i$ chứ không thay thế nó, **không dùng LayerNorm ở đầu ra**, chỉ L2-normalize. Như vậy $c_i$ vẫn nằm trong không gian của Omni text.

**Trường hợp đặc biệt:**

- **Global** ($K=1$): bước 1 dùng toàn bộ $Z$ (subsample nếu quá dài), bước 2 không có tác dụng.
- **Vector global $c_g$** (nếu dùng): $c_g=e_g$, không đi qua fusion.

### (D) Scoring — late interaction trên tập đoạn

Điểm của từng đoạn:

$$
s_i=\cos(q,c_i)
$$

**Inference** dùng MaxSim:

$$
S(q,V)=\max_i s_i
$$

**Training** dùng LogSumExp có chuẩn hoá (để gradient đi qua nhiều đoạn):

$$
S_\tau(q,V)=\tau_s\log\frac{1}{|C(V)|}\sum_i e^{s_i/\tau_s}
$$

**Timestamp miễn phí:** $i^*=\arg\max_i s_i$ cho ngay khoảng $[t^s_{i^*},t^e_{i^*}]$.

Cách gọi đúng là **late interaction một chiều** (query một vector, video nhiều vector), hay **MaxSim / MIL matching**. Không gọi là ColBERT thật.

---

## 5. Các cách biểu diễn video cần so sánh

Tất cả dùng **cùng Omni, cùng UniAV, cùng fusion, cùng loss, cùng scoring, cùng gallery**. Chỉ khác cách chia video.

| # | Cách biểu diễn | Cách tạo đoạn | Số vector | Kiểm tra điều gì |
|---|---|---|---|---|
| R0 | **Global** | Toàn video | 1 | Baseline single-vector |
| R1 | **Event-Single** | Mean các vector của R6 | 1 | Cùng thông tin với R6 nhưng bị nén |
| R2 | **Uniform-Chunk** | Đoạn dài cố định $L$ giây | $\lceil T/L\rceil$ (thay đổi) | Baseline tự nhiên, không kiểm soát số vector |
| R3 | **Uniform-M** | $M$ đoạn bằng nhau | $M=K_{pred}$ | Multi-vector không có ranh giới ngữ nghĩa |
| R4 | **Random-M** | $M$ đoạn với ranh giới ngẫu nhiên (độ dài tối thiểu $d_{\min}$, cố định seed) | $M=K_{pred}$ | Chia bừa có tốt ngang không? |
| R5 | **K-Medoids-M** (kiểu MeVTR) | K-Medoids trên $Z$ (cosine), mỗi medoid tại $t_m$ → clip $[t_m-1,t_m+1]$ qua Omni | $M=K_{pred}$ | Đa dạng *nội dung* thay vì cấu trúc *thời gian* |
| R6 | **UniAV-Pred** (phương pháp chính) | Proposals của UniAV | $K_{pred}$ | Ranh giới sự kiện tự động |
| R7 | **GT Event** (oracle) | Ranh giới GT của YouCook2 | $K_{GT}$ | Cận trên của ranh giới ngữ nghĩa |

Ghi chú:

- **Kiểm soát số vector:** MaxSim có thiên lệch theo số vector (càng nhiều vector càng dễ có một vector trùng hợp khớp cao). Vì vậy R3, R4, R5 dùng đúng $M=K_{pred}$ của từng video. Khi so với GT, chạy thêm **Uniform-M với $M=K_{GT}$** làm đối chứng về số lượng.
- **K-Medoids trên $Z$ thay vì trên frame Omni:** cách này rẻ (không phải chạy Omni-7B trên mọi cửa sổ), và chỉ cần $M$ clip Omni cho mỗi video.
- **Uniform-Chunk:** mặc định $L$ = trung vị độ dài event GT trên tập train. Thêm $L=8$ giây làm ablation.

---

## 6. Huấn luyện

**Chỉ train Context Fusion.** Omni và UniAV đều frozen. Mọi input ($e_i$, $e_g$, $q$, $Z$, proposals) đã cache, nên training nhẹ, chạy trên 1 GPU và không cần load Omni-7B.

**Nguyên tắc quan trọng:** video-side luôn dùng đoạn của **chính cách chia đang đánh giá** (với phương pháp chính là proposals của UniAV). GT chỉ dùng để (i) gán nhãn positive cho $\mathcal{L}_{evt}$ và (ii) làm oracle R7. Làm như vậy để train và test cùng một phân phối đầu vào.

### Loss 1 — Video-level (MIL)

Mỗi batch lấy **tối đa 1 caption cho mỗi video gốc** để tránh false negative. Loss là InfoNCE text→video trên $S_\tau$:

$$
\mathcal{L}_{vid}=-\frac1B\sum_i\log\frac{\exp(S_\tau(q_i,V_i)/\tau)}{\sum_j\exp(S_\tau(q_i,V_j)/\tau)}
$$

### Loss 2 — Event-level với hard negative trong cùng video

Caption $q$ có GT segment $g$. Gọi $i^+=\arg\max_i \text{tIoU}(S_i,g)$, và chỉ dùng mẫu này nếu tIoU ≥ 0.5 (với R5 K-Medoids, dùng tIoU của clip quanh medoid, hoặc bỏ loss này). Negatives là **các đoạn khác trong cùng video**:

$$
\mathcal{L}_{evt}=-\log\frac{\exp(s_{i^+}/\tau)}{\sum_{i}\exp(s_i/\tau)}
$$

Loss này ép các bước giống nhau trong cùng video phải tách xa nhau, đồng thời chống việc self-attention làm các đoạn "dính" vào nhau.

### Tổng

$$
\mathcal{L}=\mathcal{L}_{vid}+\lambda\,\mathcal{L}_{evt}
$$

Với các biểu diễn chỉ có 1 vector (R0, R1), $\mathcal{L}_{evt}$ không áp dụng. Để công bằng, báo cáo thêm một bảng phụ mà **mọi** cách biểu diễn chỉ train bằng $\mathcal{L}_{vid}$.

Checkpoint và siêu tham số chọn theo **dev R@1**, không bao giờ chọn theo test.

---

## 7. Dữ liệu, protocol và metrics

### YouCook2

- Tập test không công bố annotation, nên dùng **val làm test**.
- Tách 10% video của train làm **dev**.
- **Bắt buộc:** Omni và UniAV chỉ được fine-tune trên train (không bao gồm dev/val). Nếu checkpoint hiện tại đã thấy val thì phải fine-tune lại trên train.
- Loại các video không còn tải được, và báo cáo số video và caption thực tế.

### Báo cáo năng lực backbone sau fine-tune

- Omni: R@1 của event-clip retrieval trên dev (text → đúng clip).
- UniAV: mAP@tIoU {0.3, 0.5, 0.7} trên val, và số proposal trung bình mỗi video so với số event GT.

### Metrics

- **Text→Video (chính):** R@1, R@5, R@10, MedR. Query tìm video gốc trong toàn bộ gallery.
- **Video→Text** (theo MeVTR): R@k-Average, R@k-One-Hit, R@k-All-Hit.
- **Moment (tạp chí):** VCMR R@k với tIoU ≥ 0.5 / 0.7. Kết quả tính là đúng khi đúng video và đúng đoạn.
- **Hiệu năng:** số vector mỗi video, kích thước index, độ trễ mỗi query.

---

## 8. Thí nghiệm

### Exp 1 — Bảng chính: so sánh các cách biểu diễn (RQ1, RQ2, RQ3)

| Cách biểu diễn | #vec | (a) Omni pretrained | (b) Omni fine-tuned | (c) Omni fine-tuned + Context Fusion |
|---|---|---|---|---|
| R0 Global | 1 | | | |
| R1 Event-Single | 1 | | | |
| R2 Uniform-Chunk | thay đổi | | | |
| R3 Uniform-M | $K_{pred}$ | | | |
| R4 Random-M | $K_{pred}$ | | | |
| R5 K-Medoids-M | $K_{pred}$ | | | |
| **R6 UniAV-Pred** | $K_{pred}$ | | | **UEMR** |
| R7 GT Event (oracle) | $K_{GT}$ | | | |
| Uniform-M @ $K_{GT}$ | $K_{GT}$ | | | |

- Cột (a) và (b) **không cần train gì**, chỉ cần cache và MaxSim.
- Cột (a) dùng để xử lý confound: Omni fine-tune trên clip ngắn nên Global có thể bị thiệt. Nếu thứ hạng giữa các dòng giữ nguyên ở cả (a) và (b), kết luận sẽ vững.
- Cột (c): mỗi dòng train một fusion riêng với cùng cấu hình.

**Logic đọc bảng:**

```text
R0 Global  vs  R1 Event-Single  vs  R6 UniAV-Pred
   → lợi ích của việc GIỮ NHIỀU VECTOR (R1 và R6 cùng thông tin, chỉ khác có nén hay không)

R2 Uniform-Chunk  vs  R3 Uniform-M
   → ảnh hưởng của SỐ LƯỢNG vector

R3 Uniform-M / R4 Random-M  vs  R6 UniAV-Pred        (cùng K)
   → lợi ích của RANH GIỚI NGỮ NGHĨA, không phải số lượng

R5 K-Medoids-M  vs  R6 UniAV-Pred                    (cùng K)
   → cấu trúc THỜI GIAN vs đa dạng NỘI DUNG  (điểm khác biệt với MeVTR)

R6 UniAV-Pred  vs  R7 GT                             → khoảng cách detector–oracle
R7 GT  vs  Uniform-M @ K_GT                          → giá trị của ranh giới khi số lượng như nhau

cột (b)  vs  cột (c)                                 → giá trị của ngữ cảnh
```

**Giả thuyết:** R7 ≥ R6 > R5 > R3 ≈ R4 > R0 ≈ R1. Nếu kết quả khác đi, nghiên cứu vẫn hợp lệ vì giả thuyết bị bác bỏ một cách có kiểm soát, và đó cũng là một phát hiện.

### Exp 2 — Ablation Context Fusion (RQ3), trên R6

| Setting | R@1 | R@5 | R@10 |
|---|---|---|---|
| Omni thuần (không fusion) | | | |
| + Bước 1 (ngữ cảnh UniAV cục bộ) | | | |
| + Bước 2 (self-attention giữa các đoạn) | | | |
| + $\mathcal{L}_{evt}$ | | | |
| + vector global $c_g$ = **UEMR đầy đủ** | | | |
| *Đối chứng:* $Z$ bị xáo trộn giữa các video | | | |
| *Đối chứng:* $\delta=0$ (không nhìn ra ngoài đoạn) | | | |

Nếu phiên bản $Z$ xáo trộn vẫn tăng điểm, thì phần tăng đến từ số tham số chứ không phải từ ngữ cảnh.

Báo cáo thêm giá trị $\alpha$ học được.

### Exp 3 — Chất lượng ranh giới (RQ4)

Làm nhiễu ranh giới GT (dịch và co giãn ngẫu nhiên) để đạt tIoU trung bình so với GT lần lượt là 1.0 / 0.9 / 0.7 / 0.5 / 0.3. Vẽ R@1 theo tIoU, với hai đường: không có fusion và có fusion. Đặt các điểm UniAV-Pred, Uniform-M và Global lên cùng đồ thị.

Hai câu hỏi cần trả lời:

- Ranh giới cần chính xác đến mức nào thì mới thắng cách chia đều?
- Fusion có làm hệ thống **bền hơn** với ranh giới kém không (đường có fusion dốc xuống chậm hơn)?

### Exp 4 — Audio

Trên R6 và R7: so sánh visual-only với audio-visual, cho cả phía Omni ($e_i$) và phía UniAV ($Z$).

### Exp 5 — Scoring và hiệu năng

- So sánh Max, Top-k mean, LSE, và Max + $\beta\log\text{conf}_i$ (prior từ UniAV).
- Báo cáo số vector, kích thước index và độ trễ của từng cách biểu diễn. Liên hệ với ColBERTv2 về đánh đổi giữa lưu trữ và độ chính xác.

### Exp 6 (tạp chí) — Tổng quát hoá và localization

- Lặp lại Exp 1 và Exp 2 trên **ActivityNet Captions**: có audio, có ranh giới sự kiện, và có số liệu Me-Retriever để so sánh trực tiếp.
- VCMR: dùng $[t^s_{i^*},t^e_{i^*}]$ để đánh giá moment localization trên cùng index.

### Phân tích định tính (cả khóa luận và tạp chí)

- Các query mà Global thắng UniAV-Pred: chúng có đặc điểm gì chung?
- Các cặp bước giống nhau (ví dụ nhiều lần "khuấy", "thêm muối"): fusion có tách được không?
- Trực quan hoá trọng số attention của bước 1: đoạn nhìn vào phần ngữ cảnh nào?

---

## 9. Đóng góp

1. **UEMR**: framework kết hợp UniAV (tìm sự kiện và ngữ cảnh thời gian) với OmniRetriever (không gian audio-video-text chung) để tạo biểu diễn đa vector theo sự kiện cho video dài.
2. **Controlled study** gồm 8 cách biểu diễn trên cùng một backbone. Nghiên cứu tách riêng ba nguồn lợi ích: *có nhiều vector*, *số lượng vector* và *ranh giới sự kiện theo thời gian*, đồng thời so sánh trực tiếp với cách chọn key-event theo nội dung của MeVTR.
3. **Context Fusion có gate**, giữ nguyên không gian của Omni, kết hợp ngữ cảnh cục bộ từ UniAV và ngữ cảnh giữa các đoạn. Module được train bằng MIL kết hợp hard negative trong cùng video, không cần ranh giới GT ở inference.
4. **Phân tích độ nhạy theo chất lượng ranh giới**, định lượng khoảng cách giữa detector và oracle.
5. *(Tạp chí)* Truy hồi video và định vị moment bằng **một index duy nhất**.

### Không claim

- Multi-vector retrieval, MaxSim hay InfoNCE là mới.
- Omni hay UniAV là mô hình mới.
- Kết quả với GT là hệ thống thực tế. GT luôn được gọi là **oracle**.

---

## 10. Phạm vi

### Khóa luận (YouCook2) — bắt buộc

- Cache offline: UniAV ($Z$, proposals) và Omni (mọi cách chia, global, text).
- Exp 1 đầy đủ (ba cột).
- Exp 2.
- Exp 3.
- Phân tích định tính.

### Khóa luận — nên có

- Exp 4 (audio).
- Exp 5 (scoring).
- 3 seed cho R0, R3, R6 và UEMR.

### Tạp chí — thêm

- Exp 6: ActivityNet Captions và VCMR.
- 3 seed cho toàn bộ Exp 1, kèm paired bootstrap significance test.
- Metrics V2T theo MeVTR.
- Phân tích hiệu năng chi tiết.

### Không làm

- Fine-tune chung Omni và UniAV end-to-end.
- Sinh caption bằng LLM.
- Fusion nhiều pyramid level của UniAV.
- Multi-vector phía text (ghi vào future work).
- Fusion-as-Teacher distillation của OmniRetriever (không phục vụ các RQ).

---

## 11. Thứ tự triển khai

```text
P0  Protocol
    - Kiểm tra split fine-tune của Omni và UniAV (chỉ train?)
    - Tách dev 10% train. Liệt kê các video tải được.

P1  UniAV offline
    - Trích Z (1 level), mapping giây→index, proposals + conf
    - Đo mAP trên val. Thống kê K_pred so với K_GT.

P2  Sinh đoạn cho R0–R7 (chỉ timestamp, chưa chạy Omni)
    - Lưu thành JSON: video_id → list[(ts, te)] cho từng cách chia

P3  Omni offline
    - e_i cho mọi đoạn của mọi cách chia, e_g, q cho mọi caption
    - Cả Omni pretrained và Omni fine-tuned

P4  Exp 1 cột (a), (b)  ← ĐÃ CÓ KẾT QUẢ CHÍNH CHO KHÓA LUẬN
P5  Exp 3 (nhiễu ranh giới, không fusion)
P6  Cài đặt Context Fusion → train với L_vid → thêm L_evt
P7  Exp 1 cột (c), Exp 2, Exp 3 (có fusion)
P8  Exp 4, Exp 5, phân tích định tính
P9  (Tạp chí) Lặp lại P1–P8 trên ActivityNet Captions + VCMR
```

**Lưu ý chi phí:** P3 tốn nhiều nhất vì phải chạy Omni-7B trên mọi đoạn của 8 cách chia × 2 checkpoint. Nên ước lượng số clip trước. Có thể giảm tải bằng cách: Uniform-Chunk chỉ chạy 1 giá trị $L$; cột (a) chỉ chạy R0, R3, R6, R7.

---

## 12. Siêu tham số khởi điểm (gợi ý, chỉnh trên dev)

| Tham số | Giá trị gợi ý | Ghi chú |
|---|---|---|
| UniAV pyramid level | stride khoảng 1–2 giây | Cố định cho mọi thí nghiệm |
| $\theta$ (conf) | chọn sao cho trung bình $K_{pred}\approx K_{GT}$ trên dev | |
| $K_{\max}$ | 16 | Giống MeVTR |
| $d_{\min}$ | 2 giây | |
| $\delta$ (cửa sổ ngữ cảnh) | 5 giây | Ablation: 0 / 5 / 10 |
| Fusion heads / layers | 8 heads, 1 lớp cross-attn + 1 lớp self-attn | |
| $\tau$ (InfoNCE) | 0.05, hoặc learnable | |
| $\tau_s$ (LSE) | 0.1 | |
| $\lambda$ | 0.5 | |
| Optimizer | AdamW, lr 1e-4, weight decay 0.01, cosine schedule | |
| Batch | 128 (1 caption / video) | |
| Epoch | 20–30, early stopping theo dev R@1 | |

---

## 13. Pseudocode

### Offline

```python
for v in videos:
    Z, t2idx = uniav.features(v, level=L)                   # frozen
    props    = postprocess(uniav.proposals(v))              # [(ts, te, conf)]
    K        = len(props)
    parts = {
        "global":     [(0, v.duration)],
        "uni_chunk":  chunks(v.duration, L_chunk),
        "uni_M":      uniform(v.duration, K),
        "rand_M":     random_segments(v.duration, K, d_min, seed),
        "kmedoids_M": medoid_windows(Z, t2idx, K, half=1.0),
        "uniav_pred": [(ts, te) for ts, te, _ in props],
        "gt":         v.gt_segments,
        "uni_M_gt":   uniform(v.duration, len(v.gt_segments)),
    }
    for name, segs in parts.items():
        E = [omni.av(clip(v, ts, te)) for ts, te in segs]   # frozen
        cache(v.id, name, segs, E)
    cache(v.id, "Z", Z, t2idx, props)

for cap in captions:
    cache_text(cap.id, omni.text(cap.sentence))
```

### Context Fusion

```python
class ContextFusion(nn.Module):
    def __init__(self, d, d_z, heads=8):
        super().__init__()
        self.z_proj = nn.Linear(d_z, d)
        self.rel    = nn.Embedding(3, d)                     # trước / trong / sau đoạn
        self.cross  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.inter  = nn.TransformerEncoderLayer(d, heads, batch_first=True)
        self.out    = nn.Linear(d, d)
        self.alpha  = nn.Parameter(torch.zeros(1))

    def forward(self, E, Z_windows, rel_ids, pos):
        # E: [K,d]; Z_windows[i]: [T_i,d_z]; rel_ids[i]: [T_i]; pos: [K,d]
        r = []
        for e, Zw, rid in zip(E, Z_windows, rel_ids):
            kv = self.z_proj(Zw) + self.rel(rid)
            o, _ = self.cross(e[None, None], kv[None], kv[None])
            r.append(o[0, 0])
        u = self.inter((E + torch.stack(r) + pos)[None])[0]
        return F.normalize(E + self.alpha * self.out(u), dim=-1)
```

### Training

```python
for batch in loader_one_caption_per_video:
    q = F.normalize(load_text(batch), dim=-1)               # [B,d]
    bags, pos_idx = [], []
    for s in batch:
        segs, E = load(s.video_id, partition)
        Zw, rid = windows(s.video_id, segs, delta)
        C = fusion(E, Zw, rid, temporal_pos(len(segs)))
        bags.append(C)
        pos_idx.append(best_tiou(segs, s.gt_segment, thr=0.5))  # None nếu < thr
    S     = lse_sim_matrix(q, bags, tau_s)                  # [B,B]
    loss  = infonce(S, tau)
    loss += lam * intra_video_nce(q, bags, pos_idx, tau)
    loss.backward(); opt.step(); opt.zero_grad()
```

### Inference

```python
for cap in test_captions:
    q = load_text(cap)
    results = []
    for v in gallery:
        s = C[v] @ q                                        # [K_v]
        i = int(s.argmax())
        results.append((v.id, float(s[i]), segs[v][i]))     # điểm + timestamp
    ranking = sorted(results, key=lambda x: -x[1])
    evaluate(cap, ranking)                                  # R@k, MedR, (VCMR)
```

---

## 14. Checklist trước khi chạy

- [ ] Omni và UniAV chỉ fine-tune trên train, val chưa từng được thấy.
- [ ] Có dev split, và chọn checkpoint theo dev.
- [ ] Mapping giây → index của UniAV đã được kiểm tra trên 1 video (vẽ $Z$ so với GT).
- [ ] Mọi cách chia dùng cùng pyramid level, cùng $d_{\min}$, cùng cách cắt clip cho Omni.
- [ ] R3, R4, R5 dùng đúng $K_{pred}$ của từng video.
- [ ] Batch không chứa 2 caption của cùng một video.
- [ ] Gallery ở test chứa toàn bộ video val, và query là toàn bộ caption val.
- [ ] Ghi lại seed, config và hash của cache cho mọi lần chạy.
