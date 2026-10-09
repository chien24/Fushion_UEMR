# Nhận xét kết quả Omni event retrieval trên YouCook2 (ft và pt)

Ngày chạy: 2026-10-09.

| Lần chạy | Nội dung | Log |
|---|---|---|
| 1 | Encode và đánh giá metric thường (cell 1–15) | `log_ft.ipynb`, `log_pt.ipynb` |
| 2 | Phân tích lỗi và metric "đúng nghĩa" (cell 16) | `new/new_log_ft.ipynb`, `new/new_log_pt.ipynb`. Cell 16 phân tích cả ft và pt trong một lần, nên hai file có cùng bảng |

**Tóm tắt:**
1. Pipeline retrieval tái lập được kết quả eval lúc fine-tune: R@1 35.64, so với 35.68 của eval.
2. Fine-tune giúp rõ ở mọi metric:
   - `event R@1` tăng từ 20.9 lên **32.4**.
   - Theo nghĩa (`sem@1`, tau = 0.8): tăng từ 30.7 lên **43.7**.
   - Số event "hub" giảm mạnh.
3. Metric `event R@1` đánh giá thấp chất lượng thật: top-1 là **đúng bước nấu** trong **44–56%** query (tùy ngưỡng), và top-5 có đúng bước trong **76–86%** query.
4. Tuy vậy, trong số các câu sai, phần "đúng bước nhưng ở video khác" chỉ chiếm **khoảng 17–34%**, **không phải phần lớn** như bản nhận xét trước giả định. Phần lớn còn lại là trả về một bước thật sự khác.
5. Ba yếu tố làm giảm `event R@1` rõ rệt:
   - Bước nấu bị lặp lại ở nhiều video: R@1 giảm từ 40 xuống 11.
   - Event dài: R@1 giảm từ 40 xuống 11.
   - Câu query ngắn: R@1 giảm từ 39 xuống 22.

---

## 1. Chạy như thế nào

### 1.1 Môi trường

| Thành phần | Giá trị |
|---|---|
| Nơi chạy | Google Colab, GPU NVIDIA A100-SXM4-40GB |
| Code | repo `chien24/Fushion_UEMR`, nhánh `main` (lần 1: commit `fc7c7a9`; lần 2: thêm `omni_retrieval/analysis.py`) |
| Notebook | `notebooks/omni_retrieval_colab.ipynb` (16 cell) |
| Thư viện | Python 3.13.15, torch 2.11.0+cu130, transformers 4.51.3, peft 0.21.1; phân tích dùng thêm sentence-transformers |
| Code model | `third_party/omni_fix/`: copy nguyên văn từ Omni-fix commit `460257f`, không clone lúc chạy |
| Unit test | 23 passed (lần 2) |

### 1.2 Model (không train lại, chỉ chạy forward)

Cả hai cấu hình dùng chung model nền **WAVE-7B** (Qwen2.5-Omni thinker) + **BEATs**. Chỉ adapter LoRA là khác:

| TAG | Adapter LoRA | Nguồn |
|---|---|---|
| `pt` | OmniRetriever-7B gốc, chưa fine-tune | HuggingFace `YunzeLiu/OmniRetriever-7B` |
| `ft` | `checkpoint-1948`, fine-tune trên YouCook2 (trainsplit) | Drive `Colab Notebooks/code KL/Omni/checkpoint-1948` |

Cả hai adapter đều có đủ fusion head (`classify_linear`, `beats_ln`, `beats_proj`): 168 tensor LoRA, tổng 178 tensor.

### 1.3 Dữ liệu: database và query

- **Gallery:** 498 video = 394 video val + 104 video dev. Video train không được dùng, vì model `ft` đã học chúng.
- **Database:** **3.803 event**. Mỗi event GT của YouCook2 (`video_id`, `[t_s, t_e]`) là một vector. Gồm 3.030 event của video val và 773 event của video dev; event dev đóng vai trò nhiễu.
- **GT queries:** 3.030 caption val. GT của mỗi caption là chính event của nó.
- **Query tự tạo:** 26 câu trong `queries/custom_queries.jsonl`. Mỗi câu diễn đạt lại một caption val bằng từ khác. GT `[ts, te]` được map sang event cùng video có tIoU lớn nhất (≥ 0.5).

### 1.4 Cách tạo vector (giống hệt `eval_youcookii.py` lúc fine-tune)

- **Event (video + audio):**
  - Pipeline `LazySupervisedDataset` của Omni-fix: decord lấy **8 frame**, resize về 50.176 pixel (giữ tỉ lệ).
  - Audio cắt từ chính mp4, center-crop hoặc pad về **8 giây**, token `<|AUDIO|>` nhân đôi.
  - Prompt `"<video>\nPlease describe the video."`, batch 1.
  - Vector lấy từ head `classify_linear` (all_layer), chuẩn hoá L2.
- **Text (query):** `caption + <|im_end|>` đi qua text model của thinker. Vector là hidden state lớp cuối tại token cuối (nhánh label của eval, **không** qua `classify_linear`), chuẩn hoá L2.
- **Tìm kiếm:** cosine giữa query và toàn bộ 3.803 event, sắp xếp giảm dần.

### 1.5 Các bước chạy

**Lần 1, cho mỗi TAG:**
1. Cell 3: đặt `TAG = 'ft'` (hoặc `'pt'`).
2. Cell 5: `SMOKE = False`, `N_VIDEOS = 500` (thực tế 498), `SEED = 0`.
3. Chạy lần lượt:
   - Cell 1–6: setup và unit test.
   - Cell 7: chọn video.
   - Cell 8–9: encode 3.803 event. Với ft mất khoảng 17 phút, tức 0.26 s/event, 0 lỗi.
   - Cell 10: encode 3.030 caption.
   - Cell 11: tạo database.
   - Cell 12: đánh giá GT queries.
   - Cell 13: đánh giá query tự tạo.
   - Cell 15: gõ query tay.

**Lần 2, phân tích (không cần GPU, không nạp Omni):**
1. Chạy cell 1–2.
2. Cell 3: đặt `DOWNLOAD_MODELS = False`, rồi chạy.
3. Chạy cell 5, rồi cell 6 (23 passed). Bỏ qua cell 4.
4. Chạy **cell 16**. Cell này đọc `events.npz`, `text.npz` và vector query tự tạo của cả `ft` lẫn `pt`.

Kết quả được lưu trên Drive:
- Lần 1: `MyDrive/uemr/omni_cache/<TAG>/` (`results_gt.json`, `per_query_gt.csv`, `results_custom.json`, `per_query_custom.csv`).
- Lần 2: `MyDrive/uemr/omni_cache/<TAG>/analysis/` (`gt_summary.json`, `gt_per_query.csv`, `gt_by_*.csv`, `gt_hubs.csv`, `gt_tail.csv`, `gt_calibration.csv`, `custom_*`).

### 1.6 Kiểm tra tính đúng đắn

- **Sanity check (cell 12):** chạy text→clip trên gallery giống eval lúc fine-tune (3.030 clip). `ft` cho **R@1 35.64**. Eval lúc fine-tune là **35.68**, đọc từ `best_val.json` trong log smoke cùng ngày; file này bị mất trên Drive trước lần chạy đầy đủ. Lệch **0.04 điểm**.
- **Kiểm tra vector (cell 15):** encode lại caption đầu tiên rồi so với vector đã lưu. Kết quả cos = 0.9998 cho cả ft và pt.

### 1.7 Định nghĩa metric

| Metric | Nghĩa |
|---|---|
| `event R@k` | Event GT (đúng video **và** đúng đoạn) nằm trong top-k |
| `video@k` | Top-k có ít nhất một event thuộc đúng video |
| `tIoU.5@1` | Top-1 đúng video và trùng với GT ít nhất tIoU 0.5 |
| MedR / MnR / MRR | Hạng trung vị / hạng trung bình / trung bình của 1/hạng, của event GT |
| `exact-caption@k` | Top-k có event mà caption **trùng hệt** caption GT (sau khi chuyển chữ thường và bỏ dấu câu) |
| `sem@k (tau)` | Top-k có event mà caption **cùng nghĩa** với caption GT: cosine ≥ `tau` theo model chấm (judge) **độc lập với Omni** là `sentence-transformers/all-MiniLM-L6-v2`. Judge chỉ đọc annotation và chỉ dùng để chấm, không tham gia retrieval |
| Near-duplicate | Event khác (ở video khác) có caption cùng nghĩa với caption GT (judge ≥ 0.8) |

---

## 2. Kết quả

### 2.1 GT queries: metric thường (3.030 caption val, tìm trên 3.803 event)

| | event R@1 | R@5 | R@10 | MedR | MnR | MRR | video@1 | video@5 | tIoU.5@1 |
|---|---|---|---|---|---|---|---|---|---|
| pt | 20.86 | 45.38 | 57.33 | 7 | 56.46 | 32.62 | 23.47 | 49.21 | 20.86 |
| **ft** | **32.38** | **62.57** | **72.94** | **3** | **23.19** | **46.06** | **34.16** | **65.54** | **32.38** |

Trên gallery giống eval lúc fine-tune (3.030 clip): pt có R@1 22.41, R@5 49.21; ft có R@1 35.64, R@5 65.68.

### 2.2 GT queries: metric "đúng nghĩa"

| | ft | pt |
|---|---|---|
| event R@1 / R@5 / R@10 | 32.38 / 62.57 / 72.94 | 20.86 / 45.38 / 57.33 |
| exact-caption@1 / @5 / @10 | 33.07 / 63.66 / 73.96 | 21.25 / 46.53 / 58.42 |
| sem@1 (tau 0.7 / 0.8 / 0.9) | **55.54** / **43.70** / 35.71 | 45.02 / 30.73 / 23.37 |
| sem@5 (tau 0.7 / 0.8 / 0.9) | **85.84** / **75.81** / 66.96 | 77.56 / 60.89 / 50.07 |
| sem@10 (tau 0.7 / 0.8 / 0.9) | 91.98 / 84.32 / 77.00 | 87.33 / 72.41 / 62.28 |
| % câu sai ở top-1 mà thật ra là đúng bước (tau 0.8 / 0.7) | 16.7 / 34.2 | 12.5 / 30.5 |
| % query có ≥ 1 near-duplicate ở video khác | 53.2 | 53.2 |

Cách tính "% câu sai mà thật ra đúng bước": (sem@1 − R@1) / (100 − R@1). Ví dụ với ft, tau 0.7: (55.54 − 32.38) / 67.62 = 34.2%.

**Hiệu chỉnh ngưỡng.** Mình xem bằng mắt 10 cặp (caption GT, caption top-1 ở video khác) ngẫu nhiên trong mỗi khoảng điểm của judge (bảng [3] của cell 16):

| Khoảng judge | Số cặp thật sự cùng bước | Ví dụ |
|---|---|---|
| ≥ 0.9 | 10/10 | "boil the potatoes in water" ↔ "boil the potatos in water" |
| 0.8–0.9 | 10/10 | "coat the fish with flour and batter" ↔ "coat the fish in flour and batter" |
| 0.7–0.8 | 9/10 | "heat up oil in a wok" ↔ "pour peanut oil into the wok"; "mash the boiled potatoes with some milk added" ↔ "mash the potatoes" |
| 0.6–0.7 | khoảng 7/10 | "add salt and pepper to the pork" ↔ "season the meat with salt and pepper"; nhưng "add potatos to the pot" ↔ "push the potatoes through the sieve" là khác bước |
| 0.5–0.6 | khoảng 5/10 | phần lớn chỉ là bước liên quan (cùng nguyên liệu), không cùng hành động |

=> **tau = 0.8 là ngưỡng chặt** (gần như không có dương tính giả, nhưng bỏ sót nhiều cách diễn đạt khác nhau). **tau = 0.7 sát thực tế hơn.** Vì vậy các con số "đúng nghĩa" thật nằm trong khoảng giữa tau 0.8 và tau 0.7, có thể cao hơn một chút.

### 2.3 Top-1 thuộc loại nào (GT queries, tau = 0.8)

| Top-1 là | ft | pt |
|---|---|---|
| Đúng event GT | 32.38% | 20.86% |
| Cùng video, sai đoạn | 1.78% | 2.61% |
| Video khác, **cùng nghĩa** | 10.99% | 9.57% |
| Video khác, **khác nghĩa** | 54.85% | 66.96% |

Với tau 0.8, nhóm "video khác, khác nghĩa" vẫn còn chứa các cách diễn đạt khác của cùng một bước. Ở tau 0.7, nhóm này của ft còn khoảng 44%.

### 2.4 Phân tích theo nhóm (GT queries)

**Theo độ dài event GT:**

| Độ dài | n | ft R@1 | ft R@5 | ft sem@1 (0.8) | pt R@1 |
|---|---|---|---|---|---|
| < 5s | 283 | 34.98 | 65.02 | 49.82 | 22.97 |
| 5–10s | 672 | **39.73** | 68.90 | 50.60 | 25.60 |
| 10–20s | 982 | 32.89 | 64.66 | 44.60 | 23.83 |
| 20–40s | 768 | 29.17 | 59.64 | 40.36 | 16.67 |
| 40–80s | 280 | 22.50 | 50.00 | 31.43 | 10.36 |
| ≥ 80s | 45 | **11.11** | 35.56 | 15.56 | 8.89 |

**Theo số near-duplicate ở video khác** (bước nấu bị lặp lại bao nhiêu lần trong database):

| Near-duplicate | n (%) | ft R@1 | ft R@5 | ft sem@1 (0.8) | pt R@1 |
|---|---|---|---|---|---|
| 0 | 1419 (46.8%) | **40.10** | 68.01 | 40.31 | 26.64 |
| 1–2 | 829 (27.4%) | 31.24 | 63.69 | 43.43 | 19.90 |
| 3–9 | 643 (21.2%) | 21.62 | 54.43 | 50.08 | 12.13 |
| 10–29 | 131 (4.3%) | **10.69** | 40.46 | 47.33 | 8.40 |
| ≥ 30 | 8 (0.3%) | 0.00 | 0.00 | 100.00 | 0.00 |

**Theo độ dài câu query:**

| Độ dài | n | ft R@1 | pt R@1 |
|---|---|---|---|
| 1–4 từ | 239 | **22.18** | 10.88 |
| 5–8 từ | 1455 | 30.17 | 18.76 |
| 9–12 từ | 862 | 35.15 | 23.67 |
| ≥ 13 từ | 474 | **39.24** | 27.22 |

**Event "hub"** (event bị trả về top-1 quá nhiều lần; trung bình mỗi event nên là 0.8 lần):

| | ft | pt |
|---|---|---|
| Số lần top-1 nhiều nhất của một event | 12 | **35** |
| % event không bao giờ lọt vào top-10 của query nào | 2.7% | **15.2%** |
| Tỉ lệ slot top-1 / tỉ lệ event, theo độ dài event (<5s, 5–10, 10–20, 20–40, 40–80, ≥80s) | 1.17, 0.95, 0.94, 1.07, 1.03, **0.63** | 1.40, 0.85, 1.01, 1.04, 0.81, 0.97 |

Các hub của ft đều là bước chung chung: "push the potatoes through the sieve onto a pan" (12 lần), "cook bacon in a pan until crispy" (10), "in a pan heat oil on medium heat" (9, event dài 2 giây), "add oil to a pan" (9, 2 giây).

Hub mạnh nhất của pt: "fry chopped onions garlic and salt" (35 lần).

**Đuôi dài:** số query có hạng > 100 là ft 146 (4.8%), pt 380 (12.5%).

### 2.5 Query tự tạo (26 câu)

| | event R@1 | R@5 | R@10 | MedR | sem@1 (0.7 / 0.8) | sem@5 (0.7 / 0.8) |
|---|---|---|---|---|---|---|
| pt | 11.54 | 26.92 | 42.31 | 21.5 | 30.77 / 23.08 | 73.08 / 53.85 |
| **ft** | **26.92** | **38.46** | **50.00** | **10.5** | **46.15 / 34.62** | 69.23 / **61.54** |

Hạng của event GT cho từng câu (1 nghĩa là đúng ở top-1):

| # | Query | ft | pt | Top-1 của ft (judge) |
|---|---|---|---|---|
| 0 | trim off the tips of the purslane leaves | **1** | **1** | đúng |
| 1 | peel the small onions and chop them very finely | 3 | 3 | cut a shallot into thin slices (0.68) |
| 2 | shred potatoes, wash them and dry them with a cloth | 11 | 26 | grate the potatos and squeeze the starch out (0.76) |
| 3 | make a sauce from soy, sesame oil, sugar, scallions and garlic | 10 | 316 | mix wasabi soy sauce sesame oil and chives (0.64) |
| 4 | stir ricotta together with coconut sugar | 153 | 289 | add sweetened condensed milk to the pan (0.50) |
| 5 | transfer the lentils into a saucepan | **1** | 3 | đúng |
| 6 | bring a pot of water to a boil | 85 | 28 | bring a large pan of water to boil (0.74) |
| 7 | cook the meat in a skillet with some beer | **1** | 2 | đúng |
| 8 | season with dried italian herbs | 68 | 415 | add some tomato paste seasoned beef… (0.33) |
| 9 | toss the chicken wings with spices and hot sauce | **1** | **1** | đúng |
| 10 | put scallions, cilantro, pepper and stock into the noodles | 17 | 3 | put some of the vegetables in to the pot… (0.62) |
| 11 | cook the flatbread on a hot griddle with oil | **1** | **1** | đúng |
| 12 | mash the potatoes with milk and black pepper | 51 | 91 | mash the potatos until chunky (0.61) |
| 13 | arrange apple slices and foie gras on the plate | 16 | 21 | place a piece of bread onto a plate and add a piece of foie shallots and apple (0.69) |
| 14 | dredge the chicken pieces in flour | 296 | 1101 | dip the chicken in the milk and the flour mixture (0.74) |
| 15 | put the burger patty on the lower bun | **1** | 6 | đúng |
| 16 | deep fry the fish | 28 | 158 | deep fry the samosas (0.44) |
| 17 | pour some oil into the frying pan | 7 | 11 | add oil to a pan (0.95) |
| 18 | combine the pork with the dressing in a bowl | 659 | 1122 | transfer the marinated beef meat into a container… (0.31) |
| 19 | fold the crepe in half | 7 | 80 | spread some sushi rice on top of the seaweed (0.23) |
| 20 | fry sausages with onions while stirring | 67 | 954 | add the vodka to the pot (0.35) |
| 21 | top it with grated cheese and black pepper | 16 | 115 | grate some parmesan cheese on it (0.59) |
| 22 | turn the chicken wings over | **1** | 9 | đúng |
| 23 | warm up oil for frying | 25 | 22 | deep fry sealed samosa… (cùng video GT, 0.53) |
| 24 | finely cut fresh parsley | 4 | 10 | chop up the parsley (0.99) |
| 25 | whisk soy sauce, chili sauce and sugar together | 3 | 6 | mix soy sauce sesame oil sugar… (0.67) |

So từng câu: ft tốt hơn pt ở 19/26 câu, bằng ở 4 câu, kém ở 3 câu (#6, #10, #23).

### 2.6 Thử tay ở cell 15

| Query | ft | pt |
|---|---|---|
| arrange apple slices and foie gras on the plate | Video khác: "place a piece of bread onto a plate and add a piece of foie shallots and apple" (GT ở hạng 16) | "place the radish and chives on top" |
| toss the chicken wings with spices and hot sauce | ✓ đúng event GT | ✓ đúng event GT |
| pour some oil into the frying pan | Video khác: "add oil to a pan"; cả top-5 đều là "add/heat/pour oil … pan/wok" (GT ở hạng 7) | Video khác: "add oil to a pan" (GT ở hạng 11) |

---

## 3. Vì sao có kết quả này

### 3.1 Pipeline đúng; con số là năng lực thật của model

Sanity check tái lập R@1 của eval lúc fine-tune với độ lệch 0.04 điểm, và vector encode lại khớp vector đã lưu (cos 0.9998). Muốn kết quả cao hơn phải thay đổi model, cách encode hoặc cách chia event, không phải sửa code retrieval.

**Lưu ý:** database dùng **ranh giới event GT**, tương ứng với **R7 (oracle)** trong UEMR. Vì vậy 32.4% là kết quả của Omni khi các event được chia **hoàn hảo**, chưa có Context Fusion.

### 3.2 Fine-tune có tác dụng rõ, cả về hạng lẫn về nghĩa

- R@1 tăng khoảng 1,55 lần (20.9 → 32.4) và MedR từ 7 xuống 3.
- sem@1 (tau 0.8) tăng từ 30.7 lên 43.7.
- Fine-tune làm **giảm hub**: số lần top-1 nhiều nhất của một event giảm từ 35 xuống 12, và số event "vô hình" (không bao giờ vào top-10) giảm từ 15.2% xuống 2.7%. Không gian vector sau fine-tune phân bố đều hơn.
- Với query tự tạo, những câu model chưa từng thấy, R@1 cũng tăng từ 11.5 lên 26.9.

### 3.3 Metric `event R@1` đánh giá thấp, nhưng bước trùng lặp không phải nguyên nhân chính của lỗi

*(Sửa lại nhận định của bản trước.)*

- **Metric đánh giá thấp:** top-1 là đúng bước nấu trong 43.7% (tau 0.8) đến 55.5% (tau 0.7) query, so với 32.4% theo `event R@1`. Top-5 chứa đúng bước trong 76–86% query. Với người dùng tìm kiếm, kết quả tốt hơn nhiều so với con số 32%.
- **Gần như không có caption trùng hệt từng chữ:** `exact-caption@1` chỉ cao hơn R@1 0.7 điểm. Các bước lặp lại trong YouCook2 được viết bằng nhiều cách khác nhau.
- **Nhưng trùng lặp không phải phần lớn lỗi.** Trong các câu sai ở top-1, chỉ **17% (tau 0.8) đến 34% (tau 0.7)** là đúng bước ở video khác. Bản nhận xét trước suy ra từ tỉ lệ `event R@1 / video@1` (95%) rằng "lỗi chủ yếu là đúng thao tác, sai video". Suy luận đó chỉ đúng một phần: model ít khi sai đoạn trong cùng video (1.8%), nhưng khi chọn nhầm video thì phần lớn (khoảng 44–55% tổng số query) là **một bước thật sự khác**.
- **Mức độ trùng lặp vẫn ảnh hưởng mạnh tới `event R@1`.** Bước nấu càng phổ biến thì R@1 càng thấp: 40.1 khi không có near-duplicate, 21.6 với 3–9, 10.7 với 10–29. Ngược lại sem@1 tăng theo (40 → 50 → 47). Model không "sai" ở các bước phổ biến; nó trả về đúng loại bước nhưng không biết chọn video nào. Muốn phân biệt được các bước này cần **ngữ cảnh** (món gì, bước trước và sau).
- **Ngay cả với các bước không bị trùng lặp** (47% query), R@1 cũng chỉ đạt 40%. Phần lỗi này thuộc về chất lượng embedding của Omni.

### 3.4 Event dài khó hơn rõ rệt, nhưng không thành hub

*(Sửa lại nhận định của bản trước.)*

- R@1 của ft giảm dần theo độ dài event: khoảng 40 ở 5–10 giây, 29 ở 20–40 giây, 22.5 ở 40–80 giây, **11** ở ≥ 80 giây. pt cũng có cùng xu hướng.
- Bản trước đoán event dài trở thành hub. Số liệu **bác bỏ** điều này: event ≥ 80 giây chỉ chiếm 0.63 lần tỉ lệ slot top-1 mà chúng "đáng có". Event dài **ít được trả về hơn**, tức là chúng bị **bỏ sót**, không phải chiếm chỗ của event khác.
- Giải thích hợp lý: mỗi event chỉ được nhìn bằng **8 frame và 8 giây audio** (cắt ở giữa). Event 80 giây chứa nhiều hành động, nhưng vector chỉ thấy một phần. Caption của event dài cũng thường mô tả nhiều thao tác cùng lúc.
- Hub thực sự là **các bước chung chung, thường rất ngắn** (2 giây "add oil to a pan"). Event < 5 giây chiếm 1.17 lần (ft) và 1.40 lần (pt) tỉ lệ "đáng có".

### 3.5 Câu query ngắn khó hơn

R@1 tăng đều theo độ dài câu: 22 (1–4 từ) → 30 → 35 → 39 (≥ 13 từ). Câu ngắn như "add salt", "stir" thiếu chi tiết để phân biệt giữa hàng trăm bước giống nhau. Câu dài có thêm nguyên liệu và dụng cụ, nên dễ xác định hơn.

### 3.6 Query tự tạo: khó hơn caption, và judge còn chặt

- R@1 với câu diễn đạt lại (26.9) thấp hơn với caption (32.4), vì model được fine-tune trên văn phong caption YouCook2.
- Nhiều câu "sai" thật ra trả về đúng bước nhưng judge chấm dưới 0.8:
  - "shred potatoes…" → "grate the potatos and squeeze the starch out" (0.76)
  - "bring a pot of water to a boil" → "bring a large pan of water to boil" (0.74)
  - "dredge the chicken pieces in flour" → "dip the chicken in the milk and the flour mixture" (0.74)
  
  Ở tau 0.7, sem@1 của ft đạt 46%.
- Lưu ý: 26 câu là quá ít. Mỗi câu tương ứng khoảng 3,8 điểm phần trăm, nên chỉ có giá trị định tính. Ví dụ sem@5 ở tau 0.7 của pt (73.1) cao hơn ft (69.2): chênh lệch chỉ 1 câu.

### 3.7 Điểm cosine thấp và sát nhau

Điểm top-1 chỉ khoảng 0.12–0.21, và các hạng đầu cách nhau rất ít. Chỉ **thứ hạng** có ý nghĩa; không nên dùng điểm tuyệt đối làm ngưỡng "tìm thấy / không tìm thấy".

---

## 4. Lưu ý khi đọc kết quả

- `tIoU.5@1` trùng với `event R@1` vì các event GT trong cùng một video hầu như không chồng lên nhau tới tIoU 0.5.
- Event R@1 trên toàn database (32.38) thấp hơn trên gallery của eval (35.64) vì database có thêm 773 event dev làm nhiễu.
- Judge (`all-MiniLM-L6-v2`) chấm theo **câu chữ của caption**:
  - Có thể coi là cùng nghĩa hai bước có cùng nguyên liệu nhưng khác hành động.
  - Có thể bỏ sót các cách diễn đạt rất khác nhau.
  
  Vì vậy nên đọc `sem@k` như một **khoảng** giữa tau 0.8 (chặt) và tau 0.7 (sát thực tế), không phải một con số chính xác.
- Kết quả smoke (5 video, 49 event) **không** dùng để đánh giá.
- Mốc 35.68 lấy từ `best_val.json` trong log smoke cùng ngày; file này đã mất trên Drive trước lần chạy đầy đủ.

## 5. Ý nghĩa với UEMR và hướng đi tiếp

Phân vai trong `docs/UEMR_FINAL.md`: Omni lo "cái gì", UniAV lo "khi nào". Kết quả trên tách được ba nguồn lỗi, mỗi nguồn ứng với một phần của UEMR:

| Nguồn lỗi | Bằng chứng | Phần của UEMR giải quyết |
|---|---|---|
| Bước nấu lặp lại ở nhiều video | R@1 giảm từ 40 xuống 11 khi số near-duplicate tăng; 17–34% câu sai là đúng bước ở video khác | **Context Fusion** (RQ3): ngữ cảnh `Z` của UniAV và quan hệ giữa các đoạn |
| Event dài bị nhìn quá thưa | R@1 giảm từ 40 xuống 11 khi event dài từ 5–10s lên ≥ 80s | **Cách chia video** (biến độc lập chính): hậu xử lý proposal của UniAV, ví dụ cắt proposal dài thành nhiều đoạn; không cần train lại Omni |
| Embedding chưa đủ tốt | R@1 chỉ 40% ngay cả với bước không bị trùng lặp | Backbone Omni (cố định trong UEMR; ft đã cải thiện nhiều so với pt) |

Các bước tiếp theo đề xuất (sẽ thảo luận):
1. **Đánh giá ở mức video** (text→video, MaxSim), đúng metric chính của UEMR: R0 Global, R1 Event-Single, R7 GT, Uniform-M@K_GT. Code `SegmentIndex` đã có sẵn.
2. **Thử cắt event dài** (ví dụ > 30 giây) thành nhiều đoạn rồi dùng MaxSim, để kiểm tra giả thuyết ở mục 3.4 mà không cần train lại.
3. **Viết lại query theo văn phong caption** (ví dụ "dredge" → "coat"), để kiểm tra mục 3.6.
4. Dùng `analysis/gt_tail.csv` (146 câu có hạng > 100) để xem nhóm lỗi nặng nhất có đặc điểm chung gì.
