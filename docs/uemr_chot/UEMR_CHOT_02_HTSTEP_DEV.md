# UEMR chốt, file 02: chạy UEMR trên HT-Step (dev) và khoá cấu hình cuối

Đọc `UEMR_CHOT_00_TONG_QUAN.md` mục 2, 3, 4 trước. Đầu vào: UniAV mới (file 01, `uniav_best_checks.json`), cache Omni `best` của lượt
dev 1, kho 510 video / 2.990 câu như lượt dev 1. Đầu ra: `configs/uemr_final.json` đã khoá.

Đây là **dev**: được chọn, nhưng chỉ chọn những thứ ghi ở §5 và §7, bằng đúng luật ở đó, và ghi mọi cấu hình đã chấm vào `tried.jsonl`.

Notebook mới: `notebooks/uemr/uemr_final_dev.ipynb`, ô `# ===== N.`, có `SMOKE` (5 video), resume. Dựa trên `uemr_htstep_AZ.ipynb`, bỏ môi
trường thầy cũ.

---

## 1. Chuẩn bị (local)

- Nhánh `htstep`; chuyển code chưa chạy sang `uemr_htstep/_unused/` theo file 00 §5; toàn bộ unit test pass (hoặc `skip` có lý do).
- Kiểm hash: checkpoint UniAV mới (file 01), adapter `best`, cache `a4ca43db`.
- **CHECK tái lập:** tính lại R8 (b) của lượt dev 1 từ cache: R@1 multi-positive 57.99, mảng hạng trùng `ranks_best_b_*.json`.

## 2. UniAV mới trên kho (local CPU / GPU)

Chế độ article đoán (adapter). Lưu, kèm hash:

- đề xuất đoạn + điểm (dùng cho R6 nếu luật file 01 §6 chọn UniAV mới);
- đặc trưng cần cho span head, để tính `h_i` trên đoạn bất kỳ;
- `h_i` cho mọi đoạn của mọi dòng (§4), và cho R8 với `cand` của một article **ngẫu nhiên** ≠ article đoán (seed 0) để làm phân tích C3.

Unit test: với bộ đoạn cố định, đổi `cand` chỉ đổi `h_i`, không đổi đoạn.

## 3. Tham số (luật ghi trước)

| Tham số | Luật |
|---|---|
| θ (R6) | số đoạn trung bình / video = số bước trung bình / video của train HT-Step (5.910, như lượt dev 1); lưới 0.001 |
| `L_min`, `L_fill` (R8) | chọn trên dev bằng cột UEMR-Lite (rẻ, không cần encode): lưới `L_min` ∈ {3, 5} s × `L_fill` ∈ {8, 10, 15} s, theo R@1 multi-positive của (S) trên R8; hoà thì lấy (5, 10) (kinh nghiệm YC) hoặc cặp gần nó nhất |
| L (R2) | tổng clip R2 = tổng clip R8 trên kho |

Ghi giá trị và các thống kê dùng để tính vào `tried.jsonl` và `configs/uemr_final.json`.

## 4. Các dòng

R0, R1, R2, R3, R6, R8, R7, đúng định nghĩa ở file 00 §8. R1 không cần encode. R0, R7 đã có trong cache. In số clip mới cần encode từng
dòng (theo khoá cache) và ước lượng giờ trước khi chạy.

## 5. Encode Omni `best` (Colab A100)

Clip mới của R2, R3, R6, R8: cache cũ, fp16, ghi theo lô, resume, `failed.csv`; > 2 % lỗi thì dừng. Không encode lại text (đã có).

## 6. Chấm

Mọi dòng × cột (b), (S), (U)_β, C1; strict và multi-positive; lưu hạng từng câu.

- (U)_β với β ∈ {0, 0.1, 0.2, 0.3, 0.5, 1.0}. β = 0 phải cho đúng mảng hạng của (b) (CHECK).
- **z-score không nhãn cho C1:** với cột X ∈ {(b), (S)}, μ_X và σ_X là trung bình và độ lệch chuẩn của `cos` trên **mọi cặp** (câu của kho,
  đoạn R8 của kho). `Z_X(s) = (s − μ_X) / σ_X`. Ghi μ, σ vào config. Unit test: đổi nhãn không đổi μ, σ.
- In phân bố điểm (trung bình, độ lệch) của (b) và (S).

## 7. Chọn (luật ghi trước) rồi khoá

1. **β\*** = β có R@1 multi-positive cao nhất của (U)_β trên R8; hoà thì lấy β nhỏ hơn.
2. **Phương pháp chính:** UEMR-Full ((U)_β\* trên R8) nếu β\* > 0 **và** cận dưới CI ((U)_β\* − (b)) trên R8 > 0. Không thì UEMR-Omni
   ((b) trên R8). Ghi rõ đây là lựa chọn trên dev, CI lạc quan vì β\* cũng chọn trên dev.
3. **Nguồn đoạn:** theo luật file 01 §6.
4. **Khoá** `configs/uemr_final.json`, gồm:
   - sha256 checkpoint UniAV (đoạn và `h_i`), adapter `best`, adapter pt;
   - θ (**số**, dùng nguyên cho benchmark), `L_min`, `L_fill`, luật L của R2 (tính lại trên từng benchmark, không dùng nhãn);
   - β\*, phương pháp chính, luật z-score (μ, σ **tính lại trên từng benchmark** theo cùng định nghĩa, không dùng nhãn);
   - luật chuẩn hoá text câu hỏi, định nghĩa multi-positive, chỉ số, bootstrap.

   Commit, in sha256. Sau bước này **không đổi gì nữa**.

## 8. Bảng và phân tích dev (đưa vào báo cáo, ghi là dev)

- Bảng R0, R1, R2, R3, R6, R8, R7 × (b), (S), (U)_β\*, C1 × (R@1, R@5, VCMR), multi-positive; strict ở phụ lục.
- Cặp kèm CI: R8 − R0, R6 − R0, R8 − R3, R6 − R3, R8 − R2, R6 − R1, R7 − R8, (U) − (b), (S) − R0(b), C1 − R6, VCMR R6 − R3.
- Đặt cạnh lượt dev 1 (UniAV thầy cũ) cho (b), (S), (U): chỉ tham chiếu.
- **Phân tích không tốn GPU:**
  - (i) chia câu theo bước GT bị R6 bỏ sót hay bắt được (R6 − R3, R8 − R6 từng nhóm);
  - (ii) chia theo text duy nhất hay trùng (R8 − R0, strict và multi-positive);
  - (iii) tỉ lệ đoạn đứng đầu là đoạn lấp (R8: (b), (S), (U), C1);
  - (iv) C3: (S) trên R8 với article đoán − article ngẫu nhiên, kèm CI;
  - (v) Spearman thứ hạng (b) với (S).
- **Bảng chi phí / video:** Omni s/clip (log encode); UniAV forward + span head (đo trên 20 video); trích đặc trưng K400 + BEATs (đo ở file
  03 §2, điền khi có). Số clip Omni phía video: UEMR-Omni / Full = số đoạn R8; Hybrid = số đoạn R6; Lite = 0.
- Xuất `results/uemr_final/dev/{metrics,ranks_*,pairs,groups,cost,tried}.json(l)`.

## 9. Nhật ký

(Agent ghi: CHECK từng ô, θ, `L_min`, `L_fill`, L, β\*, phương pháp chính, nguồn đoạn, sha256 `uemr_final.json`, commit, số clip, giờ chạy.)

### 2026-11-12 (agent, local): §1 mục đầu, dọn code chưa chạy

`git branch --show-current` = `htstep`. Commit `e77f59f`. Chuyển vào `uemr_htstep/_unused/` (không xoá, không notebook mới nào import):

- `_unused/cf.py`: Context Fusion cross-fitting.
- `_unused/prepare_bo_sung.py`: sinh các dòng bổ sung và fold.
- `_unused/bo_sung.py`: R3@K_GT, R4, R5, nhiễu ranh giới (R7n), R6@K, R7 + lấp, `r6_conf` (Exp 5), `article_folds`, `global_bonus` (S_G).

Giữ ở `uemr_htstep/bo_sung.py`: `zstats`, `c1z` (z-score không nhãn, UEMR-Hybrid), `missed_by`, `shared_text`, `hit_table` (phân tích
nhóm). Giữ nguyên: `score.py` (tái lập lượt dev 1), `run_uniav.py` (`student --article rand` cho C3, `feats`), R1 =
`SegmentIndex.mean_pooled`. `teacher_encode.py` **không** chuyển: nó đã chạy ở lượt dev 1 (kiểm adapter thầy cũ, text của thầy); chỉ
nhánh `--av-clips` (mục 5.5 của plan bổ sung) là chưa chạy. Notebook `uemr_final_dev.ipynb` sẽ không gọi file này.

Unit test: 37 pass trước và sau khi chuyển (`tests/`, `UNIAV_REPO` = HanLi `77fc4fa`); test của code trong `_unused/` vẫn chạy, không
`skip`. `omni_retrieval/tests/test_efficiency_qualitative.py` không import được ở máy local vì thiếu `matplotlib` (có từ trước, không
liên quan).

`docs/` đang nằm trong `.gitignore` ("Local docs (not pushed)") nên 5 file `UEMR_CHOT_*` **chưa commit** dù file 00 yêu cầu; chờ user
quyết định có `git add -f` hay không. (Cùng ngày, user quyết định: `git add -f docs/uemr_chot/`, đã commit.)

Chưa làm (cần UniAV mới của file 01): kiểm hash checkpoint mới, CHECK tái lập R8 (b) 57.99, và §2 trở đi.

### 2026-11-12 / 13 (agent, local CPU): §1 phần còn lại, §2 bắt đầu

**§1 kiểm hash:** checkpoint UniAV mới `9447de877aad65407605a338e1b946f22c140f18690aae93cf35cec5468945f0` (= log A3), adapter `best`
`5e9899aa…` (trong `config.json` của bộ thầy và meta cache dev 1), cache `a4ca43db`: `segments.npz` `16636ad17b6990b3…`, `text.npz`
`4c8a1c8f88b02b83…` (= `meta.json`).

**§1 CHECK tái lập: OK.** Tính lại R8 (b) của lượt dev 1 từ cache (`uemr_htstep.score.Main` + `run`): R@1 multi **57.99**, strict 21.81;
mảng hạng và VCMR của cả hai chế độ trùng hệt `results/uemr_htstep/a4ca43db/ranks_best_b_R8.json`.

**§2 / §3 code:** `uemr_htstep/final_dev.py` (`rows`: θ theo luật, R0, R3, R6, R7 và R8 cho lưới 2 × 3; `select`: chọn `L_min`, `L_fill`
bằng (S), L của R2, ghi `tried.jsonl`). Tự kiểm: chạy `rows` trên UniAV **cũ** cho R0, R3, R6, R7 và R8 (5, 10) trùng hệt bộ đoạn của lượt
dev 1. Test luật hoà: 43 test pass.

**Các dòng với UniAV mới** (`D:/Học/KL/output/uemr_chot/dev/segments/`, `rows.json`): θ = **0,402** (K trung bình 5,888, đích 5,910).
Số clip: R0 510, R3 3.003, R6 3.003, R7 2.990; R8 (L_min, L_fill) = (3, 8) 20.592, (3, 10) 17.167, (3, 15) 12.703, (5, 8) 20.364,
(5, 10) 16.939, (5, 15) 12.475.

**§3 chọn `L_min`, `L_fill` bằng (S)** (`final_dev select`; (S) = text Omni `best` của cache dev 1 · `h_i` của UniAV mới; ghi
`results/uemr_final/dev/tried.jsonl`):

| (L_min, L_fill) | clip R8 | R@1 multi (S) | R@1 strict (S) |
|---|---|---|---|
| (3, 8) | 20.592 | 58.09 | 21.74 |
| (3, 10) | 17.167 | 58.23 | 21.61 |
| (3, 15) | 12.703 | 57.56 | 21.34 |
| (5, 8) | 20.364 | 58.16 | 21.94 |
| **(5, 10)** | **16.939** | **58.56** | 21.64 |
| (5, 15) | 12.475 | 57.89 | 21.37 |

Chọn **(5, 10)** (cao nhất, không hoà). L của R2 = **11,89 s** (16.939 clip = R8). Bộ đoạn: `results/uemr_final/dev/segments/`
(`selected.json`, `rows.json` ghi sha256; hash git blob khớp). Commit `143e708`.

**`h_i`** (`run_uniav student`, checkpoint mới, chế độ article đoán, 0,9–1,0 s / video CPU) cho R0, R3, R6, R7, 6 biến thể R8, R2, R8 đã
chọn, và R8 với article ngẫu nhiên ≠ article đoán (seed 0, C3); đầu ra local `D:/Học/KL/output/uemr_chot/dev/student/`.

**§4 số clip mới cần encode** (khoá `video__ts_te` so với cache dev 1): 36.881 clip khác nhau của R2, R3, R6, R8, đã có 762, **mới 36.119**
(R2 16.939 / 16.939 vì L đổi 11,91 → 11,89 s; R3 2.276 / 3.003; R6 3.001 / 3.003; R8 16.904 / 16.939). Ước lượng 2,4 giờ (0,236 s/clip của A2)
đến 3,2 giờ (0,32 s/clip). R0, R7 đã có trong cache.

**§5 notebook `notebooks/uemr/uemr_final_dev.ipynb`** (commit `1cb2031`, `FUSHION_REF = 143e708`): chỉ encode clip mới, đường cột (b) của
lượt dev 1 (mp4 + `timestamps`, loader vendored), store trên Drive `uemr_chot_run/dev/best/clips`, upload `GraceFeel/encode_omni`
`uemr_final/dev/best/clips/`, CHECK lỗi ≤ 2 %. **Khác plan (ghi để user biết):** §6–8 (chấm, chọn β\*, khoá `configs/uemr_final.json`)
chạy ở local bằng `final_dev score` (numpy, không GPU) thay vì trong notebook, để không phải đưa vector `h_i` (~0,5 GB) lên Colab.

**Tự kiểm `final_dev score`** trên dữ liệu lượt dev 1 (đoạn cũ, cache cũ): cột (b) của R0, R2, R3, R6, R7, R8 × strict / multi khớp
`metrics_best.json` (12 / 12 hàng, sai khác 0); CHECK β = 0 = (b) OK.

### 2026-11-13 (user chạy Colab §5, agent chấm local §6–8)

**§5 encode** (`notebooks/uemr/uemr_final_dev.ipynb`, A100-SXM4-40GB, log `notebooks/uemr/log/uemr_final_dev_smoke_{true,false}.pdf`).
SMOKE = True và SMOKE = False: mọi CHECK OK (protocol dev 1, sha256 adapter `5e9899aa…`, sha256 `segments.npz` cache dev 1, sha256
`r2/r8.json` = `selected.json`, `r3/r6.json` = `rows.json`, kho 510 video, mp4 đọc được bằng decord, thời lượng mp4 = protocol, GPU, môi
trường encode transformers 4.51.3, WAVE-7B + BEATs). Clip khác nhau 36.881, có trong cache 762, **mới 36.119** (theo dòng: R2 16.939, R3
2.276, R6 3.001, R8 16.904). Lỗi **0 clip (0,00 %)**, CHECK ≤ 2 % OK. Encode resume: một lượt trước đã ghi 11.620 clip vào store Drive,
lượt cuối encode 20 + 24.479 clip (1 giờ 55 phút, 0,231–0,246 s/clip). Store 183 chunk, upload `GraceFeel/encode_omni`
`uemr_final/dev/best/clips/` (505 MB); agent tải về, đếm 36.119 khoá = số clip mới.

**§6 chấm** (`final_dev score`, local CPU, store vừa tải + cache dev 1 cho text, R0, R7 và 762 clip trùng). Clip thiếu vector Omni: 0 ở mọi
dòng. Phân bố điểm trên mọi cặp (câu, đoạn R8): (b) μ 0,0530, σ 0,0374; (S) μ 0,0565, σ 0,0403 (hai cột cùng thang vì (S) giờ dùng
text Omni `best`, không còn text của thầy cũ như lượt dev 1, nơi (S) có μ ≈ 0,60). **CHECK β = 0 cho đúng mảng hạng (b): OK.**

β → R@1 multi (U)_β trên R8: 0 → 59,60; 0,1 → 61,44; 0,2 → 61,30; 0,3 → 62,44; 0,5 → 63,21; 1,0 → **63,34**.

**§7 chọn và khoá** (luật ghi trước, áp nguyên):
- **β\* = 1,0.** Lưu ý: β\* nằm ở biên trên của lưới; R@1 gần như phẳng từ 0,5 (63,21) đến 1,0 (63,34). Không mở rộng lưới vì plan cố
  định lưới. β = 1 là cộng đều (b) + (S), hợp lý khi hai cột cùng thang.
- (U)_1 − (b) trên R8, multi R@1: **+3,75 [+1,53, +5,98]** (cận dưới > 0) → **phương pháp chính: UEMR-Full** ((b) + 1,0 · (S) trên R8).
  Đây là lựa chọn trên dev; CI lạc quan vì β\* cũng chọn trên dev. Strict: +1,84 [−0,31, +4,11] (chứa 0).
- Nguồn đoạn R6 và `h_i`: UniAV mới (luật file 01 §6, mAP article đoán 20,14 > 19,55).
- **`configs/uemr_final.json` khoá**, sha256 **`c9840ffbc964a4b0ed70baee157213adaaa6526178316c80a8435e7d2fcdbde3`**: UniAV sha256 `9447de87…`, adapter `best` `5e9899aa…`, adapter pt `31e81f1c…`,
  θ 0,402, R6 hậu xử lý (k_max 30, d_min 2,0, merge tIoU 0,7), L_min 5, L_fill 10, β\* 1,0, luật L của R2 (`pick_chunk`), luật z-score
  (μ, σ tính lại trên từng kho, giá trị dev ghi kèm), chuẩn hoá text, multi-positive, chỉ số, bootstrap (video, 1000, seed 0, 95 %).

**§8 bảng dev, multi-positive** (R@1 / R@5 / VCMR R@1; vec/video; đầy đủ ở `results/uemr_final/dev/metrics.json`, strict cùng file):

| Dòng | vec/video | (b) | (S) | (U) β = 1 | C1 |
|---|---|---|---|---|---|
| R0 | 1,0 | 49,00 / 80,84 / – | 50,87 / 84,52 / – | 55,08 / 88,09 / – | |
| R1 | 1,0 | 58,13 / 86,69 / – | 55,05 / 86,66 / – | 58,86 / 87,63 / – | |
| R2 | 33,2 | 57,83 / 89,16 / 10,60 | 56,52 / 88,09 / 8,70 | 62,04 / 90,90 / 11,77 | |
| R3 | 5,9 | 53,38 / 87,83 / 4,41 | 55,89 / 87,39 / 5,72 | 57,99 / 89,03 / 6,42 | |
| R6 | 5,9 | 57,49 / 87,36 / 24,48 | 58,39 / 85,95 / 22,07 | 61,57 / 88,46 / 26,45 | |
| **R8** | 33,2 | 59,60 / 90,07 / 15,92 | 58,56 / 88,19 / 15,05 | **63,34 / 90,90 / 21,34** | 58,90 / 89,77 / 15,75 |
| R7 (GT, trần) | 5,9 | 65,05 / 92,07 / 56,05 | 66,62 / 90,60 / 55,52 | 70,54 / 93,88 / 59,63 | |

Strict R@1 (phụ lục): R8 (b) 22,51, (S) 21,64, (U) 24,35, C1 21,94; R0 (b) 18,29; R2 (b) 22,44; R6 (b) 22,47; R7 (b) 26,12.

Cặp kèm CI (multi R@1, cột (b) trừ khi ghi khác; `pairs.json`):

| Cặp | A − B | CI 95 % |
|---|---|---|
| R8 − R0 | +10,60 | [+7,24, +14,08] |
| R6 − R0 | +8,49 | [+4,90, +11,90] |
| R8 − R3 | +6,22 | [+3,74, +8,73] |
| R6 − R3 | +4,11 | [+1,19, +6,91] |
| R8 − R2 | +1,77 | [−0,53, +4,02] |
| R6 − R1 | −0,64 | [−3,40, +2,24] |
| R7 − R8 | +5,45 | [+2,75, +7,95] |
| (U) − (b) trên R8 | +3,75 | [+1,53, +5,98] |
| (S) R8 − (b) R0 | +9,57 | [+6,13, +13,08] |
| C1 R8 − (b) R6 | +1,40 | [−0,78, +3,70] |
| VCMR R6 − R3 | +20,07 | [+17,55, +22,59] |

Đặt cạnh lượt dev 1 (UniAV thầy cũ, chỉ tham chiếu; multi R@1): (b) R8 57,99 → 59,60, R6 55,35 → 57,49, R3 55,69 → 53,38, R2 58,16 →
57,83; (S) R8 57,56 → 58,56; (U) R8 61,34 (β = 0,1, text thầy cũ) → 63,34 (β = 1, text `best`). Hai lượt khác nhau cả đoạn lẫn text của
(S), nên không so trực tiếp.

Phân tích không tốn GPU (`groups.json`, multi trừ khi ghi khác):
- (i) Bước bị R6 bỏ sót (tIoU < 0,3; 1.687 câu): R6 − R3 +1,48 [−1,75, +4,88], R8 − R6 **+5,69 [+2,36, +8,84]**. Bước R6 bắt được (1.303
  câu): R6 − R3 **+7,52 [+3,87, +11,14]**, R8 − R6 −2,53 [−5,81, +0,80]. Đoạn lấp giúp đúng ở chỗ R6 bỏ sót.
- (ii) Text trùng ở ≥ 2 video (2.488 câu): R8 − R0 strict +4,30 [+0,84, +7,76], multi +11,98 [+8,36, +15,77]. Text duy nhất (502 câu):
  +3,78 [−3,04, +10,30] (strict = multi).
- (iii) Tỉ lệ đoạn đứng đầu là đoạn lấp (R8): (b) 65,3 %, (S) 56,6 %, (U) 52,3 %, C1 51,7 %.
- (iv) C3: (S) R8 article đoán − article ngẫu nhiên: **+22,58 [+19,71, +25,57]** ((S) với article ngẫu nhiên 35,99): `h_i` phụ thuộc
  mạnh vào article.
- (v) Spearman hạng (b) với (S) trên R8: 0,587. hit@1: cả hai đúng 1.323, chỉ (b) đúng 459, chỉ (S) đúng 428, cả hai sai 780 → hai cột
  sai ở chỗ khác nhau, giải thích vì sao (U) hơn từng cột.

Chi phí phía video (`cost.json`): clip Omni / video: UEMR-Omni / Full 33,2 (16.939 clip R8), UEMR-Hybrid 5,9 (3.003 đoạn R6), UEMR-Lite 0.
Omni 0,231–0,246 s/clip (A100, log encode). UniAV forward + span head 0,9–1,0 s/video (CPU local, lúc tính `h_i`, §2). Trích K400 + BEATs:
chưa có (file 03 §2).

**Kết luận dev (chỉ để chọn):** event-aware R8 hơn rõ R0, R3; hơn R2 (chia đều cùng số clip) +1,77 nhưng CI chứa 0, nên trên dev chưa
tách được lợi ích của ranh giới UniAV khỏi việc có nhiều clip hơn. Cộng (S) giúp ở mọi dòng. Benchmark YouCook2 (file 03) dùng nguyên
`configs/uemr_final.json`.

