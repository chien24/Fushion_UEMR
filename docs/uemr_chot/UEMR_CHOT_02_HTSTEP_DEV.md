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

