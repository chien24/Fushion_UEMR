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
