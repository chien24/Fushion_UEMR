# UEMR chốt, file 00: tổng quan (đọc file này TRƯỚC mọi việc)

Người giao việc: GraceFeel. Ngày 2026-11-11. Đây là **kế hoạch cuối**. Không thêm thí nghiệm ngoài các file 00–04.

Repo `D:\Học\KL\Code\Fushion`, **nhánh `htstep`**. Bước đầu mỗi phiên: `git branch --show-current` phải in `htstep`, nếu không thì dừng,
báo. Được `git add`, `git commit`; **không `git push`** (user tự push trước khi chạy Colab). Chép 5 file này vào `docs/uemr_chot/` và commit.

## Bộ file

| File | Nội dung | Ai chạy |
|---|---|---|
| `UEMR_CHOT_00_TONG_QUAN.md` | phương pháp cuối, quyết định đã khoá, nguyên tắc, lộ trình | đọc đầu mỗi phiên |
| `UEMR_CHOT_01_UNIAV_BEST.md` | vector thầy Omni `best`, train lại UniAV (chỉ đổi thầy) | agent chuẩn bị dữ liệu; user train |
| `UEMR_CHOT_02_HTSTEP_DEV.md` | chạy UEMR trên HT-Step (dev), chọn tham số, **khoá cấu hình cuối** | agent + Colab |
| `UEMR_CHOT_03_BENCHMARK.md` | YouCook2 (+ bộ khác) zero-shot, chấm một lần | agent + Colab |
| `UEMR_CHOT_04_BAO_CAO.md` | khung báo cáo, đóng góp, hạn chế | agent viết, local |

**Cách làm việc để không quên:** trước mỗi giai đoạn đọc lại mục 3 (quyết định đã khoá) và mục 4 (nguyên tắc) của file này, rồi đọc file
của giai đoạn. Nhật ký ghi ở cuối file của giai đoạn đó (bản trong `docs/uemr_chot/`), không ghi lẫn sang file khác.

---

## 1. Mục tiêu và vai trò dữ liệu

| Dữ liệu | Vai trò |
|---|---|
| HT-Step train | train Omni (đã xong: `best`) và UniAV (train lại ở file 01) |
| HT-Step `val_seen`, kho 510 video, 2.990 câu | **dev**: được chọn tham số, chọn biến thể. Số trên dev **không** là kết quả kiểm định |
| YouCook2 val (+ bộ khác nếu user bật) | **benchmark cuối**: không train, không chọn gì; khoá protocol trước, **chấm một lần** |
| Kết quả YouCook2 cũ (Athena, Omni ft YC2) | chỉ là kinh nghiệm. **Không dùng số, không so số** ở bất kỳ đâu |

## 2. Phương pháp cuối: UEMR (UEMR và UEMR 2 hợp làm một)

**UEMR = Unified Event-aware Multi-vector Retrieval.**

- *Unified*: UniAV được chưng cất từ chính Omni dùng để truy hồi (`best`), nên mọi vector nằm trong **một** không gian.
- *Event-aware*: video được cắt theo bước bởi UniAV.
- *Multi-vector*: một vector cho mỗi đoạn, chấm bằng MaxSim.

Context Fusion **bị bỏ**. Ngữ cảnh đi vào qua vector trò: backbone UniAV nhìn cả video và `article_cond` mang thông tin task.

**Chỉ mục của một video.** Đoạn {S_i} lấy từ UniAV (R6) cộng đoạn lấp chỗ trống (R8). Mỗi đoạn có:

- `e_i` = Omni `best` AV của đoạn (8 khung, tiếng 8 s, `omni_retrieval.encode`): chính xác, tốn Omni 7B;
- `h_i` = vector span head của UniAV mới trên đoạn đó, cùng không gian `best`: rất rẻ, có ngữ cảnh video và task.

**Câu hỏi:** `t` = Omni `best` `encode_text`.

**Điểm** (với `Z(·)` là chuẩn hoá z-score không dùng nhãn, định nghĩa ở file 02 §6):

| Biến thể | Điểm của đoạn | Phía video có chạy Omni 7B? |
|---|---|---|
| UEMR-Omni = cột (b) | `cos(t, e_i)` | có, mọi đoạn |
| UEMR-Lite = cột (S) | `cos(t, h_i)` | **không** (Omni 7B chỉ mã hoá câu hỏi) |
| UEMR-Full = cột (U) | `cos(t, e_i) + β·cos(t, h_i)`, β chọn trên dev | có |
| UEMR-Hybrid = C1 | đoạn R6: `Z_b(cos(t, e_i))`; đoạn lấp: `Z_S(cos(t, h_i))` | chỉ trên đoạn R6 |

`S(q, V) = max_i s_i`. Khoảnh khắc trả về = đoạn `argmax_i s_i`.

**Phương pháp chính** (ghi trong `configs/uemr_final.json`) chọn bằng luật ở file 02 §7: UEMR-Full nếu thắng UEMR-Omni trên dev, không thì
UEMR-Omni. UEMR-Lite và UEMR-Hybrid luôn được báo cáo.

## 3. Quyết định đã khoá (không bàn lại)

1. Omni: `best` = LoRA HT-Step (`GraceFeel/omni_lora_htstep`, adapter sha256 `5e9899aa…`). Không train lại. Omni pt (`omniretriever-7b`)
   chỉ là baseline trên benchmark.
2. UniAV: train lại **một lần**, chỉ đổi thầy sang `best` (file 01). Mọi thứ khác giữ nguyên.
3. Không Context Fusion. Không bộ chuyển ngữ P_φ.
4. Dòng (cách cắt): R0, R1, R2, R3, R6, R8, R7. Không R4, R5, nhiễu ranh giới, Exp 5, Omni pt trên dev.
5. Chấm: MaxSim, vector fp16, multi-positive (đáp án = mọi video có bước cùng text đã chuẩn hoá) là chính, strict báo kèm. R@1, R@5,
   R@10, MdR, VCMR R@1 (đúng video ở hạng 1 và đoạn đứng đầu có tIoU ≥ 0.5 với bước của câu). CI 95 %: paired bootstrap theo video,
   1.000 lần, seed 0.
6. Kết luận chỉ viết từ benchmark (file 03). Dev chỉ dùng để chọn.

## 4. Nguyên tắc

1. Không sửa code Omni. Code UniAV (HanLi) được sửa **chỉ** ở chỗ trỏ thư mục vector thầy (file 01); pin commit.
2. Không tự điền số; mọi số có nguồn ô notebook hoặc file kết quả.
3. Mỗi ô notebook in `CHECK … OK/FAIL`; có `FAIL` thì dừng, báo.
4. Mọi cache ghi kèm sha256; bước sau kiểm hash bước trước.
5. Trên dev: ghi **mọi cấu hình đã chấm** vào `results/uemr_final/dev/tried.jsonl` (cấu hình, số, thời điểm).
6. Trên benchmark: `configs/benchmark_protocol.json` commit và in sha256 trước khi chấm; ô chấm kiểm hash; chấm một lần.
7. Tham số lấy từ kinh nghiệm YC ghi rõ nguồn. Không so số với YC.

## 5. Hiện trạng: dùng lại gì, bỏ gì

**Dùng lại:**

- Adapter `uemr_htstep/uniav_htstep.py`: tránh rò rỉ article bằng cách dùng article top-1 do model đoán. Bắt buộc dùng cho mọi lượt UniAV.
- Chọn kho, kiểm giao với train, multi-positive, VCMR, bootstrap, đổi giây ↔ lưới, notebook `uemr_htstep_AZ.ipynb`.
- Cache 41.228 clip Omni `best` và text của 2.990 câu (`GraceFeel/encode_omni`, `htstep_uemr/a4ca43db/`), khoá `video_id|start|end`.
- Kết quả lượt dev 1 (UniAV thầy cũ, protocol `a4ca43db`, báo cáo `Result/uemr/HT_Steps/KET_QUA_UEMR_HTSTEP.md`): chỉ là tham chiếu.

**Code bổ sung đã viết nhưng chưa chạy** (theo plan `HTSTEP_UEMR_BO_SUNG.md`, đã bị thay thế):

- Giữ và dùng: z-score không nhãn (C1z → UEMR-Hybrid), vector trò với article ngẫu nhiên (C3), R1, các phân tích nhóm (bước bắt được /
  bỏ sót, text duy nhất / trùng, đoạn đứng đầu là đoạn lấp), đo tốc độ.
- Không chạy, **không xoá**: Context Fusion cross-fitting, R4, R5, R3@K_GT, nhiễu ranh giới, Exp 5, môi trường thầy cũ (mục 5.5). Chuyển
  vào `uemr_htstep/_unused/`, bỏ import khỏi notebook mới. Test của chúng vẫn phải pass hoặc đánh `skip` kèm lý do.

## 6. Lộ trình (song song để tiết kiệm thời gian)

```text
Nhánh A (UniAV)                           Nhánh B (benchmark, không phụ thuộc UniAV)
01 §1-4  vector thầy best (~3 giờ A100)   03 §1-3  kiểm kê, kiểm trùng, trích đặc trưng, câu hỏi
01 §5    user train UniAV                 03 §6a   encode R0, R7, text (best + pt)
01 §6    kiểm UniAV mới
02       dev HT-Step (~2,5 giờ A100)  →  KHOÁ configs/uemr_final.json
                                          03 §4-6b UniAV + đoạn R2, R3, R6, R8 + encode
                                          03 §7-8  khoá protocol benchmark → chấm một lần
04       báo cáo
```

Ước lượng GPU (0.32 s/clip theo lượt dev 1): file 01 ~3 giờ + train UniAV; file 02 ~2,5 giờ; YouCook2 ~3,5 giờ (gồm pt). Agent in số clip
thật trước mỗi lượt encode.

## 7. Đóng góp (agent không được viết mạnh hơn mức này)

1. **UEMR:** chỉ mục đa vector theo sự kiện, gồm cắt bằng UniAV, lấp chỗ trống, MaxSim. Một chỉ mục cho cả truy hồi video lẫn định vị
   khoảnh khắc.
2. **Bộ định vị sự kiện chưng cất từ Omni truy hồi:** một lượt UniAV cho cả đoạn lẫn vector đoạn trong không gian Omni, có ngữ cảnh video
   và task. Từ đó có UEMR-Lite (không chạy Omni 7B phía video) và UEMR-Full. Kiến trúc span head và loss chưng cất là của code nhóm (HanLi).
   Đóng góp nằm ở việc dùng chính Omni truy hồi làm thầy, tích hợp vào truy hồi và đánh giá. **Không** claim distillation là mới.
3. **Giao thức đánh giá:** suy luận không rò rỉ article cho bộ định vị có điều kiện task, chấm multi-positive, dev / benchmark tách bạch,
   zero-shot.
4. **Nghiên cứu có kiểm soát** trên dev (cách cắt, cột) và **đánh giá zero-shot** trên benchmark.

Câu nào là "có hiệu quả" chỉ viết khi tiêu chí benchmark (file 03 §8) đạt.

## 8. Thuật ngữ

| Mã | Đoạn |
|---|---|
| R0 | cả video, 1 vector |
| R1 | trung bình (rồi L2-norm) các vector R6 của video, 1 vector |
| R2 | đoạn đều dài L, L chọn để tổng số clip bằng R8 |
| R3 | K_pred đoạn đều (K_pred = số đoạn R6 của video) |
| R6 | đoạn UniAV, article do model đoán, ngưỡng θ |
| R8 | R6 + lấp chỗ trống (`L_min`, `L_fill`) |
| R7 | đoạn GT (oracle) |
