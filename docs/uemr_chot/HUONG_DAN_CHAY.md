# UEMR chốt: chạy gì trước, chạy gì sau

Cập nhật 2026-11-12. File này chỉ nói **thứ tự chạy**. Nội dung từng bước nằm ở `UEMR_CHOT_00` đến `04`.

## Trước khi chạy bất cứ notebook nào

1. `git push` nhánh `htstep` của `D:\Học\KL\Code\Fushion`. Notebook benchmark clone repo theo `FUSHION_REF = 12c7775` (notebook 01: `cdb2fac`); chưa push thì ô 1
   báo "không checkout được".
2. Colab Secrets có `HF_TOKEN` của tài khoản `GraceFeel` (đọc `GraceFeel/htstep-omni-clips`, ghi `GraceFeel/htstep-omni-av-best`). Với
   notebook benchmark, tài khoản đó phải đã bấm chấp nhận licence ở hai trang model `OpenGVLab/InternVideo2-Stage1-1B-224p-K400` và
   `OpenGVLab/InternVideo2-Stage2-6B-Audio`.
3. Trên Drive phải có sẵn (đều từ các lượt trước):
   - `MyDrive/omniretriever/htstep_lora/lora_bs32/eval/val_embeds_best.npz` (notebook 01 ô 6 cần file này);
   - `MyDrive/Colab Notebooks/code KL/Omni/data/YouCookII/videos` (notebook benchmark, chỉ đọc; annotation lấy từ repo, đầu ra ghi `/content` và
     đẩy HF `GraceFeel/uemr-bench-yc2`, không ghi Drive).
4. Runtime A100, đĩa `/content` trống ít nhất 70 GB.

## Thứ tự

Nhánh A và nhánh B không phụ thuộc nhau, mở hai phiên Colab chạy song song được. Trong mỗi nhánh thì đi đúng thứ tự.

| Bước | Nhánh | Chạy ở đâu | File | Cờ | Thời gian | Xong thì có gì |
|---|---|---|---|---|---|---|
| A1 | A | Colab A100 | `notebooks/uemr/uemr_chot_01_teacher_best.ipynb` | `SMOKE = True` | ~25 phút | mọi CHECK OK trên 40 video train |
| A2 | A | Colab A100 | cùng notebook | `SMOKE = False` | ~3,7 giờ (encode ~3,0 giờ) | bộ thầy `GraceFeel/htstep-omni-av-best` + tar trên Drive |
| A3 | A | Colab A100 | `notebooks/uemr/uemr_chot_01_train_uniav.ipynb` | `SMOKE = True`, rồi `False` | tải ~10 phút + train (lượt cũ ~46 phút trên RTX 3060) | `model_best.pth.tar` mới |
| B1 | B | Colab A100 | `notebooks/uemr/uemr_final_benchmark.ipynb` | `SMOKE = True` | ~30–40 phút | CHECK tái lập đặc trưng, 5 video YouCook2 |
| B2 | B | Colab A100 | cùng notebook | `SMOKE = False` | ô 6a ~1,0–1,3 giờ; trích đặc trưng 131,5 giờ video chưa đo | kho yc2 (train + val, ~1.500 video), câu hỏi, đặc trưng, vector R0 / R7 / text; mọi thứ trên HF `GraceFeel/uemr-bench-yc2` |

Sau đó (chưa viết, agent làm khi có đầu ra của các bước trên):

| Bước | Cần gì trước | Việc |
|---|---|---|
| A4 | A3 | file 01 §6: kiểm UniAV mới (local CPU) |
| C (xong 2026-11-13) | A4 (xong) | file 02: notebook `notebooks/uemr/uemr_final_dev.ipynb` (A100, `SMOKE = True` rồi `False`, ~2,4–3,2 giờ): encode clip mới; agent chấm ở local và khoá `configs/uemr_final.json` |
| D | C và B2 | file 03 §4–9: UniAV + R2 / R3 / R6 / R8 trên YouCook2, khoá protocol, chấm một lần |
| E | D | file 04: báo cáo |

## Chi tiết từng bước

**A1, A2 (vector thầy `best`).** Mở notebook, Run all. `SMOKE = True` ghi vào `uemr_chot_run/teacher_best/smoke/` và không upload. Khi dòng cuối
in "SMOKE xong: mọi CHECK OK", đổi ô 0 thành `SMOKE = False` rồi Run all lại. Mất runtime giữa chừng thì Run all lại: clip đã encode nằm
trong store trên Drive và được bỏ qua. Ô 6 là phép kiểm quyết định (clip cắt sẵn so với `val_embeds_best.npz`, cos ≥ 0.999); ô này FAIL
thì không encode gì.

**A3 (train UniAV).** Notebook `notebooks/uemr/uemr_chot_01_train_uniav.ipynb`, A100. Không cần `FUSHION_REF`: notebook chỉ clone HanLi
`77fc4fa` (không sửa) và tải bộ thầy `GraceFeel/htstep-omni-av-best` vào `datasets/omni_av_feats`, đúng đường mặc định của `config.py`, nên
lệnh train là `python train.py --output htstep_best`, không `--set` nào. Run all với `SMOKE = True` (1 epoch, ~15 phút gồm tải đặc trưng),
rồi `SMOKE = False`. Checkpoint, `train.log`, `run_record.json`, `summary.json` ghi thẳng lên Drive
`uemr_chot_run/uniav_best/htstep_best/` và upload `GraceFeel/uniav-htstep-best` (private). `train.py` không resume được: mất runtime thì Run
all lại. Loss NaN hoặc mAP val (article GT) < 20 thì ô 5 dừng (file 01 §5).

**B1, B2 (benchmark YouCook2, phần không cần UniAV).** Run all với `SMOKE = True`, rồi `SMOKE = False`. Ô 5 (trích đặc trưng) in tốc độ và
thời gian còn lại sau 5 video đầu; nếu quá lâu cho một phiên thì đặt `RUN_FEATS = False` để chạy ô 6a trước, rồi chạy lại với
`RUN_FEATS = True`, `RUN_OMNI = False` ở phiên khác (video đã trích được bỏ qua). Notebook này không tính chỉ số truy hồi nào.

## Khi nào dừng

- Ô nào in `CHECK … FAIL` thì notebook tự dừng ở đó. Không sửa ngưỡng, không chạy tiếp ô sau; gửi log cho agent.
- Không đổi `MISSING_MAX_PCT` (8.0, đã quyết định ngày 2026-11-12) và không đổi `FUSHION_REF` trừ khi agent báo commit code mới.

## Gửi lại cho agent

- A1 / A2: output của ô 3, 6, 7, 8, 9, 10 (hoặc file notebook đã lưu output).
- A3: output của ô 2, 3, 5 và `summary.json`.
- B1 / B2: output của ô 3, 4, 5, 6a, 7.
