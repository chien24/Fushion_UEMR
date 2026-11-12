# UEMR chốt: chạy gì trước, chạy gì sau

Cập nhật 2026-11-12. File này chỉ nói **thứ tự chạy**. Nội dung từng bước nằm ở `UEMR_CHOT_00` đến `04`.

## Trước khi chạy bất cứ notebook nào

1. `git push` nhánh `htstep` của `D:\Học\KL\Code\Fushion`. Hai notebook bên dưới clone repo theo `FUSHION_REF = cdb2fac`; chưa push thì ô 1
   báo "không checkout được".
2. Colab Secrets có `HF_TOKEN` của tài khoản `GraceFeel` (đọc `GraceFeel/htstep-omni-clips`, ghi `GraceFeel/htstep-omni-av-best`). Với
   notebook benchmark, tài khoản đó phải đã bấm chấp nhận licence ở hai trang model `OpenGVLab/InternVideo2-Stage1-1B-224p-K400` và
   `OpenGVLab/InternVideo2-Stage2-6B-Audio`.
3. Trên Drive phải có sẵn (đều từ các lượt trước):
   - `MyDrive/omniretriever/htstep_lora/lora_bs32/eval/val_embeds_best.npz` (notebook 01 ô 6 cần file này);
   - `MyDrive/Colab Notebooks/code KL/Omni/data/YouCookII/{metadata,videos}` (notebook benchmark).
4. Runtime A100, đĩa `/content` trống ít nhất 70 GB.

## Thứ tự

Nhánh A và nhánh B không phụ thuộc nhau, mở hai phiên Colab chạy song song được. Trong mỗi nhánh thì đi đúng thứ tự.

| Bước | Nhánh | Chạy ở đâu | File | Cờ | Thời gian | Xong thì có gì |
|---|---|---|---|---|---|---|
| A1 | A | Colab A100 | `notebooks/uemr/uemr_chot_01_teacher_best.ipynb` | `SMOKE = True` | ~25 phút | mọi CHECK OK trên 40 video train |
| A2 | A | Colab A100 | cùng notebook | `SMOKE = False` | ~3,7 giờ (encode ~3,0 giờ) | bộ thầy `GraceFeel/htstep-omni-av-best` + tar trên Drive |
| A3 | A | máy / Colab của bạn | HanLi `77fc4fa`, `train.py` | xem bên dưới | lượt cũ ~46 phút (RTX 3060) | `model_best.pth.tar` mới |
| B1 | B | Colab A100 | `notebooks/uemr/uemr_final_benchmark.ipynb` | `SMOKE = True` | ~30–40 phút | CHECK tái lập đặc trưng, 5 video YouCook2 |
| B2 | B | Colab A100 | cùng notebook | `SMOKE = False` | ô 6a ~0,5 giờ; trích đặc trưng chưa đo | kho yc2, câu hỏi, đặc trưng, vector R0 / R7 / text |

Sau đó (chưa viết, agent làm khi có đầu ra của các bước trên):

| Bước | Cần gì trước | Việc |
|---|---|---|
| A4 | A3 | file 01 §6: kiểm UniAV mới (local CPU) |
| C | A4 | file 02: notebook `uemr_final_dev.ipynb`, dev HT-Step, khoá `configs/uemr_final.json` |
| D | C và B2 | file 03 §4–9: UniAV + R2 / R3 / R6 / R8 trên YouCook2, khoá protocol, chấm một lần |
| E | D | file 04: báo cáo |

## Chi tiết từng bước

**A1, A2 (vector thầy `best`).** Mở notebook, Run all. `SMOKE = True` ghi vào `uemr_chot_run/teacher_best/smoke/` và không upload. Khi dòng cuối
in "SMOKE xong: mọi CHECK OK", đổi ô 0 thành `SMOKE = False` rồi Run all lại. Mất runtime giữa chừng thì Run all lại: clip đã encode nằm
trong store trên Drive và được bỏ qua. Ô 6 là phép kiểm quyết định (clip cắt sẵn so với `val_embeds_best.npz`, cos ≥ 0.999); ô này FAIL
thì không encode gì.

**A3 (train UniAV, bạn chạy).** Code HanLi `77fc4fa`, không sửa. Chuẩn bị dữ liệu như lượt cũ, riêng vector thầy lấy từ bộ mới:

```bash
python tools/fetch_htstep.py --what k400          # đặc trưng như cũ; KHÔNG tải thầy cũ
hf download GraceFeel/htstep-omni-av-best --repo-type dataset --local-dir datasets/omni_av_best
python train.py --output htstep_best --set "dataset.omni_emb_folder='./datasets/omni_av_best'"
```

Giữ lại: `model_best.pth.tar` và sha256, log train, commit HanLi. Không thử cấu hình khác. Loss NaN hoặc mAP val (article GT) < 20 thì
dừng, báo (file 01 §5).

**B1, B2 (benchmark YouCook2, phần không cần UniAV).** Run all với `SMOKE = True`, rồi `SMOKE = False`. Ô 5 (trích đặc trưng) in tốc độ và
thời gian còn lại sau 5 video đầu; nếu quá lâu cho một phiên thì đặt `RUN_FEATS = False` để chạy ô 6a trước, rồi chạy lại với
`RUN_FEATS = True`, `RUN_OMNI = False` ở phiên khác (video đã trích được bỏ qua). Notebook này không tính chỉ số truy hồi nào.

## Khi nào dừng

- Ô nào in `CHECK … FAIL` thì notebook tự dừng ở đó. Không sửa ngưỡng, không chạy tiếp ô sau; gửi log cho agent.
- Không đổi `MISSING_MAX_PCT` (8.0, đã quyết định ngày 2026-11-12) và không đổi `FUSHION_REF` trừ khi agent báo commit code mới.

## Gửi lại cho agent

- A1 / A2: output của ô 3, 6, 7, 8, 9, 10 (hoặc file notebook đã lưu output).
- A3: sha256 checkpoint, log train, lệnh đã chạy.
- B1 / B2: output của ô 3, 4, 5, 6a, 7.
