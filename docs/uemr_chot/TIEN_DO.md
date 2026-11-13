# UEMR chốt: tiến độ (để phiên sau làm tiếp)

Cập nhật 2026-11-13. Nhánh `htstep` của Fushion; xem `git log` cho commit mới nhất (chưa push). Chi tiết từng bước nằm ở
Nhật ký cuối file 01 và 02; số liệu cho báo cáo ở `SO_LIEU_DU_LIEU.md`; thứ tự chạy ở `HUONG_DAN_CHAY.md`.

## Đã xong

| Bước | Kết quả | Ở đâu |
|---|---|---|
| Dọn code chưa chạy (file 00 §5) | `uemr_htstep/_unused/`, 42 test pass | commit `e77f59f` |
| File 01 §1–4: vector thầy Omni `best` (A1, A2) | `GraceFeel/htstep-omni-av-best`: 5.431 train + 510 val video, thiếu 7,16 % (ngưỡng 8 %, user quyết định) | Nhật ký file 01 |
| File 01 §5: train UniAV (A3) | ckpt sha256 `9447de87…`, epoch 9, mAP val (article GT) 30.29; `GraceFeel/uniav-htstep-best` | Nhật ký file 01 |
| File 01 §6: kiểm UniAV mới (local CPU) | mAP article đoán 20.14 > 19.55 (cũ) → **đoạn R6 lấy từ UniAV mới**; recall@0.5 33.04; θ 0,402 | `results/uemr_final/uniav_best_checks.json` |
| File 02 §1: CHECK tái lập R8 (b) lượt dev 1 từ cache | R@1 multi 57.99, strict 21.81, mảng hạng và VCMR trùng hệt: **OK** | Nhật ký file 02 |
| File 02 §5–8: encode 36.119 clip (Colab), chấm, khoá config | β\* 1,0, **UEMR-Full**, (U) − (b) R8 +3,75 [+1,53, +5,98]; `configs/uemr_final.json` sha256 `c9840ffb…` | `results/uemr_final/dev/`, Nhật ký file 02 |
| File 02 §2–4 (local CPU) | θ 0,402; (L_min, L_fill) = (5, 10) theo (S) (R@1 multi 58,56); L R2 11,89 s; `h_i` mọi dòng + C3; 36.119 clip mới | `results/uemr_final/dev/`, Nhật ký file 02 |

Đầu ra local của §6: `D:/Học/KL/output/uemr_chot/uniav_best/` (`uniav_raw.jsonl` sha256 `5db3e8cb…`, `checks.json`, `omni_checks.json`,
`predictions_*.csv`); checkpoint và bộ thầy đã tải ở `D:/Học/KL/output/uemr_chot/{uniav_best_model,omni_av_best}`.

## Việc tiếp theo (theo thứ tự)

1. Nhánh B (user chạy, sau khi push): `notebooks/uemr/uemr_final_benchmark.ipynb` B1 / B2, bản sửa 2026-11-13 (`FUSHION_REF = 12c7775`: kho =
   mọi video YouCook2 có mp4, ~1.500; annotation từ repo; đầu ra `/content` + HF `GraceFeel/uemr-bench-yc2`). Chưa có log. Gửi output ô 3, 4, 5, 6a, 7.
2. Agent: file 03 §4–9 (UniAV + R2 / R3 / R6 / R8 trên YouCook2 theo `configs/uemr_final.json`, khoá protocol, chấm một lần). Notebook mới
   phải ghi vào `/content` và đẩy HF, không ghi Drive (Drive của user gần đầy).
3. File 04: báo cáo.

## Lưu ý cho phiên sau

- Chạy module Fushion bằng `PYTHONPATH=. D:/miniconda/conda-envs/omni/python.exe -X utf8 -m …` (không dùng `-I`, nếu không sẽ không import được
  `uemr_htstep`); test: thêm `UNIAV_REPO=D:/Học/KL/Code/UniAV/UniAV-fixed`.
- Token HF mới nằm ở `D:/Học/KL/Code/pipeline/.env` (`HF_TOKEN`, tài khoản `GraceFeel`); token trong `~/.cache/huggingface` đã hết hạn.
- Log Colab là PDF trong `notebooks/uemr/log/`; đọc bằng `pdftotext -layout` (`/mingw64/bin`).
- `notebooks/uemr/uemr_htstep_AZ.ipynb` có thay đổi chưa commit (output của lượt dev 1 do user lưu): không đụng.
- Vector `h_i` của dev ở `D:/Học/KL/output/uemr_chot/dev/student/` (không commit, *.npz bị gitignore): `final_dev score` cần thư mục này.
- Store Omni clip mới của dev đã tải về `D:/Học/KL/output/uemr_chot/hf_encode_omni/uemr_final/dev/best/clips/` (505 MB).
