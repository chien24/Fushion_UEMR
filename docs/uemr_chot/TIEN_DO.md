# UEMR chốt: tiến độ (để phiên sau làm tiếp)

Cập nhật 2026-11-12 (tối). Nhánh `htstep` của Fushion, commit mới nhất `1cb2031` (chưa push). Chi tiết từng bước nằm ở
Nhật ký cuối file 01 và 02; số liệu cho báo cáo ở `SO_LIEU_DU_LIEU.md`; thứ tự chạy ở `HUONG_DAN_CHAY.md`.

## Đã xong

| Bước | Kết quả | Ở đâu |
|---|---|---|
| Dọn code chưa chạy (file 00 §5) | `uemr_htstep/_unused/`, 42 test pass | commit `e77f59f` |
| File 01 §1–4: vector thầy Omni `best` (A1, A2) | `GraceFeel/htstep-omni-av-best`: 5.431 train + 510 val video, thiếu 7,16 % (ngưỡng 8 %, user quyết định) | Nhật ký file 01 |
| File 01 §5: train UniAV (A3) | ckpt sha256 `9447de87…`, epoch 9, mAP val (article GT) 30.29; `GraceFeel/uniav-htstep-best` | Nhật ký file 01 |
| File 01 §6: kiểm UniAV mới (local CPU) | mAP article đoán 20.14 > 19.55 (cũ) → **đoạn R6 lấy từ UniAV mới**; recall@0.5 33.04; θ 0,402 | `results/uemr_final/uniav_best_checks.json` |
| File 02 §1: CHECK tái lập R8 (b) lượt dev 1 từ cache | R@1 multi 57.99, strict 21.81, mảng hạng và VCMR trùng hệt: **OK** | Nhật ký file 02 |
| File 02 §2–4 (local CPU) | θ 0,402; (L_min, L_fill) = (5, 10) theo (S) (R@1 multi 58,56); L R2 11,89 s; `h_i` mọi dòng + C3; 36.119 clip mới | `results/uemr_final/dev/`, Nhật ký file 02 |

Đầu ra local của §6: `D:/Học/KL/output/uemr_chot/uniav_best/` (`uniav_raw.jsonl` sha256 `5db3e8cb…`, `checks.json`, `omni_checks.json`,
`predictions_*.csv`); checkpoint và bộ thầy đã tải ở `D:/Học/KL/output/uemr_chot/{uniav_best_model,omni_av_best}`.

## Việc tiếp theo (theo thứ tự)

1. **User chạy `notebooks/uemr/uemr_final_dev.ipynb`** (A100): `SMOKE = True` rồi `False`. Encode 36.119 clip mới, ~2,4–3,2 giờ.
   Upload `GraceFeel/encode_omni` `uemr_final/dev/best/clips/`.
2. **Agent chấm ở local** (file 02 §6–8): tải store đó về, rồi
   ```
   PYTHONPATH=. python -X utf8 -m uemr_htstep.final_dev score --segments results/uemr_final/dev/segments \
     --student D:/Học/KL/output/uemr_chot/dev/student --text <cache dev 1 db> --omni-db <cache dev 1 db> --omni-store <store mới> \
     --val-manifest D:/Học/KL/Code/Omni/precut/meta/val_omni_htstep.jsonl --uniav-checks results/uemr_final/uniav_best_checks.json \
     --out results/uemr_final/dev --config configs/uemr_final.json
   ```
   (`<cache dev 1 db>` = `D:/Học/KL/output/htstep_uemr/encode_omni/htstep_uemr/a4ca43db/best/db`). Đọc kết quả, ghi Nhật ký file 02, commit
   `configs/uemr_final.json` (khoá), in sha256.
3. Nhánh B (song song, user chạy): `notebooks/uemr/uemr_final_benchmark.ipynb` B1 / B2 (YouCook2), chưa có log.
4. Sau đó: file 03 §4–9 (benchmark, chấm một lần), file 04 (báo cáo).

## Lưu ý cho phiên sau

- Chạy module Fushion bằng `PYTHONPATH=. D:/miniconda/conda-envs/omni/python.exe -X utf8 -m …` (không dùng `-I`, nếu không sẽ không import được
  `uemr_htstep`); test: thêm `UNIAV_REPO=D:/Học/KL/Code/UniAV/UniAV-fixed`.
- Token HF mới nằm ở `D:/Học/KL/Code/pipeline/.env` (`HF_TOKEN`, tài khoản `GraceFeel`); token trong `~/.cache/huggingface` đã hết hạn.
- Log Colab là PDF trong `notebooks/uemr/log/`; đọc bằng `pdftotext -layout` (`/mingw64/bin`).
- `notebooks/uemr/uemr_htstep_AZ.ipynb` có thay đổi chưa commit (output của lượt dev 1 do user lưu): không đụng.
- Vector `h_i` của dev ở `D:/Học/KL/output/uemr_chot/dev/student/` (không commit, *.npz bị gitignore): `final_dev score` cần thư mục này.
