# UEMR chốt: tiến độ (để phiên sau làm tiếp)

Cập nhật 2026-11-12, khi user tắt session. Nhánh `htstep` của Fushion, commit mới nhất `65ed398` (chưa push). Chi tiết từng bước nằm ở
Nhật ký cuối file 01 và 02; số liệu cho báo cáo ở `SO_LIEU_DU_LIEU.md`; thứ tự chạy ở `HUONG_DAN_CHAY.md`.

## Đã xong

| Bước | Kết quả | Ở đâu |
|---|---|---|
| Dọn code chưa chạy (file 00 §5) | `uemr_htstep/_unused/`, 42 test pass | commit `e77f59f` |
| File 01 §1–4: vector thầy Omni `best` (A1, A2) | `GraceFeel/htstep-omni-av-best`: 5.431 train + 510 val video, thiếu 7,16 % (ngưỡng 8 %, user quyết định) | Nhật ký file 01 |
| File 01 §5: train UniAV (A3) | ckpt sha256 `9447de87…`, epoch 9, mAP val (article GT) 30.29; `GraceFeel/uniav-htstep-best` | Nhật ký file 01 |
| File 01 §6: kiểm UniAV mới (local CPU) | mAP article đoán 20.14 > 19.55 (cũ) → **đoạn R6 lấy từ UniAV mới**; recall@0.5 33.04; θ 0,402 | `results/uemr_final/uniav_best_checks.json` |
| File 02 §1: CHECK tái lập R8 (b) lượt dev 1 từ cache | R@1 multi 57.99, strict 21.81, mảng hạng và VCMR trùng hệt: **OK** | chưa ghi Nhật ký file 02 (ghi ở phiên sau) |

Đầu ra local của §6: `D:/Học/KL/output/uemr_chot/uniav_best/` (`uniav_raw.jsonl` sha256 `5db3e8cb…`, `checks.json`, `omni_checks.json`,
`predictions_*.csv`); checkpoint và bộ thầy đã tải ở `D:/Học/KL/output/uemr_chot/{uniav_best_model,omni_av_best}`.

## Việc tiếp theo (theo thứ tự)

1. **Ghi Nhật ký file 02 §1**: CHECK tái lập ở trên (hash `segments.npz` `16636ad1…`, `text.npz` `4c8a1c8f…` của
   `D:/Học/KL/output/htstep_uemr/encode_omni/htstep_uemr/a4ca43db/best/db`).
2. **File 02 §2–3 (local CPU, không GPU)**, đang làm dở khi dừng (chưa viết code nào cho bước này):
   - các dòng với UniAV mới: θ theo luật (đã ra 0,402), R6, R3 (K_pred), R7, R0, và R8 cho cả lưới `L_min` ∈ {3, 5} × `L_fill` ∈ {8, 10, 15}
     (`uemr_htstep.partitions.build` / `omni_retrieval.bo_sung.fill_gaps`);
   - `h_i` bằng `run_uniav student` (checkpoint mới, `--omni-dir D:/Học/KL/output/uemr_chot/omni_av_best`) cho mọi dòng, và R8 với
     `--article rand` (seed 0) cho C3;
   - chọn `L_min`, `L_fill` theo R@1 multi của cột (S) trên R8: (S) = text Omni `best` (`text.npz` ở cache dev 1) · `h_i`
     (`SegmentIndex(h, video, ts, te)` + `score.run`); hoà lấy (5, 10) hoặc gần nhất; ghi mọi cấu hình vào `results/uemr_final/dev/tried.jsonl`;
   - L của R2 = tổng clip R2 bằng tổng clip R8 đã chọn (`partitions.pick_chunk`).
3. **Notebook `notebooks/uemr/uemr_final_dev.ipynb` (Colab A100)**: encode Omni `best` các clip mới của R2, R3, R6, R8 (cache dev 1 theo
   khoá `video__ts_te`, chỉ encode clip chưa có), chấm mọi dòng × (b), (S), (U)_β, C1 (z-score), chọn β\*, phương pháp chính, khoá
   `configs/uemr_final.json` (file 02 §5–8).
4. Nhánh B (song song, user chạy): `notebooks/uemr/uemr_final_benchmark.ipynb` B1 / B2 (YouCook2), chưa có log.
5. Sau đó: file 03 §4–9 (benchmark, chấm một lần), file 04 (báo cáo).

## Lưu ý cho phiên sau

- Chạy module Fushion bằng `PYTHONPATH=. D:/miniconda/conda-envs/omni/python.exe -X utf8 -m …` (không dùng `-I`, nếu không sẽ không import được
  `uemr_htstep`); test: thêm `UNIAV_REPO=D:/Học/KL/Code/UniAV/UniAV-fixed`.
- Token HF mới nằm ở `D:/Học/KL/Code/pipeline/.env` (`HF_TOKEN`, tài khoản `GraceFeel`); token trong `~/.cache/huggingface` đã hết hạn.
- Log Colab là PDF trong `notebooks/uemr/log/`; đọc bằng `pdftotext -layout` (`/mingw64/bin`).
- `notebooks/uemr/uemr_htstep_AZ.ipynb` có thay đổi chưa commit (output của lượt dev 1 do user lưu): không đụng.
