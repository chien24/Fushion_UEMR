# UEMR chốt: tiến độ (để phiên sau làm tiếp)

Cập nhật 2026-11-20. Nhánh `htstep` của Fushion; xem `git log` cho commit mới nhất (chưa push). Chi tiết từng bước nằm ở
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

1. **Benchmark YouCook2 xong (2026-11-20, user chốt hướng a):** kết quả chính = lượt đủ kho 1.500 video, 11.590 câu, protocol `1c2ea146…`:
   K1, K2, K3 đạt; K4, K5 không đạt. Phương án B (635 video) chỉ còn là tham chiếu. Chi tiết: Nhật ký file 03 mục 2026-11-20,
   `results/uemr_final/bench/yc2/score_print.txt`.
2. Mọi cặp CI đã có (user in từ HF, chép vào `score_print.txt`). Các file json gốc (`pairs`, `metrics`, `criteria`, `benchmark_protocol`)
   chưa tải về repo (token HF local 401). Dòng 1–3 bảng cặp của paper còn `??` (user: viết file md trước).
3. Cập nhật theo kết quả chính: paper `Report/Bao/UEMR/SSRC_template/htstep_paper.tex` §V; `D:/Học/KL/Result/uemr_final/KET_QUA_UEMR.md`
   (mục 1, 7, 10 ý 2, 12; mục 9 ví dụ định tính); báo cáo Word (`Report/scripts/uemr_chot_docx/body.txt`, các TODO benchmark).

## Lưu ý cho phiên sau

- **Dọn repo 2026-11-13** (yêu cầu của user): chỉ còn code của plan chốt: `uemr_htstep/`, phần `omni_retrieval/` mà nó dùng (encode, collect,
  manifest, evaluate, search, partitions, partitions_uemr, bo_sung, protocol, config, subset), `third_party/omni_fix`, `configs/`, `tests/`.
  Đã xoá `uemr2/`, `uemr_fusion/`, `uniav_app/`, `third_party/athena`, `uemr_htstep/_unused/`, `teacher_encode.py`, notebook và config YouCook2 cũ,
  phần HT-Step bổ sung chưa chạy. Hai hàm còn dùng được chuyển nguyên văn: `postprocess_uemr` → `uemr_htstep/postprocess.py`, `maxsim` →
  `uemr_htstep/metrics.py`. Kiểm sau khi dọn: 95 test pass; chạy lại `final_dev rows` + `score` ra thư mục tạm cho 32 / 32 file kết quả, bộ đoạn và
  `uemr_final.json` trùng hệt bản đã khoá. Giữ làm kinh nghiệm HT-Step: `results/uemr_htstep/a4ca43db/` (lượt dev 1) và
  `notebooks/uemr/uemr_htstep_AZ.ipynb`. Code cũ vẫn lấy lại được từ git (commit trước dọn: `3cbd0a7`); các notebook chốt ghim commit cũ nên
  không bị ảnh hưởng.

- Chạy module Fushion bằng `PYTHONPATH=. D:/miniconda/conda-envs/omni/python.exe -X utf8 -m …` (không dùng `-I`, nếu không sẽ không import được
  `uemr_htstep`); test: thêm `UNIAV_REPO=D:/Học/KL/Code/UniAV/UniAV-fixed`.
- Token HF mới nằm ở `D:/Học/KL/Code/pipeline/.env` (`HF_TOKEN`, tài khoản `GraceFeel`); token trong `~/.cache/huggingface` đã hết hạn.
- Log Colab là PDF trong `notebooks/uemr/log/`; đọc bằng `pdftotext -layout` (`/mingw64/bin`).
- `notebooks/uemr/uemr_htstep_AZ.ipynb` có thay đổi chưa commit (output của lượt dev 1 do user lưu): không đụng.
- Vector `h_i` của dev ở `D:/Học/KL/output/uemr_chot/dev/student/` (không commit, *.npz bị gitignore): `final_dev score` cần thư mục này.
- Store Omni clip mới của dev đã tải về `D:/Học/KL/output/uemr_chot/hf_encode_omni/uemr_final/dev/best/clips/` (505 MB).
