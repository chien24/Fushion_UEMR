# UEMR chốt: code ở đâu, chạy ở đâu

Cập nhật 2026-11-14. Dùng để trả lời nhanh khi được hỏi "code đâu, chạy chỗ nào, kết quả nào ra từ đâu".
Thứ tự chạy chi tiết: `HUONG_DAN_CHAY.md`. Lệnh đầy đủ và số liệu từng bước: Nhật ký cuối file `UEMR_CHOT_01` đến `03`.

## 1. Ba repo

| Repo | Nhánh / commit | Vai trò | Có sửa không |
|---|---|---|---|
| Fushion: `github.com/chien24/Fushion_UEMR` (local `D:\Học\KL\Code\Fushion`) | nhánh `htstep` | **code của khoá luận**: cắt đoạn, h_i, encode Omni, chấm, notebook, tài liệu, kết quả | code của mình |
| HanLi-Retriever (UniAV): `github.com/im-xiaoming/HanLi-Retriever` (local `D:\Học\KL\Code\UniAV\UniAV-fixed`) | `77fc4fa` | model UniAV, train, trích đặc trưng | **không sửa**, chỉ ghim commit |
| Omni-fix: `github.com/im-xiaoming/Omni-Retriever-Custom` | `htstep_lora` @ `ab772eb` | train LoRA Omni `best` (đã xong từ trước), loader clip cắt sẵn | không sửa; loader được chép nguyên vào `Fushion/third_party/omni_fix` |

**Trước khi đưa link GitHub cho thầy:** push nhánh `htstep`. Ngày 2026-11-14 còn 5 commit chưa push (từ `91342d8` dọn repo tới
`8384829`), nên GitHub vẫn hiện code cũ trước khi dọn. Các notebook ghim commit đã push nên chạy không bị ảnh hưởng.

## 2. Code Fushion theo chức năng

| Chức năng | File |
|---|---|
| Chạy UniAV với article đoán (hook, không sửa model), tính h_i bằng span head | `uemr_htstep/uniav_htstep.py`, `uemr_htstep/run_uniav.py` |
| Đề xuất UniAV → R6 (θ, gộp, nới đoạn ngắn) | `uemr_htstep/postprocess.py`, `uemr_htstep/partitions.py` |
| R8 (lấp chỗ trống), R2, R3 | `omni_retrieval/bo_sung.py` (`fill_gaps`), `omni_retrieval/partitions_uemr.py`, `omni_retrieval/partitions.py` |
| Encode Omni: e_i của đoạn, t của câu hỏi | `omni_retrieval/encode.py` (+ `manifest.py`, `collect.py`) |
| MaxSim, multi-positive, VCMR | `uemr_htstep/metrics.py`, `omni_retrieval/search.py` |
| Chấm, bootstrap CI | `uemr_htstep/score.py`, `omni_retrieval/evaluate.py` |
| Dev: dòng, chọn L, chấm, chọn β\*, khoá config | `uemr_htstep/final_dev.py` |
| Benchmark: kiểm kê YouCook2, câu hỏi, dòng, chấm, K1–K6 | `uemr_htstep/bench.py`, `uemr_htstep/final_bench.py` |
| z-score không nhãn (UEMR-Hybrid) | `uemr_htstep/bo_sung.py` |
| Dựng bộ vector thầy cho UniAV | `uemr_htstep/teacher_best.py` |
| Kiểm UniAV mới | `uemr_htstep/uniav_best_checks.py` |
| Unit test | `tests/` (95 test) |
| Cấu hình đã khoá | `configs/uemr_final.json` (sha256 `c9840ffb…`) |
| Sơ đồ kiến trúc | `docs/uemr_chot/UEMR_KIEN_TRUC.html` |

## 3. Kết quả nào ra từ đâu

| Bước | Chạy ở đâu | Notebook / lệnh | Code chính | Log | Đầu ra |
|---|---|---|---|---|---|
| Vector thầy Omni best (file 01 §1–4) | Colab A100 | `notebooks/uemr/uemr_chot_01_teacher_best.ipynb` | `teacher_best.py`, `encode.py` | `notebooks/uemr/log/uemr_chot_01_teacher_best_A1.pdf`, `_A2.pdf` | HF `GraceFeel/htstep-omni-av-best` (private) |
| Train UniAV mới (file 01 §5) | Colab A100 | `notebooks/uemr/uemr_chot_01_train_uniav.ipynb` → `python train.py --output htstep_best` | HanLi `train.py` | `log/uemr_chot_01_train_uniav_smoke_{true,false}.pdf` | HF `GraceFeel/uniav-htstep-best` (private), sha256 `9447de87…` |
| Kiểm UniAV mới (file 01 §6) | **local CPU** | `python -m uemr_htstep.run_uniav proposals / check / omni`, rồi `python -m uemr_htstep.uniav_best_checks` | `run_uniav.py`, `uniav_best_checks.py` | Nhật ký file 01 | `results/uemr_final/uniav_best_checks.json` |
| Dev: dòng R0–R8, h_i, chọn L (file 02 §2–4) | **local CPU** | `python -m uemr_htstep.final_dev rows`, `run_uniav student`, `final_dev select` | `final_dev.py`, `run_uniav.py` | Nhật ký file 02 | `results/uemr_final/dev/segments/`, `tried.jsonl` |
| Dev: encode 36.119 clip mới (file 02 §5) | Colab A100 | `notebooks/uemr/uemr_final_dev.ipynb` | `encode.py` | `log/uemr_final_dev_smoke_{true,false}.pdf` | HF `GraceFeel/encode_omni`, `uemr_final/dev/best/clips/` |
| Dev: chấm, chọn β\*, khoá config (file 02 §6–8) | **local CPU** | `python -m uemr_htstep.final_dev score` | `final_dev.py`, `score.py`, `metrics.py` | Nhật ký file 02 | `results/uemr_final/dev/*.json`, `configs/uemr_final.json` |
| Benchmark YouCook2 (file 03) | Colab A100 | `notebooks/uemr/uemr_final_benchmark.ipynb` (cả pipeline + chấm một lần) | `bench.py`, `final_bench.py`, `run_uniav.py`, `encode.py`, HanLi `tools/pipeline/extract_features.py` | **chưa chạy lượt thật** | HF `GraceFeel/uemr-bench-yc2` → sau đó `results/uemr_final/bench/yc2/` |
| Báo cáo | local | | | | `D:\Học\KL\Result\uemr_final\KET_QUA_UEMR.md`, `D:\Học\KL\Report\BAO_CAO_KLTN_UEMR_CHOT.docx` |

Lệnh local chạy từ thư mục `Fushion`:
`PYTHONPATH=. D:/miniconda/conda-envs/omni/python.exe -X utf8 -m uemr_htstep.<module> …`. Cờ đầy đủ in ở đầu mỗi file
(`final_dev.py` dòng 4–10, `run_uniav.py` dòng 3–7).

Dữ liệu trung gian không commit (lớn): `D:\Học\KL\output\uemr_chot\` (đầu ra UniAV, h_i của dev, store Omni đã tải về).

## 4. Câu hỏi hay gặp

- **"Phần nào là code của em?"** Toàn bộ Fushion nhánh `htstep`. UniAV (kiến trúc, span head, loss chưng cất) là code của nhóm HanLi, dùng
  nguyên ở `77fc4fa`. Omni và LoRA `best` là của Omni-fix.
- **"Sao một phần chạy local, một phần Colab?"** Máy local không có GPU CUDA. Phần cần Omni 7B (encode) hoặc cần train thì chạy Colab A100.
  Phần chạy CPU được (UniAV forward, cắt đoạn, chấm) thì chạy local. Riêng phần chấm dev chạy local để khỏi đưa ~0,5 GB vector h_i lên Colab
  (đã ghi trong Nhật ký file 02).
- **"Số dev ở đâu ra?"** `results/uemr_final/dev/metrics.json` và `pairs.json`, do `final_dev score` tạo. Bảng tóm tắt ở Nhật ký file 02 §9.
- **"Có chọn tham số trên YouCook2 không?"** Không. θ, L_min, L_fill, β\* và phương pháp chính khoá trong `configs/uemr_final.json` trên dev
  HT-Step. Ô 6 của notebook benchmark kiểm sha256 của file này trước khi chạy, ô 9 chỉ chấm một lần.
- **"Chạy lại thế nào?"** `HUONG_DAN_CHAY.md`: mỗi notebook Run all với `SMOKE = True` rồi `SMOKE = False`. Notebook tự clone đúng commit,
  nên không phụ thuộc máy.
