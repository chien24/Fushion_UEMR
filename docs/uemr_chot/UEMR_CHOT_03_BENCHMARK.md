# UEMR chốt, file 03: benchmark zero-shot (YouCook2 + bộ khác), chấm một lần

Đọc `UEMR_CHOT_00_TONG_QUAN.md` mục 2, 3, 4 trước. **Zero-shot:** không train, không chọn, không chỉnh gì trên benchmark. Mọi tham số lấy từ
`configs/uemr_final.json` (file 02 §7). Luật không dùng nhãn (L của R2, μ / σ của z-score) được tính lại trên từng bộ, đúng như config ghi.

Notebook `notebooks/uemr/uemr_final_benchmark.ipynb`, ô `# ===== N.`, cờ `BENCHMARKS = ["yc2"]`, `SMOKE`, resume.

## Bộ benchmark

| Mã | Bộ | Bắt buộc | Ghi chú |
|---|---|---|---|
| `yc2` | YouCook2, **train + val** (annotation chính thức; đổi từ "split validation" ngày 2026-11-13, xem Nhật ký) | có | mọi video tải được, có tiếng |
| `anet` | ActivityNet Captions `val_1` | chỉ khi user thêm vào `BENCHMARKS` | lấy mẫu cố định 1.000 video tải được, seed 0; khác miền (UniAV, Omni `best` học trên nấu ăn) |
| khác | do user chọn | | điều kiện: có ranh giới từng bước + câu mô tả từng bước, video có tiếng, tải được |

---

## 1. Kiểm kê và kiểm trùng (không phụ thuộc UniAV, làm song song với file 01)

- In số video ở từng bước lọc: annotation → tải được → có tiếng → đọc được (decord) → không trùng.
- **Kiểm trùng theo YouTube ID** với: train của UniAV (5.826), train của Omni (6.993, `train_omni_htstep.jsonl`), dev 510. Trùng thì loại,
  in số. Bảng ID nằm ở `configs/htstep_train_ids.json` và inventory lượt dev 1.
- Dữ liệu pretrain của Omni gốc (InternVid, InternVid-FLT, Panda-70M, PVD) cũng từ YouTube và không kiểm hết được: ghi vào hạn chế.
- Tải: theo lô, resume, kiểm đĩa; AV1 chuyển H.264 như lượt dev 1.

## 2. Trích đặc trưng UniAV (K400 + BEATs)

- Tìm đúng script và commit đã tạo `htstep-val-seen-feats` (HanLi hoặc repo trích của nhóm). Ghi vào Nhật ký.
- **CHECK tái lập:** trích lại 5 video kho HT-Step, so với `htstep-val-seen-feats`: cos từng bước lưới trung bình ≥ 0.99 cho cả k768 và
  a768. Không đạt thì dừng, báo (không có đặc trưng đúng thì không chạy được UniAV).
- Trích cho video benchmark, cùng lưới 256 bước như `htstep.py`. Đo thời gian trên 5 video, điền vào bảng chi phí (file 02 §8).

## 3. Câu hỏi và đáp án

- Câu hỏi = câu mô tả của từng bước GT, **nguyên văn**, chuẩn hoá text theo `uemr_final.json`. **Không** ghép tên công thức hay activity
  (nhãn GT).
- Multi-positive: mọi video có bước cùng text đã chuẩn hoá. Strict: video gốc. In số câu có text trùng ở ≥ 2 video.
- VCMR: tIoU ≥ 0.5 với một bước có text đó trong video đứng đầu.

## 4. UniAV trên benchmark

Adapter, chế độ article đoán (benchmark không có article HT-Step). Checkpoint, θ (số) theo `uemr_final.json`. Lưu đoạn, `h_i`, article đoán
(chỉ để mô tả: phân bố article HT-Step được đoán), hash. Chất lượng cắt so với GT (recall tIoU 0.3 / 0.5 / 0.7, % bỏ sót, độ phủ, K_pred
so với K_GT) để mô tả.

## 5. Dòng

R0, R1, R2 (L tính lại theo luật), R3, R6, R8 (`L_min`, `L_fill` của config), R7 (oracle).

## 6. Encode

- **6a (song song với file 01):** Omni `best`: R0, R7 và text. Omni pt: R0 và text.
- **6b (sau khi khoá config):** Omni `best`: R2, R3, R6, R8. Omni pt: R8.
- Cache, fp16, `failed.csv`, > 2 % lỗi thì dừng. In số clip, ước lượng giờ.

## 7. Khoá protocol benchmark

`configs/benchmark_protocol.json`: sha256 `uemr_final.json`; danh sách video và câu (+ sha256) của từng bộ; tiêu chí §8; định nghĩa
multi-positive. Commit, in sha256 **trước** ô chấm. Ô chấm kiểm hash.

## 8. Tiêu chí ghi trước (mỗi bộ, multi-positive, R@1 trừ khi ghi khác)

"UEMR" là phương pháp chính trong `uemr_final.json`.

| # | Tiêu chí | Câu được viết nếu đạt |
|---|---|---|
| K1 | cận dưới (UEMR − R0(b)) > 0 | "Trên <bộ>, không train trên bộ này, UEMR hơn một vector Omni cho mỗi video" |
| K2 | cận dưới (UEMR − R3 cùng cột) > 0 | "Cắt theo sự kiện hơn cắt đều cùng số đoạn" |
| K3 | cận dưới (VCMR R6(b) − VCMR R3(b)) > 0 | "Ranh giới sự kiện giúp định vị đúng khoảnh khắc" |
| K4 | cận dưới (UEMR-Lite R8 − R0(b)) > 0 | "Bản UEMR-Lite, không chạy Omni 7B phía video, vẫn hơn một vector Omni cho mỗi video" |
| K5 | cận dưới ((U)_β\* − (b)) trên R8 > 0 | "Vector chưng cất bổ sung được cho Omni" |
| K6 | UEMR − R2, UEMR-Hybrid − R6, R7 − UEMR, pt − `best` (R0, R8) kèm CI | chỉ báo |

**Câu kết luận cho mỗi bộ** (chọn đúng dòng khớp, không viết câu khác):

| Kết quả | Kết luận |
|---|---|
| K1 và K2 đạt | UEMR có tác dụng zero-shot trên <bộ>, cả so với một vector lẫn so với cắt đều |
| K1 đạt, K2 không | UEMR hơn một vector trên <bộ>, nhưng lợi thế so với cắt đều không phân biệt được |
| K1 không | Trên <bộ>, UEMR không hơn một vector Omni cho mỗi video; ghi là giới hạn đã đo, kèm chất lượng cắt |

## 9. Chấm và xuất

Chấm **một lần**: mọi dòng × cột × (strict, multi-positive), cặp và CI, tiêu chí → `results/uemr_final/bench/<bộ>/{inventory,uniav_quality,
metrics,ranks_*,pairs,criteria}.json`. In tóm tắt. `run_manifest.json` ghi mọi lượt, để chứng minh chỉ có một lượt chấm.

## 10. Nhật ký

(Agent ghi: số từng bước lọc và kiểm trùng, script và commit trích đặc trưng, CHECK tái lập đặc trưng, θ / L / μ / σ đã dùng, sha256
protocol, số clip, giờ chạy, CHECK từng ô.)

### 2026-11-12 (agent, local, không GPU): §1–3 phần làm được ở local, notebook cho §1–3 và ô 6a

Commit Fushion `cdb2fac` (`uemr_htstep/bench.py`, `tests/test_bench.py`), `f921c65` (`notebooks/uemr/uemr_final_benchmark.ipynb`,
`FUSHION_REF = cdb2fac`). Chưa push. Notebook chưa chạy; **không chỉ số truy hồi nào được tính**.

**§1, đếm ở local** (`D:/Học/KL/Data/YouCookII`, annotation `metadata/youcookii_annotations_uemr.json`, split `validation`):

| Bước lọc | Video |
|---|---|
| annotation (validation) | 457 |
| tải được (có mp4) | 394 |
| có tiếng, đọc được (decord) | đo ở ô 3 |
| trùng YouTube ID: train UniAV (5.826) / train Omni (6.993) / dev 510 | 0 / 0 / 0 |

- 63 video validation không có mp4 (không tải được từ các lượt YC2 trước); notebook không tải lại từ YouTube.
- Kiểm thêm: 0 video validation nằm trong 17.472 video train và 600 video val_seen của `annotations.json` HT-Step. Toàn bộ YouCook2
  (mọi split) chỉ trùng 1 ID với HT-Step train, và precut của Omni đã loại đúng 1 video vì `yc2_overlap`.
- File annotation ở local là bản `_uemr` (split train đã tách thêm `dev`), không phải file gốc. Ô 3 tải
  `youcookii_annotations_trainval.json` từ trang YouCook2 và CHECK split validation trùng (video, đoạn, câu); không tải được thì dùng bản
  Drive và ghi "chưa đối chiếu".
- Hạn chế ghi vào `inventory.json`: dữ liệu pretrain của Omni gốc (InternVid, InternVid-FLT, Panda-70M, PVD) không kiểm trùng hết được.

**§2, script trích đặc trưng:** `tools/pipeline/extract_features.py` của HanLi-Retriever, `--model k400_f16` (mặc định): InternVideo2
Stage1-1B K710→K400 f16 (`1B_ft_k710_ft_k400_f16.pth`), 4 fps, cửa sổ 16 khung, 224 × 224; BEATs của `audio_6b.pth`, 16 kHz, 3 s quanh mỗi
giây; ra `k768` + `a768` float16, một dòng mỗi giây. Nguồn: model card `nguyenminh04/uniav-htstep-omni` và README
`htstep-val-seen-feats` ("same extractor"). File này tại `77fc4fa` sửa lần cuối ở `b878676` (2026-11-08, phần chia CSV / resume);
trước đó là `cd5ebda` (2026-10-30). Commit chính xác của lượt đã tạo `htstep-val-seen-feats` **không ghi ở đâu** (tar trích trên PC,
Colab và "separately"), nên CHECK tái lập ở ô 5 là bằng chứng duy nhất. Lưới 256 bước do `htstep.py` nội suy khi đọc, không nằm trong
file đặc trưng. Ô 5 dùng `--batch-size 32 --decode-ahead 8` như `colab_dl_extract.sh`. Cần HF token đã chấp nhận licence của
`OpenGVLab/InternVideo2-Stage1-1B-224p-K400` và `InternVideo2-Stage2-6B-Audio`.

**§3, câu hỏi** (`uemr_htstep.bench.queries`; đếm thử ở local với thời lượng của annotation, số thật do ô 4 in theo thời lượng mp4):
394 video, 3.031 câu, 2.962 text khác nhau sau chuẩn hoá, 3,89 % câu có text trùng ở ≥ 2 video. Câu = `sentence` của bước, nguyên văn;
không ghép tên công thức. Chuẩn hoá hiện dùng `uemr_htstep.data.normalize` (luật của lượt dev 1); `uemr_final.json` phải ghi đúng luật
này khi khoá, nếu khác thì tính lại `headline`.

**Notebook `uemr_final_benchmark.ipynb`:** ô 3 kiểm kê + kiểm trùng; ô 4 câu hỏi, R0, R7; ô 5 CHECK tái lập đặc trưng (5 video kho
HT-Step ngắn nhất trong tar `htstep_feats_1000`, cos ≥ 0.99 cho `k768` và `a768`) rồi trích cho benchmark; ô 6a Omni `best` (R0, R7,
text) và Omni pt (R0, text). Ô 7 trở đi (UniAV, R2 / R3 / R6 / R8, khoá protocol, chấm) chưa viết, chờ `configs/uemr_final.json`.
Ước lượng: ô 6a best 394 + 3.031 = 3.425 clip ≈ 0,3 giờ, pt 394 clip ≈ 2 phút, cộng nạp model hai lần; đặc trưng: 34,2 giờ video,
tốc độ trên A100 chưa đo (ô 5 in sau 5 video).

Còn trống, điền sau khi có log Colab: số "có tiếng" / "đọc được", CHECK annotation chính thức, CHECK tái lập đặc trưng, tốc độ trích,
sha256 danh sách video và câu hỏi, số clip lỗi.

### 2026-11-13 (quyết định của user): kho YouCook2 = train + val; notebook không ghi Drive

- **Đổi so với plan gốc:** kho benchmark `yc2` là **mọi video YouCook2 có annotation và có mp4** (train + val), không chỉ split validation.
  Lý do của user: muốn dùng toàn bộ video đang có. Hợp lệ với zero-shot vì trong plan chốt không model nào train trên YouCook2 (UniAV: HT-Step;
  Omni `best`: adapter pt gốc + LoRA HT-Step). Quyết định trước khi có bất kỳ số truy hồi nào trên YouCook2 của plan này. Hạn chế đã ghi vẫn
  giữ: dữ liệu pretrain của Omni gốc không kiểm trùng hết được.
- **Đếm ở local** (`D:/Học/KL/Data/YouCookII`): annotation 1.790 video (1.333 train, trong đó 104 từng gắn nhãn `dev` ở các lượt YC2 cũ; 457
  val); có mp4 **1.500** (1.002 + 104 + 394); 11.594 bước có câu (trước khi cắt theo mp4); 131,5 giờ video (riêng val 34,2 giờ, tức kho mới
  gấp ~3,8 lần). Trùng YouTube ID với train UniAV / train Omni / dev 510: 0 / 0 / 0.
- **Lỗi notebook B:** ô 3 đòi `metadata/youcookii_annotations_uemr.json` trên Drive, Drive của user không có file này. Sửa: annotation nằm
  trong repo (`configs/yc2/youcookii_annotations_uemr.json`, bản local đã dùng), vẫn đối chiếu với bản chính thức (video, đoạn, câu) nếu tải
  được; CHECK số video 1.790.
- **Không ghi Drive** (Drive của user gần đầy): đầu ra ở `/content/uemr_chot_run/bench/yc2`, đẩy HF `GraceFeel/uemr-bench-yc2` (private) sau
  mỗi ô và mỗi 20 phút; ô 1 tải lại từ HF để resume. Drive chỉ còn được đọc (video, adapter `best`).
- Code: `bench.yc2(database, subset=None)` (commit `12c7775`); notebook `FUSHION_REF = 12c7775`, cờ `YC2_SUBSET = None`. Chưa chạy.
- **Ước lượng mới (chưa đo):** ô 6a ~14.600 clip (R0 + R7 `best`, R0 pt) ≈ 1,0–1,3 giờ A100; trích K400 + BEATs cho 131,5 giờ video: chưa đo
  (ô 5 in tốc độ sau 5 video). Các ô sau (R2, R3, R6, R8 `best` + R8 pt) ước ~165.000 clip ≈ 11–15 giờ A100, chia nhiều phiên được nhờ resume.

### 2026-11-13 (agent): gộp cả pipeline + chấm vào một notebook (yêu cầu của user)

Notebook `uemr_final_benchmark.ipynb` giờ đi hết §1–9 trong một lần Run all (`FUSHION_REF = c7e020e`): ô 3–4 kiểm kê, câu hỏi; ô 5 đặc
trưng; ô 6 UniAV → đoạn → `h_i`; ô 7 khoá protocol; ô 8 Omni; ô 9 chấm một lần; ô 10 tóm tắt. Code: `uemr_htstep/final_bench.py` (commit
`c7e020e`), dùng lại hàm của lượt dev (`partitions.r6`, `fill_gaps`, `pick_chunk`, `row_indexes`, `score.run`, `score.boot`). Chưa chạy.

**"Cùng phương thức với model đã train"** (yêu cầu của user), từng khâu:
- Đặc trưng: script HanLi `extract_features.py --model k400_f16` đã tạo đặc trưng train UniAV; CHECK tái lập cos ≥ 0,99 trên 5 video HT-Step.
- UniAV: HanLi `77fc4fa`, `HTStepDataset` không sửa, checkpoint sha256 của config, chế độ article đoán, θ = 0,402 (số của config).
  Video benchmark được ghi theo định dạng `annotations.json` của HT-Step với một bước giữ chỗ (dataset cần ít nhất một annotation).
  **Thử ở local trên 3 video dev:** với annotation giữ chỗ và không có vector thầy, đề xuất + article đoán trùng hệt `uniav_raw.jsonl` của
  lượt dev; đổi bước giữ chỗ không đổi gì; `h_i` trên đoạn R6 sai khác 0 so với lượt dev. Ô 6 lặp lại CHECK đổi bước giữ chỗ trên 5 video.
- Omni: `omni_retrieval.encode`, mp4 + `timestamps`, adapter `best` sha256 `5e9899aa…`, đúng đường cột (b) của dev.
- Chấm: `final_bench.score`, cùng `score.run` / bootstrap theo video (1000, seed 0) với dev; β\* = 1,0 và phương pháp chính đọc từ config;
  μ, σ của z-score và L của R2 tính lại trên kho (luật không nhãn của config).

**Khác plan (ghi để user biết):**
- §6a / 6b gộp thành một ô Omni chạy sau UniAV (config đã khoá nên không cần tách).
- §7: `benchmark_protocol.json` được ghi vào `results/` của lượt chạy, in sha256 và đẩy HF **trước** ô chấm; ô chấm kiểm sha256. Không commit
  git trước khi chấm được vì chấm chạy ngay trong notebook; agent commit file này vào `configs/` sau khi có log. Lịch sử commit trên HF là bằng
  chứng thứ tự thời gian.
- Chấm một lần: ô 9 không chấm lại nếu đã có `results/score/criteria.json`; `run_manifest.json` đếm số lượt chấm. `SMOKE = True` có chấm thử trên
  5 video (thư mục riêng, không đẩy HF), số đó vô nghĩa và không được dùng.
- Luật mới, không dùng nhãn, ghi trong protocol: video không có đề xuất nào ≥ θ thì không có đoạn R6 / R3 (không truy hồi được ở hai dòng đó,
  cột "video có vector" cho biết số video), R8 của nó chỉ gồm đoạn lấp. Lượt dev không gặp trường hợp này.
- K2 và K6 "UEMR − R2" so cùng cột với phương pháp chính ((U) β\* trên R8 − (U) β\* trên R3 / R2); cặp cột (b) cũng được ghi trong `pairs.json`.
- Phân tích nhóm (bước bị bỏ sót, đoạn lấp, C3) không chạy trên benchmark; hạng từng câu được lưu nên tính sau được mà không chấm lại.
- Ô 9 cần ~25 GB RAM (ma trận điểm 11,6 nghìn câu × ~50 nghìn đoạn R8).
