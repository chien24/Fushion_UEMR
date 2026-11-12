# Số liệu dữ liệu (để viết báo cáo)

Cập nhật 2026-11-12. Mỗi số có nguồn. Số chưa đo ghi "chưa có". Không so số HT-Step với YouCook2.

## 1. HT-Step: video và vai trò

Annotation: `annotations.json` chính thức (github.com/facebookresearch/htstep, commit `cca1dea`, sha256 `e40d8a2b…`): 17.472 video
train, 600 video val_seen.

| Tập | Video | Đoạn / câu | Dùng để | Nguồn số |
|---|---|---|---|---|
| Train Omni `best` (LoRA) | 6.993 | 43.261 clip | train Omni | `train_omni_htstep.jsonl`, `configs/htstep_train_ids.json` |
| Train UniAV | 5.826 | 36.026 đoạn GT | train UniAV | video có đặc trưng K400 + BEATs; `htstep_train_ids.json`, Nhật ký file 01 |
| Giao train Omni ∩ train UniAV | 5.431 | | | `configs/htstep_uemr_protocol.json` (`train_ids.both`) |
| Chỉ train Omni | 1.562 | | | như trên (`omni_only`) |
| Chỉ train UniAV | 395 | | | như trên (`uniav_only`) |
| Dev (kho val_seen) | 510 | 2.990 câu (877 Full Match) | chọn tham số, không phải kết quả | protocol `a4ca43db`; 600 val_seen → 510 có đặc trưng UniAV; giao với train hai model = 0 |

Vì sao tập train khác nhau: Omni cần mp4 cắt được clip; UniAV cần đặc trưng K400 + BEATs đã trích (`nguyenminh04/htstep-feats-k400`).

## 2. Vector thầy Omni `best` cho UniAV (file 01, A2)

Adapter `best` sha256 `5e9899aa…`. Encode bằng `omni_retrieval.encode` (đường cột (b)). Kết quả: `GraceFeel/htstep-omni-av-best`
(`config.json` sha256 `f9abbce3…`).

| | Video | Đoạn có vector | Đoạn thiếu | Video lấy từ |
|---|---|---|---|---|
| Train | 5.431 / 5.826 | 33.445 / 36.026 | 2.581 (7,16 %) | `nguyenminh04/htstep-videos` (33.326 đoạn qua clip cắt sẵn `GraceFeel/htstep-omni-clips`, phần còn lại đọc mp4) |
| Val | 510 / 510 | 2.990 / 2.990 | 0 | `nguyenminh04/htstep-val-seen-videos`; vector lấy lại từ cache lượt dev 1, không encode lại |

Lý do thiếu (train): 2.321 đoạn của 361 video không có mp4; 89 đoạn mp4 không tải / chuyển được; 171 đoạn encode lỗi (mp4 hỏng). Thầy
cũ (`nguyenminh04/htstep-omni-av`) cũng thiếu đúng 361 video không có mp4 (5.465 / 5.826 video có file). HanLi bỏ đoạn thiếu khỏi loss
chưng cất (`omni_ok`); UniAV vẫn train trên đủ 5.826 video. Ngưỡng 8 % do user quyết định (plan ghi 5 %).

Kiểm đường encode: 20 clip GT val so với `val_embeds_best.npz` cos trung bình 0,99991, nhỏ nhất 0,99986. Encode: 0,236 s/clip trên A100.

## 3. Các repo HF và cái gì nằm trong đó

| Repo | Video | Nội dung | Dùng để |
|---|---|---|---|
| `nguyenminh04/htstep-videos` | mp4 train | video gốc | nguồn của mọi clip train |
| `nguyenminh04/htstep-val-seen-videos` | mp4 val_seen | video gốc | nguồn video val |
| `GraceFeel/htstep-omni-clips` | 6.993 train + 510 val | clip cắt sẵn (8 khung + 8 s tiếng) | train Omni `best` |
| `GraceFeel/encode_omni` (`htstep_uemr/a4ca43db/`) | 510 val | 41.248 clip (R0–R8, R7) + 2.990 text, Omni `best` | truy hồi trên dev (lượt dev 1) |
| `GraceFeel/htstep-omni-av-best` | 5.431 train + 510 val | vector thầy của đoạn GT | train UniAV (A3) |
| `nguyenminh04/htstep-feats-k400`, `htstep-val-seen-feats` | train / val | đặc trưng K400 + BEATs, 1 dòng / giây | đầu vào UniAV |

## 3b. UniAV mới (A3)

Train trên 5.826 video (5.431 có thầy), 11 epoch, chọn epoch 9 theo mAP val_seen (article GT): mAP 30.29, ArtAcc 59.61, OmniCos 88.08,
OmniCosC 72.20, OmniR@1 52.04. Checkpoint sha256 `9447de87…`, `GraceFeel/uniav-htstep-best`. Nguồn: log A3, Nhật ký file 01.

## 4. YouCook2 (benchmark, zero-shot)

| Bước lọc | Video | Nguồn |
|---|---|---|
| annotation, split validation | 457 | `youcookii_annotations_uemr.json` (đếm local) |
| có mp4 | 394 | `Data/YouCookII/videos` (đếm local) |
| có tiếng, đọc được | chưa có | notebook benchmark ô 3 |
| trùng YouTube ID với train UniAV / train Omni / dev 510 | 0 / 0 / 0 | đếm local, Nhật ký file 03 |
| câu hỏi | 3.031 (đếm thử với thời lượng annotation; số thật ở ô 4) | Nhật ký file 03 |

## 5. Chưa có (điền khi có log)

- UniAV mới, phần còn thiếu: mAP article đoán, recall theo θ, OmniCos chế độ article đoán (file 01 §6).
- YouCook2: số video sau lọc tiếng / đọc được, số câu thật, đặc trưng, vector 6a (B2).
