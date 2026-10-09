# PROMPT — Chạy UEMR (Athena + Omni) theo UEMR_FINAL.md, chỉ inference

> Dán toàn bộ file này cho agent (Claude Code) chạy trên máy Windows. Kèm theo file `UEMR_FINAL.md`.
> Ngôn ngữ làm việc: tiếng Việt. Code và tên file: tiếng Anh.

---

## 0. Vai trò và mục tiêu

Bạn là kỹ sư nghiên cứu, triển khai pipeline **UEMR** theo `UEMR_FINAL.md` (đọc kỹ toàn bộ trước khi làm).
Phạm vi lần này: **P0 → P5** trong mục 11 của `UEMR_FINAL.md`, tức là:

- cache offline UniAV/Athena (Z + proposals),
- sinh đoạn cho R0–R7,
- cache embedding Omni,
- **Exp 1 cột (b) trước (Omni đã fine-tune), sau đó cột (a) (Omni pretrained)**,
- Exp 3 (nhiễu ranh giới, chưa có fusion).

**Không train Omni, không train Athena/UniAV.** Cả hai đã fine-tune xong, chỉ chạy inference.
Context Fusion (P6+) để lượt sau; lần này chỉ cần dựng sẵn cấu trúc cache sao cho P6 dùng được ngay.

---

## 1. Đường dẫn

| Tên | Đường dẫn | Ghi chú |
|---|---|---|
| `ATHENA_ROOT` | `D:\Học\KL\Code\UniAV\UniAV-fixed` | Code UniAV đã bị bạn cùng nhóm sửa và đổi tên thành **Athena** (repo `im-xiaoming/Athena-Retriever`, nhánh `ov-refine`) |
| `OMNI_ROOT` | `D:\Học\KL\Code\Omni\Omni-fix` | repo `im-xiaoming/Omni-fix`, nhánh `chien` |
| `UEMR_ROOT` | `D:\Học\KL\Code\Fushion` | **Mọi code mới viết ở đây** |
| `CKPT_ROOT` | `D:\Học\KL\output` | Chứa 2 model đã fine-tune (Athena + adapter Omni) |
| Dữ liệu YouCook2 | `D:\Học\KL\Data\YouCookII` | `videos/`, `metadata/` (theo `OMNI_ROOT/GUIDE_TRAIN_EVAL_RETRIEVAL.md`) |
| WAVE-7B, BEATs, adapter gốc | `D:\Học\KL\Code\Omni\WAVE_HOME\...`, `D:\Học\KL\Code\Omni\adapters\omniretriever-7b` | adapter gốc = Omni **pretrained** (cột a) |

Đường dẫn có tiếng Việt: luôn đặt `$env:PYTHONIOENCODING="utf-8"`, mở file bằng `encoding="utf-8"`, dùng `pathlib`.

### ⚠️ Tên gọi lẫn lộn UniAV / Athena

Repo đã đổi tên **một phần**. Khi tìm file, phải tìm cả hai tên:

- Package inference: `athena/` (`import athena as uv`) — nhưng alias trong code vẫn là `uv`.
- Thư mục local vẫn là `UniAV\UniAV-fixed`.
- Checkpoint có thể tên `uniav_iv2.pth`, `athena.pth`, `athena_<run>.pth`, hoặc checkpoint train `ckpt/<run>/best_cap.pth.tar` / `best_seg.pth.tar`.
- HF dataset vẫn tên `nguyenminh04/uniav-youcook2-data`.
- Code train vẫn ở `libs/` (`libs/modeling/event_archs.py`, `libs/datasets/youcook2_cap.py`), config `configs/youcook2_event.yaml`.

Dùng `rg -i "uniav|athena"` để định vị, đừng đoán.

---

## 2. Những gì đã biết về Athena (khác với bài báo UniAV và khác với UEMR_FINAL.md)

Đã đọc từ repo `ov-refine`. **Phải xác minh lại trên bản local** (có thể local cũ hơn hoặc mới hơn GitHub):

1. **Encoder:** không dùng ONE-PEACE. Đặc trưng đầu vào là **InternVideo2-1B (v768, hoặc v768+v512) + BEATs (a768)**, **1 dòng/giây**, L2-normalize (`tools/extract_internvideo2.py`, `athena/encoders/`).
2. **Thời gian → index:** mọi video bị **nội suy tuyến tính về `max_seq_len = 256` bước** (`force_upsampling: true`). Bước lưới level 0 ≈ `duration / 256` giây (≈1.2 s với video 5 phút). Hàm đổi chuẩn là `athena/features.py::FeatureSpec.to_seconds` (và công thức tương ứng ở `libs/datasets/youcook2_cap.py`: `feat_stride`, `feat_offset`). **Viết hàm ngược `seconds_to_index` dựa đúng công thức này**, không giả định `z_t` = giây `t`.
3. **Pyramid:** `backbone_arch [2,3,5]`, `scale_factor 2` → 6 level, stride `1,2,4,8,16,32` bước lưới. Level 0 (stride ≈ 1.2 s) thỏa yêu cầu "stride 1–2 s" của UEMR → **dùng level 0 cố định**.
4. **Nguồn Z (chọn 1, ghi vào config, cố định cho mọi thí nghiệm):**
   - `Z_feat`: `torch.cat(fV[0], fA[0])` — output backbone level 0, 1024-d (gần UniAV gốc nhất), hoặc
   - `Z_raw0`: `raw0` của `EmbedHead` level 0, 512-d (`model.last_level0`).
   Mặc định dùng `Z_feat`; lưu thêm `Z_raw0` nếu rẻ (để ablation sau). Lấy bằng forward hook / đọc `model._last['feats']`, **không sửa code Athena**.
5. **Proposals:** `EventCaptionModel.forward` trả về `segs` (đơn vị lưới), `scores` sau Gaussian soft-NMS (tối đa `max_seg_num=100`). API còn có `select_events(min_score, max_overlap=0.3)` — **không dùng bước chọn của API**; lưu proposals thô rồi áp hậu xử lý của UEMR (mục 4A: θ, K_max=16, gộp tIoU>0.7, d_min=2 s, sắp xếp theo thời gian).
   Nếu checkpoint có `state_dict_seg` (`use_seg_weights`), dùng **weights seg** để sinh proposals.
6. **Fine-tune multi-domain:** train trên YouCook2 train + **COIN** (các task "seen"), có file `data/coin_exclude.txt` loại video COIN trùng YouCook2 val.
7. **Teacher OmniRetriever:** khi train, Athena dùng embedding OmniRetriever-7B của các clip GT (`omni_emb_file`, `tools/omni_youcook2.py`). Không dùng ở inference, nhưng có nghĩa là Z của Athena **đã được kéo về gần không gian Omni pretrained**.
8. **Chọn checkpoint:** `train.py` lưu `best_cap` / `best_seg` theo **YouCook2 val** (394 video).

### Hệ quả cần ghi vào báo cáo (không tự sửa, không retrain)

| Vấn đề | Việc cần làm |
|---|---|
| Athena chọn checkpoint theo val (= test của UEMR) | Ghi rõ là **giới hạn** trong `REPORT.md`. Vẫn chạy. |
| Teacher Omni có dùng clip val không? | Kiểm tra `omni_emb_full.npz` / config: key nào được dùng làm loss khi train. Nếu chỉ train → OK. Nếu có val → **dừng và báo**. |
| COIN trùng val | Kiểm tra `coin_exclude.txt` tồn tại và được áp dụng trong `coin_data.py` / dataset. |
| Omni fine-tune có thấy val không | Kiểm tra manifest dùng để train adapter trong `CKPT_ROOT` (log/args). Nếu dùng `train_omni_video.jsonl` đầy đủ (chưa tách dev) → ghi chú. |
| Dev split | Dùng `OMNI_ROOT/scripts/split_youcookii_dev.py` (seed cố định) → `dev_videos.txt`. Dùng **cùng danh sách** cho mọi chỗ cần dev (chọn θ, sau này train fusion). |
| Embedding Athena 512-d (InternVideo2 text space) | **Không dùng** cho retrieval trong UEMR. Retrieval chỉ dùng không gian Omni. Z chỉ là ngữ cảnh. |

---

## 3. Bước 0 — Khảo sát (BẮT BUỘC, chưa viết pipeline)

Làm xong mục này thì **dừng lại, in báo cáo ngắn và chờ tôi xác nhận**.

1. `git -C ATHENA_ROOT status`, `branch`, `log -5`; so với `origin/ov-refine`. Tương tự cho `OMNI_ROOT` (`chien`). Không pull/commit gì nếu tôi chưa đồng ý.
2. Liệt kê `CKPT_ROOT` (đệ quy 2 cấp): file nào là Athena, file nào là adapter Omni. Với Athena: đọc key trong checkpoint (`state_dict`, `state_dict_seg`, config đi kèm, `iv2_video_keys`, `caption_space`, `pyramid_attn`, `text_proj`, có `ground_head` không). Với Omni: `adapter_config.json`, có `modules_to_save` (classify_linear, beats_proj) không, log train (manifest nào, bao nhiêu epoch).
3. Kiểm tra checkpoint Athena **load được** với `athena.Config(checkpoint=..., model_config=...)` và chạy được `uv.describe_sample('6uHoTJSLoL8')` (không cần encoder). Nếu config trong checkpoint khác `configs/youcook2_event.yaml`, ưu tiên config trong checkpoint.
4. Kiểm tra đặc trưng InternVideo2+BEATs của YouCook2 đã có trên máy chưa (`data/youcookii/iv2_feats/*.npz`, ~1500 file). Đếm số video train/val có đặc trưng **và** có mp4 trong `D:\Học\KL\Data\YouCookII\videos`. Phần giao này là tập video dùng cho UEMR.
5. Xác minh các điểm ở mục 2 (đặc biệt 2, 5, 7, 8) trên code local. Chỗ nào khác GitHub thì ghi lại.
6. Kiểm tra mapping giây → index trên 1 video val: vẽ norm/PCA-1 của Z level 0 theo thời gian, chồng ranh giới GT và proposals. Lưu `checks/mapping_<vid>.png`.
7. Ước lượng chi phí P3: tổng số clip Omni = Σ_partition Σ_video K. In bảng theo partition × split. Omni-7B **không chạy được trên máy local** (theo guide của Omni) → P3 chạy trên **Colab A100**; ước lượng giờ GPU.

Báo cáo Bước 0 gồm: bảng checkpoint, các xác minh mục 2 (✅/❌ + bằng chứng file:dòng), số video dùng được, ước lượng chi phí, các câu hỏi cần tôi quyết.

---

## 4. Cấu trúc code trong `UEMR_ROOT`

```text
Fushion/
  configs/uemr.yaml            # mọi siêu tham số (mục 12 UEMR_FINAL) + đường dẫn + seed
  uemr/
    paths.py                   # đọc config, sys.path tới ATHENA_ROOT / OMNI_ROOT (không copy code của họ)
    data.py                    # annotation YouCook2, split train/dev/val, danh sách video hợp lệ
    athena_offline.py          # P1: Z level 0, t2idx, proposals thô + hậu xử lý
    partitions.py              # P2: R0–R7 + Uniform-M@K_GT -> JSON
    omni_manifest.py           # P3a: sinh manifest JSONL cho omniretriever.cli extract
    omni_collect.py            # P3b: gom .npz của Omni về cache chuẩn
    scoring.py                 # MaxSim / top-k mean / LSE
    evaluate.py                # T2V R@1/5/10, MedR; V2T theo MeVTR; VCMR (để sẵn)
    perturb.py                 # Exp 3: làm nhiễu GT theo tIoU mục tiêu
  scripts/                     # entry point từng phase (PowerShell + Colab notebook)
  cache/                       # KHÔNG commit
  results/                     # bảng CSV/MD
  REPORT.md                    # nhật ký quyết định, sai khác, giới hạn
```

Quy tắc:

- **Không sửa** file nào trong `ATHENA_ROOT` và `OMNI_ROOT`. Cần gì thì import, hook hoặc wrap.
- Mọi output có `config_hash` + seed + commit hash của 2 repo, ghi trong `cache/<phase>/meta.json`.
- Mỗi phase **idempotent và resume được** (bỏ qua video đã có cache).

---

## 5. P1 — Athena offline (chạy local, GPU RTX 3060 là đủ)

Cho mọi video trong train ∪ dev ∪ val:

1. Đọc đặc trưng InternVideo2+BEATs từ `.npz` (đúng key mà checkpoint cần), `FeatureSpec.prepare`.
2. Forward Athena (weights seg nếu có). Lưu:
   - `Z` level 0 `[256, d_z]` fp16, mask độ dài hợp lệ,
   - `t2idx`: tham số để đổi giây ↔ index (n, stride, window, fps, max_seq_len),
   - proposals thô: `(t_s, t_e, score)` theo **giây** (dùng `to_seconds`).
3. Hậu xử lý UEMR: lọc `score ≥ θ`, tối đa `K_max=16`, nếu rỗng giữ proposal tốt nhất, gộp tIoU>0.7, kéo dài đoạn < `d_min=2 s`, sắp xếp theo thời gian.
4. **Chọn θ trên dev** sao cho mean K_pred ≈ mean K_GT. In bảng θ → (K_pred trung bình, F1@0.5) trên dev.
5. Báo cáo năng lực detector trên val: mAP@tIoU{0.3,0.5,0.7} (class-agnostic), R@0.5/R@0.7 kiểu Athena (k = #GT) để đối chiếu với README của họ (R@0.5 ≈ 53–54), K_pred vs K_GT.

Cache: `cache/athena/<video_id>.npz` + `cache/athena/proposals.json`.

---

## 6. P2 — Sinh đoạn R0–R7

Theo bảng mục 5 của `UEMR_FINAL.md`, lưu `cache/partitions/<name>.json`: `video_id → [[ts, te], ...]`.

- `global`, `uni_chunk` (L = trung vị độ dài event GT trên train), `uni_M`, `rand_M` (seed cố định, độ dài ≥ d_min), `kmedoids_M` (K-Medoids cosine trên Z level 0, mỗi medoid → `[t_m−1, t_m+1]` giây, dùng `t2idx` để đổi), `uniav_pred`, `gt`, `uni_M_gt`.
- R3/R4/R5 dùng **đúng K_pred của từng video**.
- Mọi đoạn clamp vào `[0, duration_thực_của_mp4]` (một số mp4 ngắn hơn metadata — xem `convert_youcookii.py`).
- Kiểm tra tự động: số đoạn mỗi partition, phân bố độ dài, không đoạn nào < d_min (trừ kmedoids = 2 s).

---

## 7. P3 — Omni offline (Colab A100)

Dùng nguyên CLI của Omni: `python -m omniretriever.cli extract`. CLI đã hỗ trợ `timestamps` trong manifest (cắt frame và audio từ chính video theo cửa sổ).

1. `omni_manifest.py` sinh JSONL, mỗi dòng:
   `{"id": "<vid>__<partition>__<k>", "video": "<vid>.mp4", "timestamps": [ts, te]}` (không có `audio` → audio cắt từ video).
   Gộp các đoạn trùng nhau giữa partition (cùng vid, cùng ts/te làm tròn 0.01 s) để **không encode 2 lần**; giữ bảng ánh xạ.
   Text: một manifest riêng `{"id": "<vid>#<i>", "text": "<caption>"}` cho mọi caption train/dev/val.
2. Lệnh (chạy cho từng adapter):
   ```bash
   python -m omniretriever.cli extract segs.jsonl \
     --base-model /content/WAVE_HOME/WAVE-7B --adapter <ADAPTER> \
     --output omni_<tag>_av.npz --modalities av --device cuda --dtype bfloat16 --batch-size 1
   python -m omniretriever.cli extract caps.jsonl ... --modalities text --batch-size 32
   ```
   - **Lượt 1 — cột (b):** `<ADAPTER>` = adapter fine-tune trong `CKPT_ROOT`, `tag=ft`.
   - **Lượt 2 — cột (a):** `<ADAPTER>` = `YunzeLiu/OmniRetriever-7B` gốc, `tag=pt`. Nếu thiếu GPU, chỉ chạy R0, R3, R6, R7 (theo mục 11 UEMR_FINAL).
   Dùng **đúng cùng tham số tiền xử lý** (số frame, `--pin-video-resolution`, `--duplicate-audio-tokens`, instruction) đã dùng khi fine-tune adapter — đọc từ log train trong `CKPT_ROOT` hoặc `scripts/eval_youcookii.py`. Ghi lại trong `meta.json`.
3. **Sanity check trước khi chạy hết:** encode 50 clip GT val + caption của chúng bằng adapter `ft`, tính R@1 text→clip; phải khớp (±1) với số eval Omni đã có của tôi. Nếu lệch → dừng và báo.
4. Lưu ý quan trọng cần ghi vào `REPORT.md`: Omni lấy **8 frame** và **cắt audio về 8 s (center-crop)** cho mọi cửa sổ. Với R0 Global (cả video dài vài phút) nghĩa là audio chỉ còn 8 s ở giữa. Đây là thiết kế của Omni, giữ nguyên, nhưng phải nêu rõ vì nó ảnh hưởng R0.
5. `omni_collect.py` gom về `cache/omni/<tag>/{segments.npz, text.npz}` (L2-normalize, fp16), kiểm tra không thiếu id.

---

## 8. P4 — Exp 1, cột (b) rồi cột (a)

Không train gì. Với mỗi partition và mỗi tag:

- `S(q,V) = max_i cos(q, e_i)`; query = **toàn bộ caption val**, gallery = **toàn bộ video val hợp lệ**.
- Metrics T2V: R@1, R@5, R@10, MedR, MnR. V2T theo MeVTR (R@k-Average, One-Hit, All-Hit) nếu kịp.
- Thêm cột phụ: R1 Event-Single = mean các e_i của R6 rồi L2-normalize.
- Xuất `results/exp1_<tag>.csv` và bảng Markdown đúng format bảng Exp 1 của UEMR_FINAL, kèm cột `#vec` trung bình.
- Thêm bảng "logic đọc bảng" (mục 8 UEMR_FINAL): tính sẵn các hiệu số R6−R3, R6−R5, R7−R6, R7−UniformM@K_GT, R6−R1, R1−R0.
- Paired bootstrap (1000 lần, theo query) cho R6 vs R3, R6 vs R5, R6 vs R0 → p-value / CI.

**Thứ tự:** chạy xong `ft` (cột b) → báo cáo cho tôi → mới chạy `pt` (cột a). Sau cột (a), so sánh thứ hạng các dòng giữa (a) và (b) (Kendall τ).

---

## 9. P5 — Exp 3: chất lượng ranh giới (không fusion, tag `ft`)

- Làm nhiễu GT (dịch tâm + co giãn ngẫu nhiên, seed cố định) để tIoU trung bình với GT ≈ 1.0 / 0.9 / 0.7 / 0.5 / 0.3 (tìm biên độ nhiễu bằng bisection trên train, áp lên val).
- Encode các đoạn nhiễu bằng Omni (thêm vào manifest P3 nếu chạy chung được thì gộp luôn để tiết kiệm Colab).
- Vẽ R@1 theo tIoU; chấm thêm điểm UniAV-Pred (tIoU trung bình của nó với GT), Uniform-M, Global.

---

## 10. Chuẩn bị cho P6 (chưa làm, chỉ đảm bảo cache đủ)

Cache phải đủ cho Context Fusion mà không cần load Omni hay Athena lần nữa: `e_i`, `e_g`, `q`, `Z` level 0, `t2idx`, proposals, partitions, split. Viết `uemr/cache_loader.py` trả về một mẫu train đầy đủ cho 1 video và test nó.

---

## 11. Điểm dừng bắt buộc (hỏi tôi trước khi đi tiếp)

1. Sau Bước 0 (khảo sát).
2. Nếu teacher Omni của Athena dùng clip val, hoặc adapter Omni fine-tune đã thấy val.
3. Trước khi bắt đầu P3 trên Colab (gửi ước lượng chi phí + manifest đã gộp trùng).
4. Sau P4 cột (b), trước khi chạy cột (a).
5. Bất kỳ khi nào phải sửa code trong `ATHENA_ROOT` / `OMNI_ROOT`.

## 12. `REPORT.md` phải có

- Bảng sai khác so với `UEMR_FINAL.md` (InternVideo2 thay ONE-PEACE; Z level 0 nguồn nào; multi-domain COIN; teacher Omni; chọn ckpt Athena theo val; audio Omni 8 s; số video thực tế mỗi split).
- Checklist mục 14 của UEMR_FINAL, đánh dấu từng mục ✅/❌/⚠️ kèm bằng chứng.
- Kết quả P1 (detector), P4 (cột b, a), P5, kèm config hash.
- Danh sách việc còn lại cho P6+.