# UEMR chốt, file 01: train lại UniAV với thầy Omni `best`

Đọc `UEMR_CHOT_00_TONG_QUAN.md` mục 3, 4 trước. Mục tiêu: một UniAV HT-Step **chỉ khác** checkpoint hiện tại (`model_best.pth.tar`,
sha256 `a7c46353…`) ở vector thầy: vector thầy bây giờ là Omni `best`, tính bằng **đúng đường encode của cột (b)**.

---

## 1. Đọc định dạng dữ liệu thầy của HanLi (local, không GPU)

HanLi `77fc4fa`: `libs/datasets/htstep.py:46-48` (thư mục train / val), `:193-202` (gán vector thầy cho đoạn GT), `libs/core/config.py:59-63`
(loss), config yaml của checkpoint. Thêm `config.json` và một file mẫu của dataset `htstep-omni-av` cùng `htstep-val-seen-feats/omni_av/`.

Ghi vào Nhật ký: định dạng file (một file / video?), khoá của đoạn (giây hay chỉ số bước, làm tròn ra sao), dtype, có chuẩn hoá không, cách
dataset xử lý đoạn **thiếu** vector thầy, số đoạn và video thầy cũ phủ (train, val).

**CHECK:** dump `cfg` trong `model_best.pth.tar`, so với yaml trong repo; khác thì dùng `cfg` của checkpoint và ghi lại.

## 2. Danh sách đoạn cần vector thầy

- **Train:** đúng các đoạn GT mà `htstep.py` sinh cho 5.826 video train của UniAV, cùng luật lọc. Đếm.
- **Val:** đúng các đoạn GT cho 510 video kho (thầy cũ chỉ phủ 179 video; bản mới phủ đủ 510).
- **Nguồn clip:**
  1. Clip cắt sẵn `GraceFeel/htstep-omni-clips` (đường đã khớp `val_embeds_best.npz` ở lượt dev 1), khi cửa sổ clip trùng đoạn (lệch
     ≤ 0.01 s).
  2. Không có clip thì cắt từ mp4 (`nguyenminh04/htstep-videos`, tải theo tar, resume, kiểm đĩa, xoá tar), encode bằng `timestamps`.
  3. Không đọc được mp4 thì đánh dấu thiếu.
- Cho val: dùng lại cache R7 của lượt dev 1 khi khoá trùng. Không trùng thì encode.
- **CHECK:** tỉ lệ đoạn thiếu ≤ 5 %, không thì dừng, báo. Theo cách dataset xử lý đoạn thiếu (mục 1), ghi rõ đoạn thiếu bị bỏ khỏi loss
  chưng cất hay gì khác. Nếu dataset không xử lý được đoạn thiếu, user sửa HanLi tối thiểu (mặt nạ cho loss chưng cất), commit riêng, ghi
  Nhật ký.

## 3. Encode (Colab A100)

- `omni_retrieval.encode` của Fushion (decord, 224 px, 256 token; không giới hạn 500 MB, commit `bfd10de`), adapter `best`, fp16, L2-norm
  như cột (b). Cache theo khoá đoạn, ghi theo lô, resume, `failed.csv`.
- **CHECK trước khi encode cả tập:** 20 clip GT val, cos với `val_embeds_best.npz` ≥ 0.999 (trung bình và từng clip); 20 clip train encode
  hai lần cho cos 1.0. Mp4 cắt bằng `timestamps` so với clip cắt sẵn trên 10 đoạn: ghi cos (chỉ báo).
- Ước lượng: số đoạn × 0.32 s. In trước khi chạy.

## 4. Ghi thành dataset thầy mới

- Đúng định dạng mục 1, chỉ thay vector. HF dataset `GraceFeel/htstep-omni-av-best` (hoặc Drive). `config.json` ghi: adapter `best`
  sha256 `5e9899aa…`, đường encode, commit Fushion, số đoạn, số đoạn thiếu, sha256 từng file.
- **CHECK:** đọc lại bằng chính `htstep.py` (một batch train, một batch val): shape đúng, không NaN, norm ≈ 1.

## 5. Train UniAV (user chạy, code HanLi)

- Cùng `cfg` của checkpoint cũ: kiến trúc, loss và trọng số (nhiệt độ 0.05, KL 0.2, cos 2.0), lr, số epoch, batch, seed, EMA, luật chọn
  checkpoint (epoch mAP tốt nhất trên val_seen, như cũ). **Chỉ đổi đường dẫn vector thầy** (train và val).
- Lưu `model_best.pth.tar` mới, sha256, log train (loss từng phần theo epoch), commit HanLi đã dùng.
- Không thử cấu hình khác. Nếu train hỏng (loss NaN, mAP val ở chế độ article GT < 20 sau khi xong), dừng và báo; khi đó file 02 dùng
  UniAV cũ cho đoạn (xem §6).

## 6. Kiểm UniAV mới (local CPU được)

Bằng adapter của Fushion, trên 510 video:

| Chỉ số | UniAV cũ (đã đo) | UniAV mới |
|---|---|---|
| mAP@[0.3:0.7], article GT | 29.70 | |
| mAP@[0.3:0.7], article đoán | 19.55 | |
| ArtAcc | 60.20 | |
| Recall bước GT, tIoU 0.5, article đoán, θ theo luật | 31.74 | |
| OmniCos / OmniCosC / OmniR@1, đoạn GT, article GT, so với vector `best` val | (thầy cũ: 91.06 / 75.05 / 59.72) | |
| OmniCos / OmniCosC, đoạn GT, **article đoán** | chưa đo | |

- OmniCosC trừ vector trung bình của vector thầy **train** mới (cách tính theo model card; agent đọc code để lấy đúng công thức).
- Unit test adapter chạy lại trên checkpoint mới: đổi `cand` GT thì đầu ra chế độ chính không đổi.
- **Luật chọn nguồn đoạn** (ghi trước, dùng ở file 02): đoạn R6 lấy từ UniAV có **mAP article đoán cao hơn** trên dev (mới hay cũ). Vector
  `h_i` **luôn** lấy từ UniAV mới (span head chạy trên đoạn bất kỳ). Mặc định là mới.
- Ghi `results/uemr_final/uniav_best_checks.json` + sha256 checkpoint.

## 7. Nhật ký

(Agent / user ghi: kết quả mục 1, số đoạn train / val / thiếu, kết quả CHECK, sha256 dataset và checkpoint, commit HanLi, thời gian encode
và train, bảng mục 6.)

### 2026-11-12 (agent, local, không GPU): §1, §2 và notebook cho §3–4

Nhánh `htstep`. Commit Fushion: `0a8f2aa` (`uemr_htstep/teacher_best.py`, `omni_retrieval/encode.py`: `OMNI_QWENVL_DIR`), `f921c65`
(`notebooks/uemr/uemr_chot_01_teacher_best.ipynb`, `FUSHION_REF = cdb2fac`). Chưa push. Unit test: 41 pass (`tests/`, có `UNIAV_REPO`).

**§1. Định dạng thầy của HanLi `77fc4fa`** (đọc `libs/datasets/htstep.py`, `libs/datasets/helper.py::load_omni_segments`,
`libs/modeling/event_archs.py::omni_losses`, `config.json` + README của `nguyenminh04/htstep-omni-av`, 179 file `omni_av/` val ở local):

- Một file / video: `<omni_emb_folder>/{train,val}/<video_id>.npz`. `htstep.py:46-48` chọn `train` nếu split có `train`, không thì `val`.
- Khoá của đoạn: `seg_idx` = **chỉ số của đoạn trong list `annotations` của json** (không phải giây, không làm tròn). File còn có `step_id`
  (`global_step_index`), `segment` (float64, giây, đúng như annotation, không cắt theo đặc trưng), `partial`.
- `emb` (k, 3584) float16, đã L2-norm khi ghi; loader chuẩn hoá lại (`vector / max(norm, 1e-6)`), float32.
- Đoạn **thiếu** vector thầy: `omni_ok[slot] = False`; `omni_losses` chỉ lấy teacher có `omni_ok` (`teachers = omni_seg[omni_ok]`, point head
  và span head đều lọc `target >= 0`). Video không có file thì mọi slot `False`. Tức đoạn thiếu **bị bỏ khỏi cả hai loss chưng cất**
  (KL và cos), các loss khác không đổi. **Không cần sửa HanLi.** `seg_idx` mà dataset đã loại (đoạn bắt đầu sau khi đặc trưng hết) bị bỏ qua.
- Thầy cũ phủ: train 5.465 / 5.826 video (đếm file của `nguyenminh04/htstep-omni-av` qua API HF; README của nó: 5.644 / 6.005 id, 34.785
  bước); val 179 / 510 video, 1.080 đoạn (đếm local). Thầy cũ: `encode_av`, PyAV, 336 px, cửa sổ = đoạn annotation, bỏ đoạn end ≤ start.

**CHECK cfg của checkpoint: KHÔNG làm được như plan viết.** `model_best.pth.tar` (sha256 `a7c46353…`) chỉ có khoá `epoch` (= 9),
`state_dict`, `metrics`; **không có `cfg`**. Thay bằng: `config.py` đi kèm checkpoint trên HF (`nguyenminh04/uniav-htstep-omni`) trùng
nội dung (sau khi bỏ CR) với `libs/core/config.py` tại `77fc4fa`; model card ghi "`python train.py` with the defaults of `config.py`:
12 epochs (early stop), one RTX 3060, best mAP at epoch 9". Repo không có yaml. Kết luận dùng cho §5: train bằng mặc định của
`config.py` tại `77fc4fa`, chỉ thêm `--set "dataset.omni_emb_folder='…'"`. Không có log nào cho biết lượt cũ có `--set` khác hay không
(user / người train xác nhận).

**§2. Danh sách đoạn** (`uemr_htstep.teacher_best`, đếm trên `annotations.json` sha `e40d8a2b…`, manifest precut, `tar_index.csv`):

| Tập | Video | Đoạn | cache dev 1 | clip cắt sẵn | mp4 + timestamps | thiếu (không có mp4) |
|---|---|---|---|---|---|---|
| train | 5.826 | 36.026 | 0 | 33.326 | 379 (154 video) | 2.321 (361 video) |
| val (kho) | 510 | 2.990 | 2.990 | 0 | 0 | 0 |

- Danh sách = **mọi annotation** của video (không video nào > 48, không đoạn nào end ≤ start). Luật lọc của `htstep.py` (bỏ đoạn bắt đầu
  sau khi đặc trưng hết, 28 / 38.957 theo comment trong code) cần file đặc trưng nên không áp ở đây: danh sách là tập cha, loader tự bỏ
  qua dòng thừa.
- 379 đoạn mp4: 252 đoạn thuộc video precut đã loại (`empty_video_track` 18 video, `transcode_mismatch` 13, `past_end` 3, …), 127 đoạn
  thuộc video có clip nhưng bước đó không cắt được hoặc cửa sổ clip lệch đoạn > 0.01 s. Nhiều đoạn trong số này sẽ lại lỗi khi encode.
- Val: cả 2.990 đoạn GT đã có vector Omni `best` trong cache lượt dev 1 (dòng R7, `segments.npz` sha `16636ad1…`), không cần encode.

**CHECK tỉ lệ đoạn thiếu ≤ 5 %: FAIL (đã biết trước khi encode). DỪNG, chờ user quyết định.** Cận dưới của tỉ lệ thiếu ở train là
2.321 / 36.026 = **6,44 %**: 361 video train của UniAV có đặc trưng nhưng **không có mp4** trong `nguyenminh04/htstep-videos`
(`htstep_split_omni.json`: `no_mp4`), nên không có clip và không cắt được. Cộng thêm đoạn mp4 lỗi thì cao nhất khoảng 7,5 %. Thầy cũ
cũng thiếu **đúng 361 video này** (5.465 file), tức UniAV cũ đã train với cùng tập video không có thầy; bộ thầy mới không làm tình
trạng này tệ hơn và điều kiện "chỉ đổi thầy" vẫn giữ. Notebook để `MISSING_MAX_PCT = 5.0` đúng plan và sẽ dừng ở ô 3 khi `SMOKE = False`
(SMOKE chỉ in cảnh báo để thử được đường chạy). Việc cần user chọn: (a) chấp nhận mức thiếu này (đổi `MISSING_MAX_PCT`, ghi lý do ở
đây), hoặc (b) tìm mp4 của 361 video.

**Chỗ plan không khớp code (đã xử lý, ghi để user biết):**

- Plan §2 ghi clip cắt sẵn là "đường đã khớp `val_embeds_best.npz` ở lượt dev 1". Thực tế lượt dev 1 encode từ **mp4 + timestamps**
  (`omni_retrieval.encode`), và phép kiểm cos ≥ 0.999 khi đó là mp4 so với `val_embeds_best.npz` (bản này do eval của Omni tạo từ clip
  cắt sẵn). Loader vendored của Fushion (`third_party/omni_fix`, Omni-fix `460257f`) **không đọc được record `clip`**. Notebook dùng loader
  của Omni-fix `ab772eb` (nhánh `htstep_lora`: bản vendored + 33 dòng nhánh `clip`, phần model và record video không đổi) qua biến môi
  trường mới `OMNI_QWENVL_DIR`; code Omni không sửa. Ô 6 kiểm lại đường này: 20 clip GT val so với `val_embeds_best.npz` ≥ 0.999.
- Val lấy vector từ cache dev 1 (mp4, cửa sổ làm tròn 0.01 s), train lấy từ clip cắt sẵn (cửa sổ đúng annotation). Ô 6 in cos giữa hai
  đường trên 20 clip val và 10 clip train (chỉ báo).

**Notebook `uemr_chot_01_teacher_best.ipynb`** (chưa chạy): ô 3 danh sách + CHECK số đếm ở bảng trên; ô 5 tải clip theo shard (32 shard,
32,2 GB clip cần dùng), kiểm sha256 từng clip, tải mp4 của 154 video; ô 6 ba phép kiểm của §3; ô 7 encode, store trên Drive, resume,
`failed.csv`; ô 8 ghi `train/<id>.npz`, `val/<id>.npz`, `config.json` (sha256 từng file); ô 9 đọc lại bằng `make_dataset('htstep', …)`
của HanLi, một batch 32 video train + một batch val (đặc trưng lấy từ hai tar mẫu `htstep_feats_1000`, `_1006`); ô 10 tar lên Drive,
upload `GraceFeel/htstep-omni-av-best` (private). Bộ thầy đặt đúng layout `omni_emb_folder`, nên train không cần sửa HanLi.
Ước lượng: 33.705 đoạn × 0,32 s ≈ 3,0 giờ encode; cả notebook khoảng 3,7 giờ A100 (SMOKE khoảng 25 phút).

Còn trống, điền sau khi có log Colab: kết quả CHECK ô 6, số đoạn lỗi, sha256 dataset, thời gian encode; §5–6 (user train, kiểm UniAV).

### 2026-11-12, quyết định của user (trước khi encode)

1. **Đoạn thiếu: chấp nhận, `MISSING_MAX_PCT = 8.0`** (plan §2 ghi 5 %). Lý do: 361 video train của UniAV không có mp4 trong
   `nguyenminh04/htstep-videos` (2.321 / 36.026 đoạn = 6,44 %); thầy cũ cũng thiếu đúng các video này; HanLi bỏ đoạn thiếu khỏi loss chưng
   cất bằng `omni_ok`. Notebook: ô 0 đặt 8.0, ô 3 kiểm cận dưới trước khi encode, ô 8 kiểm tỉ lệ thật sau khi encode (cùng ngưỡng).
2. **cfg của checkpoint:** thay phép "dump `cfg`" bằng phép dựng model. Local CPU, HanLi `77fc4fa` (working tree sạch): dựng
   `EventCaptionTransformer` bằng `cfg['model']` mặc định của `libs/core/config.py`, `set_step_pool` như `train.py`, rồi
   `load_state_dict(state_dict của model_best.pth.tar, strict=True)`:

   ```text
   sha256 checkpoint: a7c46353bf6635ad33d9cfecb04eb8ca275dc8fb4eab40d62457aef5c70c2616
   sha256 config.py (git blob 77fc4fa = file config.py trên HF): 21c27e7c2381c9043d0d0d0899b596fb52931336d653847111dd3e08663eb1ca
   load_state_dict(strict=True): <All keys matched successfully>
   tensors: checkpoint 544 | model 544 | same keys True | same shapes True
   parameters: 123917332 | epoch 9
   ```

   Phép này xác nhận **kiến trúc** (mọi khoá và shape) trùng config mặc định. Nó không nói gì về các tham số không nằm trong
   `state_dict`: trọng số loss, nhiệt độ, lr, batch, số epoch, seed, EMA. Phần đó dựa vào `config.py` trên HF trùng `77fc4fa` và câu
   "`python train.py` with the defaults" của model card; user hỏi người train cũ về `--set`.
3. **`OMNI_QWENVL_DIR`: đồng ý.** Ô 6 giữ CHECK cos ≥ 0.999 (từng clip, nên cả trung bình) giữa clip cắt sẵn và `val_embeds_best.npz`;
   FAIL thì dừng trước khi encode cả tập.
4. `docs/uemr_chot/` được `git add -f` và commit (5 file plan + `HUONG_DAN_CHAY.md`).

### 2026-11-12, kết quả A1 (SMOKE) và A2 (lượt thật), notebook `uemr_chot_01_teacher_best.ipynb`

Log: `notebooks/uemr/log/uemr_chot_01_teacher_best_A{1,2}.pdf`. Fushion `cdb2fac`, Omni-fix `ab772eb`, HanLi `77fc4fa`, A100-SXM4-40GB.
A1 in "SMOKE xong: mọi CHECK OK". A2: mọi CHECK OK, không FAIL.

- Ô 3: số đoạn đúng bảng §2 (36.026 / 33.326 clip / 379 mp4 / 2.321 không có mp4); val 2.990 đoạn trong cache dev 1.
- Ô 5: 33.336 clip, sha256 trùng `precut_index`.
- Ô 6 (CHECK §3):
  - clip cắt sẵn so với `val_embeds_best.npz`, 20 clip GT val: cos mean **0,99991**, min **0,99986** (≥ 0.999, OK);
  - cùng 20 clip so với cache dev 1 (đường mp4, cửa sổ làm tròn 0,01 s): mean 0,99873, min 0,99299 (chỉ báo);
  - 20 clip train encode hai lần: cos min 1,000000 (OK);
  - 10 đoạn mp4 + `timestamps` so với clip cắt sẵn: mean 0,99986, min 0,99978 (chỉ báo).
- Ô 7: 22.596 đoạn còn lại sau lượt bị ngắt trước đó (store resume), 22.425 encode được, **171 lỗi, đều nguồn mp4** (decord
  "Cannot create buffer source" ở video `empty_video_track`, "past the end" ở video mp4 ngắn hơn annotation, đúng các video precut đã
  loại); 0,236 s/clip. Thời gian encode cộng dồn 8.062 s (2,2 giờ).
- Ô 8: train 5.431 file, 33.445 đoạn có vector, **2.581 thiếu = 7,16 %** (2.321 không có mp4, 89 mp4 không tải / chuyển được, 171 encode
  lỗi), CHECK ≤ 8 % OK. Val 510 file, 2.990 đoạn, 0 thiếu.
- Ô 9: đọc lại bằng `htstep.py`: train 32 video / 194 đoạn, val 32 video / 185 đoạn, mọi đoạn có thầy, norm 1 ± 1e-7, sai khác 0. OK.
- Ô 10: `GraceFeel/htstep-omni-av-best` (private) đã upload. `config.json` sha256
  `f9abbce3876926d96d531de27e5469078d4f728c49ff5b0de2378df9335512b3`; tar trên Drive
  `uemr_chot_run/teacher_best/htstep-omni-av-best.tar` sha256 `ec697aa00187dcf5129e7e915334574a688b8da12bb420c359aa69df612d6b3a`.

Ghi chú: vector thầy val (cache dev 1, mp4) và train (clip cắt sẵn) đi hai đường đọc khác nhau, lệch trung bình 1 − 0,99873 trên 20 clip.
Chỉ ảnh hưởng tới số OmniCos / chọn checkpoint trên val, không ảnh hưởng tới loss train.

§5: notebook `notebooks/uemr/uemr_chot_01_train_uniav.ipynb` (A100). Bộ thầy mới đặt ở `datasets/omni_av_feats` (đường mặc định), nên
lệnh là `python train.py --output htstep_best` với `config.py` mặc định, không `--set`; `num_workers` để mặc định (2) như lượt cũ trên
RTX 3060 (`colab_train.sh` cũ dùng 8, thứ tự crop ngẫu nhiên phụ thuộc số worker).

