# Vendored: Athena (UniAV, nhánh `ov-refine`)

- Nguồn: `github.com/im-xiaoming/UniAV-fixed` (README gọi là `Athena-Retriever`), nhánh `ov-refine`.
- Commit: **`48918c5db3c36511c3c62f808a4dc33760cff5cc`** (2026-10-07, "Rename to Athena: package athena, train.py, ...").
- Cách copy: `git cat-file blob 48918c5:<path> > third_party/athena/<path>`. Nội dung giống từng byte với commit,
  kể cả xuống dòng LF (bản checkout trên Windows của repo gốc dùng CRLF vì `core.autocrlf=true`, nên không copy từ đó).
  Đã kiểm tra bằng `git hash-object --no-filters <file>`, kết quả trùng blob của commit cho mọi file.
  `third_party/athena/.gitattributes` (`* -text`) cấm git đổi xuống dòng trong thư mục này: `LICENSE` gốc dùng CRLF,
  và `core.autocrlf=true` trên Windows đã từng biến nó thành LF lúc commit (1070 thay vì 1091 byte).
  Test `uniav_app/tests/test_vendored.py` kiểm tra lại kích thước và SHA256 trên Colab.
- Giấy phép: MIT (`LICENSE`, Copyright (c) 2024 Tiantian Geng).
- Cấu trúc thư mục được giữ nguyên để hai chỗ sau vẫn đúng: `athena/config.py::ROOT` (= thư mục cha của gói
  `athena`) và `athena/encoders/internvideo2.py::_extract_module` (nạp `ROOT/tools/extract_internvideo2.py` theo đường dẫn).

## Không vá

**Không file nào bị sửa.** Mọi đường dẫn khác với mặc định của repo đều được truyền qua các field của
`athena.config.Config`: `checkpoint`, `caption_pool`, `index_dir`, `iv2_video_encoder`, `iv2_audio_encoder`,
`iv2_repo`, `device`, `use_seg_weights`. Việc này làm trong `uniav_app/athena_api.py`.

## Model nào làm gì (checkpoint API `athena.pth`, run `ov_E1`)

| Việc | Trọng số | Gọi qua |
|---|---|---|
| Phân đoạn (proposal, soft-NMS) và chuỗi Z | `state_dict_seg` (epoch best_seg) → `AthenaPipeline.seg_model` | `seg_model(fv, fa)`, `seg_model._last['feats'][0]` |
| Caption cho đoạn | `state_dict` (epoch best_cap) → `AthenaPipeline.model` | `model(fv, fa)` rồi `model.span_vectors(g)` và `captioner.caption(vecs)` |
| Sanity check `final_eval` | `state_dict` cho mọi việc (`use_seg_weights=False`) | như trên |

Đây đúng là cách chia của `AthenaPipeline.process_features`.

## Danh sách file (SHA256)

| File | Byte | SHA256 | Dùng cho |
|---|---|---|---|
| `LICENSE` | 1091 | `d10e78ca0b8e7b1c0e7d6fd7576ebd54284e15071c1e132161c382574f5b2d37` | giấy phép |
| `athena/__init__.py` | 4075 | `780a202727ec8e4a1097bd5b913f6bcb3249f29075b0c9d89a3fa5d39fa21452` | import gói |
| `athena/config.py` | 3887 | `9f76d200a64046e0d7e6fa691e105f3b5a7e4bdb707834ca9d99303643f14a04` | `Config`, `pick_device` |
| `athena/pipeline.py` | 22316 | `1143079d36a6ae3c47a4396c5a026d1f7338123738211884d681ba39c62d9dbc` | `AthenaPipeline`, `select_events` |
| `athena/features.py` | 3652 | `a1ecb898b71a84a38eb9f9567a13fb67a8760dd184dc2244b4413fe54e309b6e` | `FeatureSpec` (prepare, to_seconds) |
| `athena/captioner.py` | 5569 | `cf7e1eab35f4d9031c818935f372c5294486733d8a12e2066907a03acba46bd0` | `Captioner` (chọn caption MBR) |
| `athena/postprocess.py` | 2345 | `834ba4c88e6d132531f0e28502c1f5fb8d4249eb6418d419eb7e59aa7934900c` | soft-NMS numpy |
| `athena/report.py` | 6877 | `755bbc9f624e81a0fd51642690016f0c2fa83fdd00d305f6352d6e60f62488f4` | được `__init__` import |
| `athena/calibrate.py` | 3863 | `5e424bef8ac26fee67315508ddba6c50f91a478879e7ad8e2d49e82074df4c24` | `match` (F1 tham chiếu 0.535 trong sanity) |
| `athena/model/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | |
| `athena/model/backbone.py` | 8403 | `4e428311dc8c6d40366a3449885e45b9c8720e3feed8d332b641304839f3c772` | backbone pyramid |
| `athena/model/blocks.py` | 14695 | `64a0427c8d6d78b25de4855b751ed65954baa3e9625627c2455541f35860c642` | khối conv/transformer |
| `athena/model/event_model.py` | 14674 | `eacaa6d2dd952378c144038b261bfa70ff6c4a5e98c9b73cfb4d96babe8945d5` | `EventCaptionModel` |
| `athena/model/point_generator.py` | 2582 | `84239ecd4190f413ae06d1a603a80d6389b7e04945a6c642c328de888f1d8173` | điểm neo |
| `athena/encoders/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | |
| `athena/encoders/internvideo2.py` | 4822 | `faa837738d5350582872327009e413725700c2c8806ec4ae9e2e9d8399c806ce` | `InternVideo2AVEncoder` (đường `'extract'`) |
| `athena/encoders/media.py` | 1447 | `378e371e40b411063b4df17cedf3e660c18ee32fc36f28cfc0d95c613625329b` | `find_ffmpeg`, `probe` |
| `athena/encoders/internvideo2_text.py` | 2894 | `a90a05c28794461603de94927800b5fa8fbad1d95d571693815297289648858b` | tháp text (app không dùng; `pipeline.py` import lazy) |
| `athena/assets/caption_pool.npz` | 8002990 | `7ac57ce9056538c59e4938cc4722d806bb146c1483ebf2b1f0a0e3e6d7830030` | 8218 câu train (danh sách ứng viên) |
| `tools/extract_internvideo2.py` | 11396 | `8121e4adab769a1f365b094377f640f26a7882827398af4c1076250526352b90` | giải mã + encoder (đường `'extract'`) |
| `data/youcookii/annotations/youcookii_annotations_trainval.json` | 1575867 | `6be6f3cdcf4448b4bdef27a869fadddfddeb9b20249d55558afabe354b246e80` | GT cho sanity `final_eval` |
| `data/youcookii/caption_emb_iv2j.npz` | 11326587 | `6755d6cb786e6988819e490daeb496dc340d060b0aa1d8e66fa62175eb9ff490` | vector câu cho ORACLE vocabulary |
| `athena/samples/-Ju39A-G0Dk.npz` | 1862395 | `28b1dfd7cfb6af2b3d2b1a4a0fa06f68eeebfc17e30d3ee19c9c33b2820f3111` | SMOKE / parity |
| `athena/samples/6uHoTJSLoL8.npz` | 689737 | `3f3e28b4fe1f13e459a6330a373b7deb134e1c0ee3ee517696b0182af4439457` | SMOKE / parity |
| `athena/samples/9GX8f5EwwE4.npz` | 917190 | `6e1da2571a0cff8ebaa32368193c144c41b4f357cffd09a2629e974dce36ef81` | SMOKE / parity |
| `athena/samples/SOMsxGGSTUk.npz` | 1211272 | `e0475edd87ad792ce89fc85eb77516b615fdfa302dacc911c3d727ef46f5d9d6` | SMOKE / parity |
| `athena/samples/W2gnFLOi_AQ.npz` | 1134351 | `0e65027998997bf113554eb3cdb961106131cbdd6b7a13d07c75ddfdbb6d2fa8` | SMOKE / parity |
| `athena/samples/XEifm-iXMvs.npz` | 1099709 | `0894647cd3786b3a6627f1e783e1556199006a807ebe40126fa70228b4b9f315` | (không dùng mặc định) |
| `athena/samples/cMzyB4m3VHY.npz` | 837795 | `cb1e253ccfb65563ee2e7fd78f6089f07f10114d0a02da057a59c9d471ea0244` | (không dùng mặc định) |
| `athena/samples/e8S1vFC8zYk.npz` | 580514 | `7965e9db7f7f36e2a39ab63d978b2297e7250de52b1d4d0ec42dc329bd0b2941` | (không dùng mặc định) |

Thư mục `athena/samples/` được thêm so với danh sách ở bước 0, vì SMOKE và cell parity cần nó
(`TRA_LOI_BUOC_0_UNIAV.md`, mục 2.4 và 2.5). Copy cả 8 file như trong repo; SMOKE dùng 5 file đầu theo thứ tự tên.

## Không copy

`athena/generator.py` (bộ sinh GPT-2: checkpoint cũ, không có; `pipeline.py` chỉ import lazy khi
`caption_mode='generate'`), `athena/demo*.py|ipynb`, `libs/`, `train.py` (chỉ để train), các checkpoint.
Upstream `OpenGVLab/InternVideo` không copy: notebook `git clone --depth 1` khi cần đường `'extract'`.
