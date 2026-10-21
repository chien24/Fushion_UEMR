# Hoàn tất UEMR trên YouCook2: notebook bổ sung để lấp các lỗ hổng

Ngày lập: 2026-10-21. Tiếp theo `Result/uemr/YouCook_2/KET_QUA_UEMR_NEW_ATHENA.md` (Athena sạch `961f1a89`).

**Mục tiêu.** Tạo **một** notebook chạy được trên Colab, lấp các lỗ hổng mà hội đồng sẽ hỏi, rồi **chốt UEMR trên YC2** để chuyển
sang UEMR 2. Sau file này, UEMR trên YC2 không chạy thêm gì nữa.

**Phạm vi.** Chỉ YC2, chỉ UEMR. **Không** UEMR 2, **không** HT-Step, **không** train lại Athena hay Omni.

## 0. Đầu vào và đầu ra

|                      |                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Notebook cần tạo     | `notebooks/uemr/uemr_yc2_bo_sung.ipynb` (local: `D:\Học\KL\Code\Fushion\notebooks\uemr\`; tạo thư mục nếu chưa có)                  |
| Notebook tham chiếu  | `notebooks/uemr_final_AZ_clean.ipynb` (**AZ**), `notebooks/uemr_B1_uniav_athena_sach.ipynb` (**B1**): dùng lại hàm, cache, cách nạp |
| Athena               | `MyDrive/Colab Notebooks/code KL/Fushion/uemr/athena_sach/961f1a89/athena_yc2clean.pth`, assert sha256 `961f1a895f4b…643d58`        |
| Omni                 | `ft` = `checkpoint-1948`; `pt` = `omniretriever-7b` (như AZ)                                                                        |
| Protocol chính       | `configs/final_protocol.json`, sha `133125b9549bc74a`: **giữ nguyên**, assert sha                                                   |
| Protocol bổ sung     | `configs/bo_sung_protocol.json` (tạo mới ở việc 1, commit **trước** khi chạy Colab)                                                 |
| Kết quả cũ (chỉ đọc) | `uemr/results/961f1a89/`, `uemr/results/056d388e/`                                                                                  |
| Kết quả mới          | `uemr/results/961f1a89/bo_sung/` (Drive); cache encode mới theo khoá `seg_key` như AZ                                               |
| File kết quả         | `Result/uemr/YouCook_2/KET_QUA_UEMR_YC2_BO_SUNG.md` (mới) + cập nhật `KET_QUA_UEMR_NEW_ATHENA.md`                                   |
| File chốt            | `configs/uemr_yc2_final.json` + mục "Chốt UEMR YC2" trong file kết quả (đầu vào cho UEMR 2)                                         |

## Quy tắc

- Git: được `add`, `commit`; **không push**.
- Local: viết notebook, chạy được các phần CPU (việc 2, 3), viết kết quả. Colab A100: mọi phần cần GPU. Người dùng tự chạy Colab và gửi
  log **mọi ô**.
- Không ghi đè `results/961f1a89/`, `results/056d388e/` hay cache cũ. Không sửa `UniAV-fixed`, Omni, `final_protocol.json`.
- **Mọi lựa chọn tham số làm trên dev (104 video).** Val (394 video) chỉ chấm **một lần** cho mỗi cấu hình đã khoá trong
  `bo_sung_protocol.json`. Không chỉnh gì sau khi nhìn val.
- Bootstrap: paired theo video, 1.000 lần, seed 0, giống AZ. Mọi Δ đều báo kèm CI 95%.
- Số trong file kết quả phải lấy nguyên từ log. Số nào log không có thì ghi "không có trong log".
- Kết quả trái mong đợi vẫn ghi đủ. Không bỏ dòng, không đổi tiêu chí.

## Notebook: cấu trúc chung

1. **Thiết lập**: clone Fushion (in commit), mount Drive, assert sha256 Athena, sha của hai protocol, in phiên bản thư viện.
2. **Cờ chạy**: `RUN_CHECK`, `RUN_CI`, `RUN_BUDGET`, `RUN_R8`, `RUN_R5B`, `RUN_PT`, `RUN_AUDIO`, `RUN_CLIP`, `RUN_CLIPRET`, mặc định `True`.
3. **Ô ước lượng chi phí** (trước mọi bước encode): in số clip cần encode cho từng việc (đã trừ clip có sẵn trong cache), thời gian
   ước lượng dựa trên tốc độ đo được ở AZ/B1, và tổng. Nếu tổng > 10 giờ A100, in cảnh báo và gợi ý cờ nào nên tắt.
4. **Encode chạy lại được**: ghi cache lên Drive theo từng lô; bị ngắt thì chạy lại ô đó là tiếp tục. Ghi `failed.jsonl` cho clip lỗi.
5. **Kiểm tái lập đầu notebook** trên kho val 394, Omni `ft`: R0 8.32, R3 18.02, R2 27.16, R6 24.52, R7 37.52. Lệch là lỗi, dừng.
6. Mỗi việc ghi kết quả ra `bo_sung/<viec>/` (csv thứ hạng từng câu + json tổng hợp), in bảng cuối mỗi phần.
7. **Ô cuối**: `run_manifest.json` (commit, sha, cờ, số clip đã encode, thời gian từng phần) và bảng tổng hợp mọi việc.

## Việc 1: ghi trước tiêu chí (local, commit trước khi chạy Colab)

Tạo `configs/bo_sung_protocol.json` và ghi lại cùng nội dung vào đầu file kết quả. Ghi đủ:

1. Lưới tham số của việc 4 và 5, quy tắc chọn trên dev.
2. **Tiêu chí đổi phương pháp chính** (việc 5): phương pháp chính chuyển từ R6 MaxSim sang R8 **chỉ khi** trên val R8 − R6 có
   **cận dưới CI 95% > 0**. Không đạt → giữ R6 MaxSim.
3. Câu được phép viết về R8 so với R2 (việc 5):
   - cận dưới CI của R8 − R2 > 0 → "R8 vượt R2, với X vector/video so với 22.66";
   - CI chứa 0 → "không phân biệt được R8 và R2 về R@1" (**không** viết "tương đương" hay "bằng");
   - cận trên < 0 → "R2 vẫn cao hơn".
4. Câu được phép viết về đường cong ngân sách (việc 4): "R6 hơn cắt đều ở ngân sách K" chỉ khi cận dưới CI của R6_K − R3_K > 0 ở K đó.
5. Các việc 6–10 chỉ **mô tả**, không có tiêu chí thắng/thua, không đổi phương pháp chính.

## Việc 2: kiểm lại số (CPU)

1. **Tỉ lệ bỏ sót.** File kết quả ghi "R6 bỏ sót 27.16% event GT của val (AZ ô 12)", trùng đúng R@1 của R2 (27.16), trong khi bảng UniAV
   ghi 27.19. Tính lại cả hai định nghĩa trên val:
   - theo **event GT**: tỉ lệ event GT có max tIoU với mọi đoạn R6 < 0.3;
   - theo **câu hỏi**: tỉ lệ câu mà event GT của nó bị bỏ sót.
     In cả hai, đối chiếu với AZ ô 12 và ô 39. Nếu số trong file kết quả sai, ghi số đúng và sửa ở việc 11.
2. In lại K_pred trung bình và tIoU trung bình của R6 trên val (để dùng ở việc 4).

## Việc 3: CI còn thiếu (CPU, từ thứ hạng từng câu đã lưu)

Đọc thứ hạng từng câu trong `results/961f1a89/` và `results/056d388e/`. Nếu không có file thứ hạng, tính lại từ vector đã cache (không
encode mới). Báo Δ (mới − cũ, paired theo video) kèm CI cho:

- R6 R@5, R@10; R6 VCMR R@1 (tIoU 0.5); R6 R@1 trên kho 498.
- Với Athena sạch: R6 − R2 trên kho 498; R6 ∪ R2 − R2 trên kho 394 và 498.

## Việc 4: đường cong ngân sách vector (GPU: encode Omni `ft`)

**Câu hỏi:** khi cho cùng số vector mỗi video, cắt theo UniAV có hơn cắt đều ở **mọi** mức không, và R6 nhiều đoạn hơn có đuổi kịp R2
(22.66 vector/video) không?

1. **Tạo R6_K** với trung bình K ≈ 7.5 (R6 hiện tại), 10, 14, 18, 22 đoạn/video trên val. Đọc kỹ hậu xử lý UniAV trước:
   - nếu giảm θ đủ để đạt các mức K: quét θ;
   - nếu không: lấy top-K proposal theo điểm (sau NMS, trước ngưỡng θ).
     Ghi rõ cách nào được dùng. Mức K của từng biến thể chỉnh trên **dev** (K_pred trung bình dev gần mức mục tiêu nhất), rồi áp nguyên
     tham số đó cho val. In K trung bình thực tế trên val và tỉ lệ chồng lấn giữa các đoạn.
2. **Tạo R3_K**: cắt đều với **đúng số đoạn của R6_K ở từng video** (như `uni_M` hiện tại).
3. Encode Omni `ft` cho mọi đoạn mới (gallery 498 video; chỉ đoạn chưa có trong cache).
4. Báo bảng và vẽ đồ thị R@1 theo vector/video (val 394), gồm các đường R6_K, R3_K, và điểm R2, R6 ∪ R2, R7, R8 (từ việc 5):

   | K mục tiêu | K thực tế val | R6_K R@1 | R3_K R@1 | R6_K − R3_K (CI) | R6_K − R2 (CI) | VCMR R@1 R6_K |
   | ---------- | ------------- | -------- | -------- | ---------------- | -------------- | ------------- |

5. Kiểm tái lập: R6_K ở mức 7.5 phải ra 24.52, R3_K ở mức đó ra 18.02.

## Việc 5: R8, cắt theo UniAV rồi lấp khoảng trống (GPU)

**Lý do (ghi vào file kết quả):** ở bản Athena cũ, phần lớn khoảng cách R7 − R6 nằm ở các câu có bước bị UniAV bỏ sót; R6 ∪ R2 cao hơn
cả hai cho thấy hai cách cắt bổ sung cho nhau. R8 giữ ranh giới UniAV và chỉ thêm vector ở vùng UniAV không phủ.

1. **Định nghĩa R8:** đoạn R8 = mọi đoạn R6 + với mỗi khoảng thời gian **không đoạn R6 nào phủ** dài ≥ `L_min`, cắt khoảng đó thành các
   cửa sổ đều dài khoảng `L_fill` (số cửa sổ = max(1, round(độ dài / `L_fill`))). Đầu và cuối video cũng tính là khoảng trống.
2. **Lưới, chọn trên dev:** `L_min` ∈ {3, 5, 10} s, `L_fill` ∈ {7, 10, 14} s. Quy tắc: R@1 dev cao nhất; hoà (chênh < 0.3) chọn cấu
   hình ít vector hơn. Khoá cấu hình vào `bo_sung_protocol.json` (commit) **trước** khi chấm val.
3. Chấm val một lần cho cấu hình đã khoá: R@1/5/10, VCMR R@1, vector/video, độ phủ video, tỉ lệ bỏ sót (định nghĩa việc 2).
4. Báo: R8 − R6, R8 − R2, R8 − R6 ∪ R2 (CI); kho 498; đặt R8 lên đồ thị việc 4.
5. **Phân tích lỗi:** chia câu theo nhóm "bước bị R6 bỏ sót" / "không bỏ sót" (định nghĩa như bản cũ), báo R6, R8, R7 trong từng nhóm.
   R8 phải tăng chủ yếu ở nhóm bỏ sót; nếu không, ghi rõ.
6. Áp **tiêu chí đổi phương pháp chính** ở việc 1. In một dòng kết luận: `phương pháp chính YC2 = R6 MaxSim` hoặc `= R8`.

## Việc 6: R5b, biến thể đa dạng nội dung công bằng hơn (GPU)

R5 hiện tại chỉ lấy 2 s quanh mỗi khung đại diện (phủ 5.5% video), dễ bị hội đồng cho là baseline yếu có chủ đích.

- R5b: giữ đúng các medoid như R5 (K = số đoạn R6 của từng video), mỗi medoid thành một cửa sổ dài `thời lượng / K` đặt giữa medoid,
  cắt gọn trong video. Độ phủ xấp xỉ R3.
- Báo R5b R@1, độ phủ, R6 − R5b (CI). Trong file kết quả gọi là "biến thể kiểu đa dạng nội dung", **không** gọi là tái hiện MeVTR.

## Việc 7: cột Omni `pt` với Athena sạch (GPU nếu thiếu cache)

Kiểm cache `pt` có từ AZ ô 16 (6.106 clip) đã phủ những dòng nào; encode phần còn thiếu. Báo với Omni `pt`, kho val 394:

| Dòng | R0  | R3  | R6  | R7  | R2  | R8  |
| ---- | --- | --- | --- | --- | --- | --- |

và R6 − R0, R6 − R3, R7 − R6 (CI). Mục đích: trả lời "lợi ích có phải do Omni `ft` đã học clip cắt theo GT không".

## Việc 8: Exp 4, vai trò âm thanh (GPU)

1. Đọc code Omni để tìm cách encode **chỉ hình** đúng chuẩn mô hình (bỏ hẳn đầu vào âm thanh nếu mô hình hỗ trợ; nếu chỉ thay được
   bằng im lặng thì ghi rõ đó là xấp xỉ). Thử trên 5 clip trước, in vector có hữu hạn và khác bản có âm thanh.
2. Encode gallery không âm thanh với Omni `ft` cho R0, R3, R6, R7 (và R8 nếu qua tiêu chí).
3. Báo R@1 có / không âm thanh và Δ (CI) cho từng dòng; Δ của R6 − R3 khi có và khi không âm thanh.
4. Phía UniAV giữ nguyên (Athena dùng âm thanh, không train lại). Ghi đây là hạn chế: chỉ đo vai trò âm thanh ở phía Omni.

## Việc 9: backbone ngoài, zero-shot (GPU)

**Câu hỏi:** thứ tự R0 < R3 < R6 < R7 có phụ thuộc Omni không?

- Mô hình: CLIP ViT-L/14 (trọng số OpenAI, qua `open_clip`), vector đoạn = trung bình 8 khung lấy đều trong đoạn (khớp số khung của
  Omni), câu hỏi encode bằng text encoder, cosine.
- Dòng: R0, R2, R3, R6, R7, R8. Báo R@1 và R6 − R3, R6 − R0, R7 − R6 (CI).
- Ghi phiên bản `open_clip` và tên trọng số. Không tinh chỉnh gì.

## Việc 10: số để định vị với công trình khác (CPU nếu đủ cache)

Bài toán chính của UEMR (câu bước → video trong kho 394) không phải protocol chuẩn. Để có một con số gần protocol quen thuộc hơn:

- **Text → clip** trên các clip GT của val (3.030 clip, mỗi câu tìm đúng clip của nó trong toàn bộ clip val), cho Omni `ft`, Omni
  `pt`, CLIP. Báo R@1/5/10 và MdR.
- Ghi rõ khác biệt với protocol thường dùng trong các bài báo (chỉ 394 video val có trong dữ liệu, cách cắt clip, Omni `ft` đã học clip
  GT của trainsplit).
- **Không** điền số của bài báo khác từ trí nhớ. Để sẵn bảng "số đã công bố" với cột nguồn, chỉ điền khi có trích dẫn từ bài báo đã đọc;
  nếu chưa có, để trống và ghi "người dùng bổ sung".

## Việc 11: viết kết quả và chốt UEMR trên YC2 (local, sau khi có log)

### 11a. `Result/uemr/YouCook_2/KET_QUA_UEMR_YC2_BO_SUNG.md`

Cùng phong cách `KET_QUA_UEMR_NEW_ATHENA.md` (ghi nguồn số theo notebook và ô). Cấu trúc:

1. Thiết lập (commit, sha, cờ đã chạy, số clip đã encode).
2. Tiêu chí ghi trước (chép nguyên `bo_sung_protocol.json`, kèm commit đã ghi nó).
3. Kiểm tra: tái lập đầu notebook; kết quả việc 2.
4. Một mục cho mỗi việc 3–10: bảng số, rồi 2–4 câu nhận xét nói rõ số nào chứng minh.
5. **Trả lời các câu hỏi hội đồng có thể hỏi**, mỗi câu 2–4 câu, chỉ dùng số có trong log:
   - R2 cao hơn R6, vậy cắt theo sự kiện có ích gì? (việc 4, 5)
   - Đóng góp phương pháp là gì khi Context Fusion không giúp? (việc 5)
   - Lợi ích có do Omni `ft` học clip GT không? (việc 7, 9)
   - Âm thanh đóng vai trò gì? (việc 8)
   - Baseline kiểu đa dạng nội dung có bị làm yếu không? (việc 6)
   - Kết quả đứng ở đâu so với công trình khác? (việc 10)
6. Hạn chế còn lại.

### 11b. Cập nhật `KET_QUA_UEMR_NEW_ATHENA.md`

Sửa số sai nếu việc 2 phát hiện; thêm các CI của việc 3 vào cột "Đổi"; sửa mục 4b, 5, 6 cho khớp kết quả mới (đặc biệt nếu phương pháp
chính đổi sang R8). Không xoá số cũ; số đã thay thì ghi chú "(sửa ngày …, xem BO_SUNG)".

### 11c. Chốt UEMR trên YC2: `configs/uemr_yc2_final.json` + mục "Chốt UEMR YC2" cuối file 11a

Ghi đủ để UEMR 2 so sánh mà **không cần chạy lại UEMR**:

- Phương pháp chính (R6 MaxSim hoặc R8) và mọi tham số (θ, `L_min`, `L_fill`, cách gộp `max`, fp16).
- Checkpoint và sha (Athena, Omni `ft`), commit Fushion, sha hai protocol.
- Đường dẫn cache: đoạn, vector gallery, vector câu hỏi, thứ hạng từng câu (val 394, kho 498, dev).
- Số tham chiếu UEMR 2 phải so (dev và val 394): R@1/5/10, VCMR R@1 của phương pháp chính, R6, R2, R6 ∪ R2, R7; file thứ hạng từng
  câu để chạy paired bootstrap với UEMR 2.
- Danh sách việc **không** làm trên YC2 và lý do (HT-Step riêng; Athena không thầy Omni; Context Fusion mục 8b).

Sau 11c, ghi một dòng vào đầu `KET_QUA_UEMR_NEW_ATHENA.md`: "UEMR trên YC2 đã chốt ngày …, xem `configs/uemr_yc2_final.json`."

## Điều kiện dừng và báo người dùng

- Kiểm tái lập đầu notebook lệch.
- Việc 4 không đạt được mức K ≥ 14 bằng cả hai cách (θ, top-K): làm tới mức cao nhất đạt được, ghi rõ, báo người dùng.
- Ước lượng chi phí > 10 giờ A100: báo trước khi bàn giao notebook, kèm đề xuất bật/tắt cờ.
- Việc 8 không tìm được cách bỏ âm thanh hợp lệ: báo, không tự chọn cách xấp xỉ khác ngoài im lặng.
- Bất kỳ kết quả nào làm thứ tự R6 > R3 hoặc R6 > R0 đổi (ở Omni `ft`, kho 394): dừng viết, báo người dùng.

## Bàn giao

1. **Trước Colab:** đường dẫn notebook; commit chứa `bo_sung_protocol.json`; bảng ước lượng chi phí theo cờ; các ô nào chạy được trên
   CPU.
2. **Sau khi có log:** danh sách commit để push; đường dẫn `KET_QUA_UEMR_YC2_BO_SUNG.md` và `configs/uemr_yc2_final.json`; 5–7 câu tóm
   tắt, gồm một dòng "phương pháp chính YC2 = …" và một dòng "UEMR 2 so với: …".
