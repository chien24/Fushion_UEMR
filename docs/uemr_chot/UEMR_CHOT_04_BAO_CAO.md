# UEMR chốt, file 04: báo cáo cuối

Đọc `UEMR_CHOT_00_TONG_QUAN.md` mục 2, 7 trước. Viết local sau khi user gửi log và kết quả của file 01–03.

File: `Result/uemr_final/KET_QUA_UEMR.md`. Văn phong như báo cáo HT-Step lượt dev 1: câu ngắn; mỗi giải thích ghi **[đã kiểm chứng]** hoặc
**[giả thuyết]**; mọi số có nguồn (ô notebook hoặc file); hiệu tự tính ghi "tính từ bảng". Không dùng và không so số YouCook2 cũ.

## Khung

1. **Tóm tắt:** câu kết luận của từng benchmark (file 03 §8), phương pháp chính, 3–4 số chính kèm CI.
2. **Bài toán và vai trò dữ liệu:** HT-Step train / dev, benchmark zero-shot, vì sao tách như vậy.
3. **Phương pháp:** UEMR (file 00 §2): cắt, lấp, hai vector cho mỗi đoạn, bốn biến thể, MaxSim, khoảnh khắc. Hình vẽ pipeline. Giải thích
   ngữ cảnh đi vào qua vector trò.
4. **Huấn luyện UniAV với thầy `best`** (file 01): dữ liệu thầy, độ giống thầy (OmniCos / OmniCosC / OmniR@1, article GT và article đoán),
   chất lượng cắt mới so với cũ.
5. **Kiểm tra độ tin cậy:** rò rỉ article và cách tránh; tái lập; không giao train; một lượt chấm benchmark; kiểm đặc trưng benchmark.
6. **Dev HT-Step** (ghi rõ là dev): bảng file 02 §8, các lựa chọn (θ, `L_min`, `L_fill`, β\*, phương pháp chính) và số cấu hình đã thử
   (`tried.jsonl`); phân tích nhóm; C3; lượt dev 1 (thầy cũ) làm tham chiếu.
7. **Benchmark:** mỗi bộ một mục: kiểm kê, chất lượng cắt, bảng chính, bảng tiêu chí, câu kết luận, pt so với `best`.
8. **Chi phí:** số vector / video, kích thước chỉ mục, thời gian phía video của bốn biến thể.
9. **Ví dụ định tính:** 3–4 câu lấy từ thứ hạng đã lưu (một câu UEMR thắng R0, một câu bước bị bỏ sót, một câu VCMR đúng, một câu thất bại),
   kèm khung hình, đoạn trả về, đoạn GT.
10. **Đóng góp:** đúng 4 ý ở file 00 §7. Ý 2 chỉ viết "có hiệu quả" nếu K4 hoặc K5 đạt ở ít nhất một benchmark.
11. **Hạn chế** (tối thiểu):
    - Checkpoint của cả hai model được chọn trên dev HT-Step; số dev lạc quan; kết luận chỉ từ benchmark.
    - UniAV học trên ít video hơn Omni (5.826 so với 6.993); vector thầy thiếu cho một phần đoạn (số thật, file 01).
    - UniAV có bảng article đóng của HT-Step; trên benchmark, article đoán là một task HT-Step gần nhất. Lúc train UniAV luôn được cho
      article GT, lúc dùng phải đoán (mAP giảm, số của file 01 §6). Chưa thử train với article đoán.
    - `best` học trên dạng câu `"<activity>: <headline>"`; câu benchmark là câu tự do.
    - Dữ liệu pretrain của Omni gốc có thể trùng video benchmark (cùng nguồn YouTube), không kiểm hết được.
    - Một lượt, một seed; CI chỉ phản ánh biến thiên theo video.
    - Không có Context Fusion, cắt ngẫu nhiên / K-Medoids, nhiễu ranh giới, vai trò âm thanh (để làm sau).
    - (Nếu có) bộ benchmark khác miền: UniAV và `best` chỉ học nấu ăn.
12. **Tái lập:** commit của các repo, sha256 checkpoint, config, protocol, file kết quả; ô notebook của từng bảng.
