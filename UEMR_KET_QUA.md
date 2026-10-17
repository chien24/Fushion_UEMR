# Kết quả UEMR: đề tài hỏi gì, tìm được gì, và vì sao tin được

> **Bản lần 4, chờ chốt** (2026-10-17). Phần 1 (mục 4.0–4.3, 4.5) đã sửa theo `docs/UEMR_DUYET_KET_QUA_PHAN1.md`; phần 2 (mục 4.4,
> 4.6–4.14, phụ lục) viết theo `docs/UEMR_AGENT_VIET_KET_QUA.md` và sửa theo `docs/UEMR_DUYET_KET_QUA_PHAN2.md`; cả bản sửa theo
> `docs/UEMR_DUYET_KET_QUA_LAN2.md` và `docs/UEMR_DUYET_KET_QUA_LAN3.md`. Không còn mục [CHỜ CHẠY] hay "chờ xác nhận".
>
> **Thay đổi so với bản lần 3** (để người duyệt tập trung):
> - Mục 4.0, 4.2, 4.12 (câu 14), 4.13: hai ngoại lệ "chọn sau khi thấy tập kiểm tra" (lịch dạy thêm Omni; thu nhỏ Context Fusion) được
>   **giữ**, ghi theo những gì ghi chép cho thấy; thêm bằng chứng Omni chỉ dạy thêm trên 1.002 video train (7.787 câu, 1.948 bước).
> - Mục 4.9, 4.14: thêm ước lượng phần lấy lại được nếu chỉ tăng recall (khoảng 6.9, không phải 9.11).
> - Mục 4.0: "khoảng 3/4" → "gần 3/4 (72%)".
> - Mục 4.8–4.11 sửa theo một lượt đọc thử: bỏ bằng chứng "bỏ sót cao gấp 3–4 lần ở câu bị lật" (hiệu ứng chọn mẫu); "Omni phạt
>   nặng đoạn lệch" → "nhạy với ranh giới" (giả thuyết); hạ các kết luận dựa trên khoảng tin cậy chồng nhau; bỏ cặp 8.8 / 3.9 khỏi
>   mục 4.9; ghi rõ hai nghĩa của "cắt đều"; thêm công thức β, LSE, định nghĩa "bước gần trùng" và vài thuật ngữ.
>
> Đọc sau `docs/UEMR_FINAL.md`. Số liệu lấy từ lần chạy cuối `notebooks/log/uemr/uemr_final_AZ.ipynb` (2026-10-16). Nguồn của từng con
> số: `docs/ket_qua/nguon.csv`. Sổ số liệu chi tiết cho người làm: `docs/UEMR_TONG_KET.md`. Số bổ sung sau lần chạy A-Z: cell trong
> `notebooks/uemr_ket_qua_bo_sung.ipynb`, output ở `notebooks/log/uemr/uemr_ket_qua_bo_sung_*.md`.

---

## 4.0 Đọc trong 2 phút

**Bài toán.** Bạn gõ "thái hành tây thành hạt lựu" vào ô tìm kiếm. Trong kho có 394 video nấu ăn, mỗi video dài khoảng 5 phút và gồm
khoảng 7–8 bước nấu. Câu tìm kiếm chỉ tả **một** bước. Hệ thống phải đưa video có bước đó lên **hạng 1**.

**Ý tưởng chính.**
- **Omni** hiểu "cái gì": nó biến một đoạn video (hình và tiếng) và một câu chữ thành hai **vector**, tức hai dãy số tóm tắt nội dung.
  Đoạn và câu càng khớp nhau thì hai vector càng gần nhau.
- **UniAV** tìm "khi nào": nó xem cả video và đánh dấu mỗi bước nấu bắt đầu, kết thúc ở giây nào.
- **UEMR** ghép hai thứ đó: cắt video theo các bước UniAV tìm được, mỗi bước một vector Omni, và chấm video bằng **bước khớp nhất** với
  câu tìm kiếm.

**Bốn câu trả lời.** "Đúng hạng 1" (viết tắt **R@1**) nghĩa là: trên 100 câu tìm kiếm, bao nhiêu câu có video đúng đứng đầu.

| Câu hỏi | Trả lời |
|---|---|
| **RQ1.** Mỗi bước một vector có tốt hơn một vector cho cả video không? | **Có.** Đúng hạng 1 ở khoảng 25 trên 100 câu, so với khoảng 8. |
| **RQ2.** Khi số đoạn bằng nhau, cắt đúng chỗ bước nấu có tốt hơn cắt đều không? | **Có.** Hơn cắt đều khoảng 7 câu trên 100. Nếu giữ nguyên các đoạn UniAV tìm được nhưng đặt chúng vào chỗ ngẫu nhiên trong video, thì mất khoảng 7.5 câu. |
| **RQ3.** Cho mỗi đoạn "nhìn" thêm ngữ cảnh xung quanh (Context Fusion) có giúp không? | **Không đo được lợi ích** khi video có nhiều đoạn: tăng khoảng 0.4 câu trên 100, nằm trong mức may rủi. Ngoại lệ: khi chỉ có một vector cho cả video, tăng khoảng 3.4 (mục 4.7; chỉ đo trên bản Context Fusion nhỏ, xem ngoại lệ ở mục 4.13). |
| **RQ4.** Ranh giới bước nấu phải chính xác tới đâu? | Đoạn phải trùng với bước thật ít nhất khoảng **một nửa** (phần thời gian chung chiếm ít nhất một nửa phần thời gian gộp của cả hai, gọi là tIoU ≥ 0.5, mục 4.8) mới chắc thắng cắt đều. UniAV hiện tại còn kém mức trần (dùng đáp án của người gán nhãn) khoảng 12.6 câu trên 100; gần 3/4 (72%) phần thiếu này nằm ở các câu có bước bị UniAV bỏ sót. |

**Bảng tổng** (394 video, 3.030 câu tìm kiếm; tên đầy đủ của từng cách cắt ở mục 4.1):

| Cách cắt video | Số đoạn trung bình mỗi video | Đúng hạng 1 (trên 100 câu) |
|---|---|---|
| Cả video một đoạn | 1 | 8.32 |
| Trung bình các bước UniAV thành một vector | 1 | 9.24 |
| Chọn đoạn theo độ đa dạng nội dung (kiểu MeVTR, một phương pháp đã công bố) | 7.25 | 14.19 |
| Cắt ngẫu nhiên | 7.25 | 16.17 |
| Cắt đều | 7.25 | 18.02 |
| **Theo bước UniAV tìm được (phương pháp chính)** | **7.25** | **24.88** |
| Cắt mỗi 14 giây | 22.66 | 27.16 |
| Gộp "theo bước UniAV" với "mỗi 14 giây" | 29.91 | 29.77 |
| *Theo đáp án của người gán nhãn (oracle, không dùng được thực tế)* | 7.69 | *37.52* |

Lưu ý khi đọc bảng: cắt mỗi 14 giây cho kết quả **cao hơn** phương pháp chính, nhưng dùng gấp khoảng 3 lần số đoạn. Trong các cách có
**cùng số đoạn** (7.25), phương pháp chính đứng đầu. Cắt theo bước UniAV còn **chỉ ra đúng khoảnh khắc** tốt hơn hẳn, tức là vừa đưa đúng
video lên hạng 1 vừa chỉ ra một đoạn chồng ít nhất một nửa với bước cần tìm: khoảng 17 trên 100 câu, so với khoảng 10 khi cắt mỗi 14
giây. Mục 4.6 giải thích chi tiết.

**Hạn chế lớn nhất.** Bản UniAV đang dùng có tên **Athena** (checkpoint do nhóm UniAV train trước đó). Khi train, Athena đã học trên
**toàn bộ** phần dữ liệu chọn thiết lập (dev), và trong các lần lưu của quá trình train, người train đã chọn lần lưu cho kết quả tốt nhất
**trên tập kiểm tra**. Cả hai việc này làm số đo trên tập kiểm tra có thể **cao hơn thực tế** một chút, nên mọi con số phụ thuộc UniAV là
**tạm thời**. Dấu hiệu nằm ở điểm F1, thước đo mức UniAV tìm đúng các bước nấu (cao nhất là 100). Bình thường, dữ liệu mô hình chưa
học sẽ cho điểm thấp hơn dữ liệu đã học, như tập kiểm tra (54.4) so với train (63.5). Dev lại đạt 65.4, ngang thậm chí cao hơn train:
dev hành xử như dữ liệu đã học (mục 4.13). Các thước đo AP mà `UEMR_FINAL.md` §7 yêu cầu ở mục 4.4. Một hạn chế nhỏ hơn ở phía Omni:
lịch dạy thêm Omni được đặt sau khi xem kết quả trên tập kiểm tra của một lần dạy thêm trước đó, nên cả các số "chính thức" cũng có
thể cao hơn thực tế một chút; thứ tự giữa các cách cắt khó đổi vì mọi cách dùng chung một Omni (mục 4.13).

| Phần dữ liệu | train | dev | kiểm tra (val) |
|---|---|---|---|
| F1 của UniAV | 63.49 | 65.37 | 54.40 |

---

## 4.1 Bài toán và hệ thống, nói bằng lời thường

**Các thuật ngữ dùng từ đây:**
- **Vector:** một dãy số tóm tắt nội dung của một đoạn video hoặc một câu. Hai thứ càng giống nhau thì hai vector càng gần nhau.
- **Đa vector:** thay vì một vector cho cả video, mỗi đoạn (ví dụ mỗi bước nấu) có một vector riêng.
- **Truy hồi video:** cho một câu, xếp hạng mọi video trong kho theo mức khớp.
- **Fine-tune:** dạy thêm một mô hình có sẵn trên dữ liệu của mình. Omni `ft` là Omni đã được dạy thêm trên 1.002 video nấu ăn; Omni `pt`
  là bản gốc chưa dạy thêm.
- **Oracle:** dùng đáp án do người gán nhãn. Không dùng được trong thực tế, chỉ để biết mức trần.
- **Tham số và thiết lập:** *tham số* là các con số bên trong mô hình, mô hình tự học khi train. *Thiết lập* là các lựa chọn do người làm
  quyết định, như ngưỡng của UniAV, cách chấm điểm, thời điểm dừng train; thiết lập được chọn trên phần dữ liệu riêng gọi là dev (mục 4.2).

**Đường đi từ video tới kết quả:**
1. **Đầu vào:** một video nấu ăn có tiếng, và một câu tìm kiếm tiếng Anh (mô tả một bước nấu, lấy từ nhãn của YouCook2).
2. **UniAV đọc cả video** và trả về danh sách bước nấu, mỗi bước là [giây bắt đầu, giây kết thúc, độ tin cậy]. Ở tập kiểm tra, UniAV tìm
   trung bình 7.25 bước mỗi video; người gán nhãn đánh dấu 7.69 bước. UniAV cũng trả về một **chuỗi ngữ cảnh**: video chia thành 256 lát
   thời gian bằng nhau (với video khoảng 5 phút, mỗi lát khoảng 1.1 giây), mỗi lát một vector mô tả hình và tiếng.
3. **Cắt video thành đoạn** theo một trong các cách ở bảng dưới.
4. **Omni đọc từng đoạn** và cho ra một vector. Mỗi đoạn, dù dài 2 giây hay cả video, đều được Omni xem qua 8 khung hình lấy đều trong
   đoạn và 8 giây âm thanh ở giữa đoạn (đoạn ngắn hơn 8 giây thì âm thanh được đệm cho đủ). Câu tìm kiếm cũng đi qua Omni thành một vector.
5. *(Tuỳ chọn)* **Context Fusion:** một mô-đun nhỏ (khoảng 3.8 triệu tham số) chỉnh nhẹ vector của mỗi đoạn dựa trên chuỗi ngữ cảnh xung
   quanh và các đoạn khác của cùng video.
6. **Chấm điểm:** điểm của video là mức khớp của **đoạn khớp nhất** với câu tìm kiếm (gọi là MaxSim). Đoạn khớp nhất cũng cho luôn mốc
   thời gian của bước cần tìm.
7. **Xếp hạng** mọi video trong kho theo điểm.

**Các cách cắt video được so sánh.** Mọi cách dùng **cùng Omni, cùng câu tìm kiếm, cùng kho video, cùng cách chấm**; chỉ khác chỗ cắt.
Đề tài so 8 cách cắt chính (R0–R7, `UEMR_FINAL.md` §5) và 4 cách đối chứng. Số đoạn là trung bình mỗi video trên tập kiểm tra.

Cột "Trạng thái" dùng ba nhãn, áp dụng cho mọi kết quả trong báo cáo:
- **Chính thức:** không phụ thuộc đầu ra UniAV. Hiện chỉ có R0, R2 và các số năng lực của Omni. "Chính thức" nghĩa là không phụ thuộc
  UniAV, **không** có nghĩa là không có rò rỉ dữ liệu nào: Omni cũng có một điểm cần lưu ý (mục 4.13).
- **Tạm thời:** phụ thuộc đầu ra UniAV hiện tại: R1, R3, R4, R5, R6, `uniav_shift`, R6 ∪ R2, và mọi kết quả có Context Fusion (vì
  Context Fusion dùng chuỗi ngữ cảnh của UniAV). Sẽ đổi khi UniAV được train lại đúng quy trình.
- **Oracle:** dùng đáp án của người gán nhãn: R7, R1_gt, `uni_M_gt`, `gt_shift` và mọi mức ranh giới bị làm nhiễu của Exp 3
  (`gt_noise_*`).

| Ký hiệu | Tên đời thường | Đoạn được tạo thế nào | Số đoạn | Dùng để kiểm tra | Trạng thái |
|---|---|---|---|---|---|
| R0 | **Cả video** | toàn bộ video là một đoạn | 1 | mốc so sánh: cách làm thông thường | Chính thức |
| R1 | **Trung bình các bước** | lấy trung bình các vector của R6 thành một vector | 1 | cùng thông tin với R6 nhưng bị nén lại | Tạm thời |
| R2 | **Mỗi 14 giây** | cắt đều mỗi 14 giây (14 giây là độ dài trung vị của một bước nấu trong dữ liệu train) | 22.66 | cách cắt đơn giản, không khống chế số đoạn | Chính thức |
| R3 | **Cắt đều** | chia video thành đúng K đoạn bằng nhau, K = số bước UniAV tìm được ở video đó | 7.25 | nhiều đoạn nhưng không theo bước nấu | Tạm thời¹ |
| R4 | **Cắt ngẫu nhiên** | K đoạn, ranh giới ngẫu nhiên, mỗi đoạn ít nhất 2 giây | 7.25 | cắt bừa có tốt ngang không | Tạm thời¹ |
| R5 | **Đa dạng nội dung** (kiểu MeVTR) | gom các lát thời gian có nội dung giống nhau thành K nhóm; mỗi nhóm lấy lát giống các lát khác trong nhóm nhất làm đại diện, và lấy 2 giây quanh lát đó | 7.25 | chọn theo nội dung thay vì theo thời gian (cách của MeVTR) | Tạm thời¹ |
| R6 | **Theo bước UniAV** (phương pháp chính) | các bước UniAV tìm được | 7.25 | ranh giới bước nấu tự động | Tạm thời |
| R7 | **Theo đáp án** | ranh giới do người gán nhãn | 7.69 | mức trần khi ranh giới hoàn hảo | Oracle |
| `uni_M_gt` | **Cắt đều theo số bước thật** | K đoạn bằng nhau, K = số bước người gán nhãn | 7.69 | đối chứng cho R7: cùng số đoạn, không theo bước | Oracle |
| `uniav_shift` | **Bước UniAV bị dời chỗ** | giữ nguyên độ dài từng đoạn của R6, đặt ngẫu nhiên vào chỗ khác | 7.25 | lợi ích của R6 do vị trí hay do độ dài đoạn | Tạm thời |
| `gt_shift` | **Bước thật bị dời chỗ** | như trên, với đoạn của R7 | 7.69 | như trên, ở mức oracle | Oracle |
| R6 ∪ R2 | **Gộp bước UniAV và mỗi 14 giây** | mọi đoạn của R6 và R2 | 29.91 | hai cách có bổ sung cho nhau không | Tạm thời |

¹ R3 và R4 không dùng ranh giới của UniAV, chỉ dùng **số bước** K mà UniAV đếm được. R5 dùng cả K lẫn **chuỗi ngữ cảnh** của UniAV để
chọn đoạn. Vì vậy cả ba vẫn phụ thuộc UniAV.

Ngoài ra còn một biến thể phụ của R1: **trung bình các bước theo đáp án** (R1_gt, Oracle; đúng hạng 1 ở 10.30 trên 100 câu), dùng ở mục
4.5. Khi tài liệu khác ghi "R1 = 10.30", đó là R1_gt; R1 trong báo cáo này (9.24) là trung bình các bước **UniAV**.

---

## 4.2 Đánh giá thế nào cho công bằng

Mỗi lựa chọn dưới đây chặn một kiểu bắt bẻ cụ thể. (Ba nhãn Chính thức / Tạm thời / Oracle định nghĩa ở mục 4.1.)

**Dữ liệu: YouCook2.** Video nấu ăn có nhãn từng bước (giây bắt đầu, giây kết thúc, một câu mô tả).

| Phần dữ liệu | Video | Câu mô tả | Dùng để |
|---|---|---|---|
| Train (trainsplit) | 1.002 | 7.787 | dạy thêm Omni (`ft`) và train Context Fusion |
| Chọn thiết lập (dev) | 104 | 773 | chọn mọi thiết lập: ngưỡng của UniAV, cách chấm điểm, thời điểm dừng train |
| Kiểm tra (val) | 394 | 3.030 | chỉ để báo cáo kết quả; không lựa chọn nào dựa trên tập này (trừ các ngoại lệ dưới đây) |

- **Tập test chính thức của YouCook2 không công bố đáp án**, nên tập val được dùng làm tập kiểm tra. Trong mọi thí nghiệm của UEMR, ba
  phần không có video nào chung.
  → *Chặn câu hỏi:* "Có chọn thiết lập trên tập test không?" Mọi thiết lập của UEMR chọn trên dev; tập kiểm tra chỉ dùng để báo cáo kết quả.
  Có ba ngoại lệ, ghi ở mục 4.13:
  - hai ngoại lệ nằm ở **mô hình nền**, được train trước khi UEMR bắt đầu: UniAV (Athena) đã học cả dev khi train và được chọn lần lưu
    theo tập kiểm tra; số vòng dạy thêm Omni được đặt sau khi xem kết quả trên tập kiểm tra của một lần dạy thêm trước đó (theo ghi chép
    trong notebook dạy thêm: lần trước chọn checkpoint theo tập kiểm tra và đạt đỉnh ở 0.5 epoch). Bản thân dữ liệu dạy thêm chỉ gồm
    phần train (mục 4.13);
  - một ngoại lệ ở UEMR: kích thước của Context Fusion được thu nhỏ khi bản lớn đã được chấm trên tập kiểm tra (giảm 1.6). Ghi chép
    không cho biết quyết định có trước hay sau khi thấy số đó, nên báo cáo tính đây là ngoại lệ (cách tính thận trọng); trên dev cũng đã
    có dấu hiệu khớp quá vào train (4/7 lần chạy đạt đỉnh ở epoch 1, dev 49.42 so với 49.68).
    Ở cả hai kích thước, Context Fusion không có lợi ích đo được với mọi cách cắt có nhiều đoạn, nên lựa chọn này không làm thổi
    phồng kết luận về các cách cắt nhiều đoạn. Riêng +3.42 ở "cả video" chỉ đo trên bản nhỏ (bản lớn chưa chạy cho cả video), tức bản
    được chọn sau khi đã thấy tập kiểm tra của cắt theo bước UniAV, nên đọc kèm ngoại lệ này. Tốc độ học riêng của α được đổi cùng lúc,
    nên thuộc cùng ngoại lệ này.

  Ngoài ra, tập kiểm tra đã được chấm ở các lần chạy thăm dò trước (P4–P6) và ở lần chạy cuối; lần chạy cuối tái lập đúng các số đó
  (79 số khớp). Ngoài ba ngoại lệ trên, không lựa chọn nào dựa vào các lần chấm này.
- **Câu tìm kiếm:** cả 3.030 câu mô tả của tập kiểm tra, mỗi câu có đúng một video gốc. **Kho video chính:** 394 video kiểm tra.
  **Kho phụ:** 498 video (thêm 104 video dev không có câu tìm kiếm, chỉ làm nhiễu).
  → *Chặn:* "Kết quả có giữ khi kho lớn hơn không?" Mọi kết luận được chấm trên cả hai kho.
- **Thước đo:**
  - **R@1** (đọc là "recall tại 1"): trên 100 câu tìm kiếm, bao nhiêu câu có video đúng đứng **hạng 1**. Đây là thước đo chính.
  - **R@5, R@10:** như trên, nhưng tính đúng khi video đúng nằm trong top 5, top 10.
  - **MedR:** hạng trung vị của video đúng; càng nhỏ càng tốt.
  - **Chỉ ra đúng khoảnh khắc** (VCMR R@1): video đúng ở hạng 1 **và** đoạn khớp nhất trùng với bước cần tìm ít nhất một nửa (tIoU ≥ 0.5,
    mục 4.8). Thước đo phụ, dùng ở mục 4.6 và 4.10.
- **Khoảng tin cậy 95% (CI).** Mọi chênh lệch dùng để kết luận đều đi kèm một khoảng. Nếu khoảng **không chứa 0** thì chênh lệch gần như
  chắc chắn không phải do may rủi. Khoảng này tính bằng **bootstrap theo video**: rút ngẫu nhiên lại 394 video (có lặp) 1.000 lần, mỗi lần
  tính lại chênh lệch. Rút theo video chứ không theo từng câu, vì các câu của cùng một video không độc lập với nhau; so với rút theo câu,
  cách này cho khoảng rộng hơn, tức **thận trọng hơn**.
  → *Chặn:* "Chênh lệch có phải do may rủi không?"
- **Làm tròn.** Trong bảng, số giữ 2 chữ số thập phân; trong câu, làm tròn 1 chữ số. Chênh lệch được tính từ số chưa làm tròn, nên có thể
  lệch 0.01 so với phép trừ hai số đã làm tròn trong bảng (ví dụ 37.5248 − 28.4158 = 9.109, làm tròn 9.11, trong khi 37.52 − 28.42 = 9.10).
- **Chỉ thay một thứ mỗi lần.** Mọi cách cắt dùng cùng Omni, cùng câu tìm kiếm, cùng kho video, cùng cách chấm. Cắt đều, cắt ngẫu nhiên
  và chọn theo nội dung dùng **đúng số đoạn** của UniAV ở từng video.
  → *Chặn:* "Hơn nhau là nhờ có nhiều đoạn hơn, chứ không phải nhờ cắt đúng chỗ."
- **Chạy lại ra cùng số.** Toàn bộ kết quả được tạo lại từ đầu trong một notebook duy nhất, một phiên bản code. **79 con số** đã báo
  ở các lần chạy trước (UniAV, bảng chính, khoảng tin cậy, Exp 3, Context Fusion, Exp 5, định tính) được tính lại và khớp; phần lớn lệch không
  quá 0.005, riêng Context Fusion cho phép lệch 0.2 vì train lại có ngẫu nhiên. **89 kiểm thử tự động** của code (cách cắt, cách chấm,
  đọc dữ liệu, Context Fusion) đều qua.
  → *Chặn:* "Có tái lập được không?"

---

## 4.3 Bảng đối chiếu với mục tiêu trong `UEMR_FINAL.md`

Cột "Chi tiết" trỏ tới mục của báo cáo này.

**Câu hỏi nghiên cứu (`UEMR_FINAL.md` §2)**

| Mục tiêu | Đã làm | Kết luận một câu | Chi tiết |
|---|---|---|---|
| RQ1: đa vector so với một vector | Xong | Đa vector hơn rõ rệt (24.9 so với 8.3), và lợi ích mất gần hết khi nén về một vector | 4.5 |
| RQ2: ranh giới bước nấu, cùng số đoạn | Xong | Theo bước UniAV hơn cắt đều, cắt ngẫu nhiên, chọn theo nội dung; lợi ích đến từ **vị trí** cắt. Cắt mỗi 14 giây với gấp 3 lần số đoạn thì cao hơn | 4.6 |
| RQ3: ngữ cảnh (Context Fusion) | Xong | Không đo được lợi ích (+0.4, khoảng tin cậy chứa 0); thứ hạng giữa các cách cắt không đổi | 4.7 |
| RQ4: độ chính xác ranh giới cần có | Xong (không có nhánh có Context Fusion) | Đoạn cần trùng với bước thật từ khoảng một nửa trở lên; UniAV còn cách mức trần 12.6, khoảng 72% trong đó nằm ở các câu có bước bị UniAV bỏ sót | 4.8, 4.9 |
| RQ5 (tạp chí): một **index** (bộ vector lưu sẵn để tìm) cho cả tìm video và chỉ ra khoảnh khắc | Một phần | Đã đo việc chỉ ra khoảnh khắc trên YouCook2 bằng cùng index; chưa làm ActivityNet | 4.10 |

**Thí nghiệm (`UEMR_FINAL.md` §8)**

| Mục tiêu | Đã làm | Kết luận một câu | Chi tiết |
|---|---|---|---|
| Exp 1 cột (a): Omni chưa dạy thêm | Một phần: cả video, trung bình các bước, cắt đều, theo bước UniAV, theo đáp án | Thứ tự giữa các cách cắt giữ nguyên như cột (b) | 4.5, 4.6 |
| Exp 1 cột (b): Omni đã dạy thêm | Xong: mọi cách cắt và mọi đối chứng | Bảng chính của khoá luận | 4.5, 4.6, phụ lục C |
| Exp 1 cột (c): thêm Context Fusion | Một phần: cả video, cắt đều, theo bước UniAV, theo đáp án | Không cách nào tăng có ý nghĩa, trừ "cả video" | 4.7 |
| Exp 2: tháo từng phần của Context Fusion | Xong cả 7 cấu hình của bảng Exp 2 | Không phần nào tạo lợi ích đo được | 4.7 |
| Exp 3: làm nhiễu ranh giới | Xong nhánh không có Context Fusion | Đoạn theo bước còn hơn cắt đều một cách chắc chắn khi trùng với bước thật từ khoảng một nửa trở lên. Nhánh có Context Fusion không chạy, vì Context Fusion không tạo lợi ích ở mọi cách cắt có nhiều đoạn | 4.8 |
| Exp 4: vai trò của âm thanh | **Chưa làm** | Code đã sẵn, chưa chạy; chỉ làm được phía Omni và bằng cách thay âm thanh bằng im lặng | 4.13 |
| Exp 5: cách chấm điểm và chi phí | Xong; thiếu thời gian chạy UniAV mỗi video | Với cách cắt theo bước UniAV, "đoạn khớp nhất" là cách chấm tốt nhất trên cả dev và tập kiểm tra; dùng độ tin cậy của UniAV không giúp; lưu vector bằng nửa bộ nhớ (nén 8 bit) không mất độ chính xác | 4.10 |
| Exp 6 (tạp chí): ActivityNet, định vị khoảnh khắc | **Chưa làm** (ngoài phạm vi khoá luận) | – | 4.14 |
| Định tính 1: câu nào "cả video" thắng "theo bước UniAV" | Xong | Bước bị UniAV bỏ sót hay gặp ở các câu này (32.2%, chưa phải đa số); tỉ lệ này ngang mức chung của mọi câu mà cắt theo bước UniAV sai (khoảng 34%), nên không phải đặc điểm riêng | 4.11 |
| Định tính 2: các bước giống nhau có tách được không | Một phần | Đã đo mức độ của vấn đề: nhầm sang bước lặp lại trong cùng video chỉ ở 25 trên 298 lượt đúng video sai đoạn, tức nguồn nhỏ; bước gần trùng ở video khác thì hại rõ (mục 4.9). Chưa đo riêng Context Fusion có tách được không | 4.11 |
| Định tính 3: Context Fusion nhìn vào đâu | Xong | Mỗi đoạn dồn khoảng 82–86% sự chú ý vào chính khoảng thời gian của nó | 4.7, 4.11 |

**Phạm vi khoá luận (`UEMR_FINAL.md` §10)**

| Hạng mục | Mức | Đã làm |
|---|---|---|
| Cache sẵn đầu ra UniAV và Omni | bắt buộc | Xong |
| Exp 1 ba cột | bắt buộc | (b) đủ; (a), (c) một phần như trên |
| Exp 2, Exp 3, định tính | bắt buộc | Exp 2 xong / Exp 3 xong một nhánh / định tính: 2 câu xong, 1 câu một phần |
| Exp 4 | nên có | **Chưa làm** |
| Exp 5 | nên có | Xong, thiếu thời gian chạy UniAV |
| 3 lần chạy với 3 seed cho cả video, cắt đều, theo bước UniAV có Context Fusion | nên có | Xong (theo đáp án: 1 lần) |

**Đóng góp (`UEMR_FINAL.md` §9)**

| Đóng góp | Số liệu có ủng hộ không | Cách viết đúng |
|---|---|---|
| 1. Framework UEMR (UniAV + Omni, đa vector theo bước) | **Có** | Giữ nguyên |
| 2. Nghiên cứu có kiểm soát 8 cách cắt chính (cùng 4 đối chứng) trên cùng mô hình nền | **Có**, kèm ba kết quả trái giả thuyết (cắt mỗi 14 giây cao hơn; chọn theo nội dung thấp hơn cắt đều; với bước dài từ 40 giây, cắt theo bước không hơn cắt đều) | Giữ, báo cáo cả ba kết quả trái giả thuyết (mục 4.6) |
| 3. Context Fusion | **Không có lợi ích đo được** | Viết lại: "Mô-đun được cài đặt và kiểm chứng là học được (train trên 32 video, mô-đun đưa kết quả trên chính 32 video đó từ 70.87 lên 91.74, tức đủ sức ghi nhớ dữ liệu; hệ số α, tức mức chỉnh vector, học ra khác 0 ở cả 3 lần chạy; có lợi ích +3.4 khi chỉ dùng một vector cho cả video, đo trên bản nhỏ được chọn sau khi thấy tập kiểm tra, mục 4.13). Với cách cắt nhiều đoạn, cấu hình và lượng dữ liệu này, không phát hiện lợi ích." |
| 4. Phân tích độ nhạy theo chất lượng ranh giới | **Có** | Giữ nguyên |
| 5. (Tạp chí) Một index cho hai bài toán | Chưa đủ | Chỉ nêu phần đã đo trên YouCook2 |

Các điều **không claim** (giữ đúng `UEMR_FINAL.md` §9): đa vector, MaxSim (chấm video bằng đoạn khớp nhất) hay InfoNCE (một hàm
mất mát phổ biến dùng để train mô hình tìm kiếm) không phải ý tưởng mới; Omni và UniAV không phải mô
hình mới; kết quả dùng đáp án của người gán nhãn luôn gọi là oracle.

---

## 4.4 Năng lực của hai mô hình nền

**Trả lời ngắn:** Omni tìm đúng đoạn của một bước nấu ở khoảng 36 trên 100 câu, khi phải chọn giữa 3.030 đoạn của tập kiểm tra. UniAV bỏ
sót khoảng 28 trên 100 bước nấu ở tập kiểm tra.

**Omni: một câu tìm đúng đoạn của nó không?** Mỗi câu mô tả phải tìm đúng đoạn video (cắt theo đáp án) của chính nó, giữa mọi đoạn của
tập. Đây là bài toán Omni được dạy thêm để giải.

| Tập | Số đoạn phải chọn | Đúng hạng 1, Omni đã dạy thêm | Đúng hạng 1, Omni chưa dạy thêm |
|---|---|---|---|
| kiểm tra (val) | 3.030 | 35.64 | 22.41 |
| dev | 773 | 56.40 | – |

Dạy thêm giúp Omni tăng khoảng 13 câu trên 100. Dev cao hơn chủ yếu vì phải chọn giữa ít đoạn hơn khoảng 4 lần: trên dữ liệu train,
với kho cùng cỡ dev (khoảng 809 đoạn, lấy mẫu 30 lần), Omni đạt 54.75 ± 1.91, ngang dev. Omni chỉ được dạy thêm trên phần train.

**UniAV: tìm các bước nấu tốt tới đâu?** Một bước nấu gọi là **bị bỏ sót** khi không đoạn nào của UniAV chồng với nó được ít nhất 0.3
(tIoU, xem mục 4.8). **F1** là thước đo tổng hợp mức "tìm đúng" và "không tìm thừa" (cao nhất là 100), tính với ngưỡng chồng 0.5. **AP**
là thước đo tương tự nhưng có xét độ tin cậy UniAV gán cho từng đoạn.

| | train | dev | kiểm tra (val) |
|---|---|---|---|
| Số bước UniAV tìm / số bước thật (trung bình mỗi video) | 7.47 / 7.77 | 6.95 / 7.43 | 7.25 / 7.69 |
| F1 (ngưỡng chồng 0.5) | 63.49 | 65.37 | 54.40 |
| AP ở ngưỡng chồng 0.3 / 0.5 / 0.7 | 68.06 / 49.90 / 27.15 | 70.23 / 52.82 / 27.79 | 59.30 / 36.71 / 13.78 |
| Bước bị bỏ sót (%) | 22.60 | 21.99 | 27.82 |

**Điều này cho thấy:** UniAV đếm số bước gần đúng (7.25 so với 7.69), nhưng ở tập kiểm tra bỏ sót khoảng 28 trên 100 bước, và đoạn nó
tìm chỉ trùng khít (chồng ≥ 0.7) với bước thật ở mức thấp (AP 13.8). Dev tốt hơn cả train là dấu hiệu UniAV đã thấy dev khi train (mục
4.13). Mục 4.9 cho thấy bỏ sót là dạng lỗi rõ nhất của UEMR; khoảng 72% khoảng cách tới mức trần nằm ở các câu có bước bị bỏ sót.

**Trạng thái:** Omni: chính thức. UniAV: tạm thời.
**Nguồn:** Omni: `docs/UEMR_TONG_KET.md` §5.1 (log `notebooks/log/omni/new_log_ft.ipynb` c25, `notebooks/log/uemr/P6/uemr_p6_colab_v2.ipynb`
c10, `notebooks/log/uemr/P6/uemr_p6_patch_colab_log.ipynb` c14); UniAV: `notebooks/log/uemr/uemr_final_AZ.ipynb` c10, c12.

---

## 4.5 Chia video thành nhiều đoạn có tốt hơn dùng một vector cho cả video không?  (RQ1)

**Trả lời ngắn:** Có. Cắt theo bước UniAV cho đúng hạng 1 ở khoảng 25 trên 100 câu tìm kiếm, so với khoảng 8 khi dùng một vector cho cả
video.

**Đã kiểm tra bằng cách nào:** So ba cách biểu diễn, cùng mô hình Omni, cùng 394 video, cùng 3.030 câu tìm kiếm:
- **R0, cả video:** một vector cho cả video.
- **R6, theo bước UniAV:** mỗi bước UniAV tìm được một vector, video được chấm bằng bước khớp nhất.
- **R1, trung bình các bước:** lấy chính các vector của R6 rồi tính trung bình thành một vector.

R1 và R6 chứa **cùng một thông tin**, chỉ khác là R1 bị nén lại thành một vector. So R6 với R1 vì thế trả lời đúng một câu hỏi: lợi ích đến
từ việc **giữ riêng từng bước**, hay chỉ từ việc có thông tin về các bước.

**Kết quả** (kho 394 video):

| So sánh | Chênh lệch đúng hạng 1 (trên 100 câu) | Khoảng tin cậy 95% |
|---|---|---|
| Theo bước UniAV − cả video | +16.57 | [+14.69, +18.61] |
| Theo bước UniAV − trung bình các bước | +15.64 | [+13.49, +17.77] |
| Trung bình các bước − cả video | +0.92 | [−0.50, +2.48] |
| *Oracle:* theo đáp án − trung bình các bước theo đáp án | *+27.23* | *[+24.65, +29.70]* |

**Điều này cho thấy:** Lợi ích gần như hoàn toàn đến từ việc **giữ riêng từng bước**. Khi nén các bước lại thành một vector, kết quả chỉ
còn ngang cách dùng cả video: chênh 0.9 câu và khoảng tin cậy chứa 0. (Với Omni chưa dạy thêm, nén lại vẫn hơn cả video một chút, xem
câu hỏi thứ hai bên dưới; phần lớn lợi ích vẫn nằm ở việc giữ riêng.) Ở mức oracle, khi ranh giới hoàn hảo, khoảng cách giữa "giữ riêng"
và "nén lại" còn lớn hơn (khoảng 27 câu trên 100).

**Vì sao lại như vậy:** Một câu tìm kiếm chỉ tả một bước nấu. Khi trộn mọi bước vào một vector, bước đó bị "pha loãng" bởi 6–7 bước khác.
Khi giữ riêng, bước đúng khớp trực tiếp với câu tìm kiếm. Phép so R6 với R1 **ủng hộ** trực tiếp cách giải thích này: hai bên có cùng thông
tin, chỉ khác ở việc nén, và nén thì mất lợi ích. Phép so này chưa chứng minh cơ chế "pha loãng" bên trong vector.

**Hội đồng có thể hỏi:**
- *"Nhiều vector thì dễ có một vector trùng hợp khớp cao. Sao biết lợi ích là nhờ bước nấu?"* → Câu này do RQ2 trả lời (mục 4.6): so với
  cắt đều có **cùng số đoạn**, cắt theo bước UniAV vẫn hơn khoảng 7 câu trên 100.
- *"Cả video chỉ được Omni xem 8 khung hình, còn theo bước UniAV được xem khoảng 58 (7.25 đoạn × 8). Hơn nhau có phải chỉ vì được xem
  nhiều hơn?"* → Không. Trung bình các bước (R1) dùng **đúng các vector** của theo bước UniAV, tức đã xem đúng số khung hình đó, nhưng chỉ
  ngang cả video (+0.92 [−0.50, +2.48]). Cắt đều (R3) cũng xem đúng số khung hình như theo bước UniAV mà thua 6.86. Vậy lợi ích không đến
  từ việc xem nhiều khung hình hơn, mà từ việc giữ riêng từng đoạn và cắt đúng chỗ.
- *"Cả video thua vì Omni được dạy thêm trên đoạn ngắn, nên không quen với video dài?"* → Cột (a) dùng **Omni chưa dạy thêm**. Ở đó thứ tự
  giữ nguyên: cả video 7.89, trung bình các bước 9.60, theo bước UniAV 18.94. Theo bước UniAV vẫn gấp khoảng 2.4 lần cả video.

  | Cột (a), Omni chưa dạy thêm, kho 394 | Chênh lệch | Khoảng tin cậy 95% |
  |---|---|---|
  | Theo bước UniAV − cả video | +11.06 | [+9.21, +12.89] |
  | Theo bước UniAV − trung bình các bước | +9.34 | [+7.49, +11.22] |
  | Trung bình các bước − cả video | +1.72 | [+0.19, +3.30] |

  Một khác biệt so với Omni đã dạy thêm: ở đây "trung bình các bước" hơn "cả video" **có ý nghĩa** (+1.7, khoảng tin cậy không chứa 0).
  Một cách giải thích là Omni chưa dạy thêm chưa quen với clip dài cả video, nên vector trung bình của các đoạn ngắn tốt hơn một chút;
  sau khi dạy thêm, chênh lệch này không còn ý nghĩa. Kết luận chính không đổi: giữ riêng từng bước vẫn hơn nén lại khoảng 9 câu trên
  100. Kho 498 video cho cùng kết luận (+10.86, +9.01, +1.85; phụ lục C).
- *"Kho lớn hơn thì sao?"* → Với kho 498 video (thêm 104 video gây nhiễu), kết luận giữ nguyên: theo bước UniAV − cả video +15.74
  [+13.98, +17.50]; trung bình các bước − cả video +1.29 [−0.07, +2.75].

**Trạng thái:** Tạm thời (R6 và R1 dùng đầu ra UniAV hiện tại). R0 là chính thức; dòng theo đáp án là oracle.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c18 (bảng cột (a), (b), bootstrap theo video); đầu ra trên Drive
`results/056d388e/exp1_b.csv`, `exp1_b_bootstrap.csv`, `exp1_a_pt.csv`; `notebooks/log/uemr/NHAN_XET_KET_QUA_AZ.md` §2; khoảng tin cậy cột
(a): `notebooks/uemr_ket_qua_bo_sung.ipynb` cell B1, output ở `notebooks/log/uemr/uemr_ket_qua_bo_sung_B1.md`
(Drive: `results/056d388e/exp1_a_pt_bootstrap.csv`). Từng số: `docs/ket_qua/nguon.csv`, mục 4.5.

---

## 4.6 Khi số đoạn bằng nhau, cắt đúng chỗ bước nấu có tốt hơn không?  (RQ2)

**Trả lời ngắn:** Có. Với cùng khoảng 7 đoạn mỗi video, cắt theo bước UniAV đúng hạng 1 ở khoảng 25 trên 100 câu, so với khoảng 18 khi
cắt đều, 16 khi cắt ngẫu nhiên và 14 khi chọn theo độ đa dạng nội dung. Lợi ích đến từ **vị trí** của chỗ cắt.

**Đã kiểm tra bằng cách nào:**
- **Cùng số đoạn:** cắt đều (R3), cắt ngẫu nhiên (R4) và đa dạng nội dung (R5) dùng **đúng số đoạn** UniAV tìm được ở từng video. Nếu R6
  hơn, thì không phải nhờ có nhiều vector hơn.
- **Cùng độ dài đoạn:** "bước UniAV bị dời chỗ" giữ nguyên số đoạn và độ dài từng đoạn của R6, chỉ đặt chúng vào chỗ ngẫu nhiên. Nếu R6
  hơn, thì không phải nhờ độ dài đoạn phù hợp.
- **Cùng độ phủ:** **độ phủ** là phần trăm thời lượng video nằm trong ít nhất một đoạn. R6 phủ 51.6% video, bản bị dời chỗ chỉ phủ 41.6%
  (đoạn bị dời chồng lên nhau nhiều hơn). Để loại yếu tố này: mỗi video cho một cặp số, gồm R6 hơn bản bị dời chỗ bao nhiêu và R6 phủ
  nhiều hơn bao nhiêu. Kẻ một đường thẳng qua mọi cặp đó, rồi đọc giá trị của đường tại chỗ chênh độ phủ bằng 0: đó là chênh lệch **khi độ
  phủ bằng nhau** (hồi quy tuyến tính; công thức ở phụ lục E).
- **Ở mức oracle:** làm lại các phép so trên với ranh giới của người gán nhãn.

**Kết quả** (kho 394 video):

| So sánh (cùng số đoạn) | Chênh lệch | Khoảng tin cậy 95% |
|---|---|---|
| Theo bước UniAV − cắt đều | +6.86 | [+4.97, +8.75] |
| Theo bước UniAV − cắt ngẫu nhiên | +8.71 | [+6.93, +10.50] |
| Theo bước UniAV − đa dạng nội dung | +10.69 | [+8.83, +12.37] |
| Theo bước UniAV − bước UniAV bị dời chỗ | +7.52 | [+5.79, +9.28] |
| Như dòng trên, khi độ phủ bằng nhau (hồi quy) | +8.99 | [+6.26, +11.88] |
| *Oracle:* theo đáp án − cắt đều theo số bước thật | *+18.61* | *[+16.49, +20.95]* |
| *Oracle:* theo đáp án − bước thật bị dời chỗ | *+19.80* | *[+17.66, +21.97]* |

Thứ tự cắt đều < theo bước UniAV < theo đáp án giữ nguyên với Omni chưa dạy thêm (15.15 < 18.94 < 25.31).

**Điều này cho thấy:** Cắt đúng chỗ bước nấu có giá trị riêng, tách khỏi số đoạn, độ dài đoạn và độ phủ. Dời chính các đoạn đó sang chỗ
khác thì mất khoảng 7.5 câu trên 100; khi tính cả việc độ phủ bằng nhau, phần mất còn lớn hơn (khoảng 9). Ở mức oracle, giá trị của vị
trí lớn gấp khoảng 2.6–2.7 lần (19.80 so với 7.52; 18.61 so với 6.86).

R6 chỉ phủ khoảng một nửa thời lượng video. Nếu bước cần tìm nằm trong nửa không được phủ, không vector nào đại diện cho nó, và video
thường bị xếp thấp. Đây chính là lỗi "bỏ sót" phân tích ở mục 4.9.

**Vì sao lại như vậy:** Lợi ích tập trung ở **bước ngắn**. Bảng dưới là số câu đúng hạng 1 trên 100 (R@1), chia theo độ dài bước cần
tìm:

| Độ dài bước cần tìm | Số câu | Cắt đều | Theo bước UniAV | Theo đáp án (oracle) |
|---|---|---|---|---|
| dưới 10 giây | 955 | 11.94 | 21.05 | 44.29 |
| 10–40 giây | 1.750 | 19.43 | 26.46 | 35.83 |
| từ 40 giây | 325 | 28.31 | 27.69 | 26.77 |

Đoạn cắt đều dài trung vị 39 giây. Một bước 5 giây nằm trong đoạn 39 giây thì chỉ chiếm phần nhỏ, và vector của đoạn bị các bước khác lấn
át. Đoạn của UniAV bao gọn bước đó. Ví dụ: với câu "mix corn flour and water to make paste" (bước dài 5 giây), cắt đều xếp video đúng
ở hạng 206, cắt theo bước UniAV xếp hạng 1; với "bring a large pan of water to boil" (2 giây): hạng 158 so với hạng 1. Với bước dài từ 40 giây, đoạn cắt đều đã đủ dài để chứa bước, nên lợi thế biến mất. Phần giải
thích "lấn át" là giả thuyết khớp với số liệu theo độ dài, chưa có thí nghiệm tách riêng.

Ngay cả theo đáp án, kết quả cũng giảm khi bước dài ra (44.29 → 26.77), và ở nhóm từ 40 giây còn thấp hơn cắt đều. Một cách giải thích:
mỗi đoạn chỉ được Omni xem qua 8 khung hình và 8 giây âm thanh (mục 4.1), nên với bước dài từ 40 giây, Omni chỉ thấy một phần nhỏ của
bước. Nhóm này nhỏ (325 câu) và chênh lệch giữa ba cột chưa được tính khoảng tin cậy.

**Ba kết quả trái giả thuyết** (báo cáo đầy đủ, không giấu):

1. **Cắt mỗi 14 giây cao hơn cắt theo bước UniAV về R@1:** 27.16 so với 24.88, chênh −2.28 [−4.33, −0.27]. Cách đọc đúng:
   - Cắt mỗi 14 giây dùng **gấp khoảng 3 lần số đoạn** (22.66 so với 7.25 mỗi video). Câu trả lời của RQ2 chỉ đúng khi số đoạn bằng nhau.
     Chính số đoạn có ảnh hưởng lớn: cắt đều 7 đoạn thua cắt mỗi 14 giây 9.14 câu trên 100 [−11.09, −7.10].
   - Cắt mỗi 14 giây phủ 100% video nên không bỏ sót bước nào. Ở những câu nó đúng mà cắt theo bước UniAV sai, 34.7% là bước UniAV bỏ sót
     (không đoạn nào của UniAV trùng với bước đó từ 0.3 trở lên, mục 4.4).
   - Ở kho 498 video, chênh lệch không còn ý nghĩa: −1.68 [−3.68, +0.17].
   - Cắt theo bước UniAV **chỉ ra đúng khoảnh khắc** tốt hơn hẳn: đúng cả video lẫn đoạn ở 16.93 so với 9.87 trên 100 câu (tính trên mọi câu,
     nên so sánh công bằng). Một số phụ: trong các câu mà mỗi cách đã đưa đúng video lên hạng 1, đoạn khớp nhất trùng ít nhất một nửa với
     bước cần tìm ở 68.0% so với 36.3%; hai tỉ lệ này tính trên hai tập câu khác nhau nên chỉ để tham khảo.
   - Hai cách **bổ sung cho nhau:** gộp đoạn của cả hai cho 29.77, cao nhất trong mọi cách đã chạy, hơn cắt mỗi 14 giây +2.61 [+1.35, +3.86].
2. **Đa dạng nội dung (kiểu MeVTR) thấp hơn cả cắt đều:** 14.19 so với 18.02, trái giả thuyết R5 > R3 của `UEMR_FINAL.md` §8. Giả thuyết:
   mỗi đoạn của R5 chỉ dài 2 giây quanh lát đại diện, nên R5 chỉ phủ 5.5% thời lượng video, trong khi cắt đều phủ 100% và cắt theo bước
   UniAV phủ 51.6%. Phần lớn video không có vector nào. Đây cũng là khác biệt thiết kế với MeVTR gốc: MeVTR chọn **khung hình** đại diện
   bằng mô hình ảnh CLIP, còn ở đây mỗi điểm đại diện phải thành một clip để đưa qua Omni.
3. **Với bước dài từ 40 giây,** cắt theo bước UniAV không hơn cắt đều (bảng trên).

**So với giả thuyết gốc** R7 ≥ R6 > R5 > R3 ≈ R4 > R0 ≈ R1 (`UEMR_FINAL.md` §8):

| Phần của giả thuyết | Kết quả | Đúng hay sai |
|---|---|---|
| R7 ≥ R6 (đáp án ≥ bước UniAV) | 37.52 > 24.88 | Đúng |
| R6 > R5 (bước UniAV > đa dạng nội dung) | +10.69 | Đúng |
| R5 > R3 (đa dạng nội dung > cắt đều) | 14.19 < 18.02 | **Sai** |
| R3 ≈ R4 (cắt đều ≈ ngẫu nhiên) | 18.02 và 16.17 | Gần đúng: cắt đều hơn khoảng 1.9 (chưa tính khoảng tin cậy cho cặp này) |
| R4 > R0 (ngẫu nhiên > cả video) | 16.17 > 8.32 | Đúng |
| R0 ≈ R1 (cả video ≈ trung bình các bước) | +0.92 [−0.50, +2.48] | Đúng với Omni đã dạy thêm (với Omni chưa dạy thêm R1 hơn 1.7, mục 4.5) |

**Hội đồng có thể hỏi:**
- *"R6 phủ nhiều thời lượng hơn bản bị dời chỗ, nên hơn là nhờ độ phủ?"* → Khi độ phủ bằng nhau (hồi quy theo từng video), R6 vẫn hơn
  +8.99 [+6.26, +11.88]. Ngoài ra R6 phủ 51.6% vẫn hơn cắt đều phủ 100% (+6.86).
- *"Khác gì MeVTR?"* → MeVTR chọn đơn vị theo **nội dung**; dòng "đa dạng nội dung" tái hiện ý đó trên cùng mô hình nền và thua cắt theo
  bước UniAV 10.69 câu trên 100. Phần giải thích vì sao R5 thấp là giả thuyết (độ phủ 5.5%), chưa có thí nghiệm tách riêng.
- *"Cắt mỗi 14 giây cao hơn, sao vẫn nói cắt theo bước có giá trị?"* → Vì claim là "tốt nhất **khi cùng số đoạn**", cộng ba điểm: dùng ít
  hơn khoảng 3 lần số vector; chỉ ra khoảnh khắc tốt hơn hẳn; và hai cách bổ sung cho nhau (gộp lại cao nhất).

**Trạng thái:** Tạm thời (mọi dòng có R3–R6); các dòng theo đáp án là oracle; cắt mỗi 14 giây là chính thức.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c18 (bảng cột (a), (b), bootstrap), c19 (theo độ dài, hồi quy độ phủ), c12 (độ phủ,
độ dài trung vị), c30 (nhóm lật); ví dụ "mix corn flour…", "bring a large pan…": `notebooks/log/uemr/NHAN_XET_KET_QUA_A4.md` §2.2.

---

## 4.7 Cho mỗi đoạn "nhìn" thêm ngữ cảnh xung quanh có giúp không?  (RQ3)

**Trả lời ngắn:** Không đo được lợi ích. Context Fusion đưa cắt theo bước UniAV từ 24.88 lên 25.31 (trung bình 3 lần chạy), tăng khoảng
0.4 câu trên 100, nằm trong mức may rủi.

**Đã kiểm tra bằng cách nào:** Context Fusion là một mô-đun nhỏ (3.8 triệu tham số) đặt sau Omni. Với mỗi đoạn, nó nhìn chuỗi ngữ cảnh
của UniAV quanh đoạn đó (bước 1) và các đoạn khác của cùng video (bước 2), rồi chỉnh nhẹ vector Omni của đoạn. Mức chỉnh do một hệ số
**α** quyết định; α bắt đầu bằng 0, tức là ban đầu giữ nguyên vector Omni, và mô-đun tự học α.
- Train trên 1.002 video train, chọn thời điểm dừng trên dev, rồi chấm tập kiểm tra để báo cáo.
- Chạy 3 lần với 3 seed (khởi tạo ngẫu nhiên khác nhau) cho cả video, cắt đều và cắt theo bước UniAV.
- **Exp 2:** bỏ dần từng thành phần để xem thành phần nào có tác dụng.
- **Hai đối chứng:** (1) **xáo trộn chuỗi ngữ cảnh** giữa các video: nếu bản xáo trộn vẫn tăng, phần tăng là do thêm tham số chứ không
  phải do ngữ cảnh; (2) **δ = 0**: không cho đoạn nhìn ra ngoài khoảng thời gian của nó (cấu hình chính cho nhìn ra ngoài 5 giây mỗi bên).

**Kết quả:**

| Cách cắt | Không có Context Fusion | Có Context Fusion | Chênh lệch | Khoảng tin cậy 95% |
|---|---|---|---|---|
| Theo bước UniAV (3 lần: 25.18 / 25.08 / 25.68) | 24.88 | 25.31 | +0.43 | [−0.46, +1.31] |
| Cắt đều (3 lần) | 18.02 | 18.10 | +0.08 | [−0.39, +0.49] |
| *Theo đáp án, oracle (1 lần)* | *37.52* | *37.72* | *+0.20* | *[−1.48, +1.95]* |
| Cả video (3 lần) | 8.32 | 11.74 | **+3.42** | **[+2.21, +4.67]** |

Exp 2 (cắt theo bước UniAV, lần chạy seed 0; chênh lệch so với không có Context Fusion):

| Cấu hình | Đúng hạng 1 | Chênh lệch | Khoảng tin cậy 95% |
|---|---|---|---|
| chỉ bước 1 (nhìn chuỗi ngữ cảnh) | 24.52 | −0.36 | [−1.30, +0.59] |
| + bước 2 (nhìn các đoạn khác) | 24.39 | −0.50 | [−1.53, +0.60] |
| + hàm mất mát mức đoạn | 25.08 | +0.20 | [−0.71, +1.04] |
| + vector cả video (cấu hình đầy đủ) | 25.18 | +0.30 | [−0.56, +1.09] |
| *Đối chứng:* chuỗi ngữ cảnh bị xáo trộn | 24.36 | −0.53 | [−1.50, +0.42] |
| *Đối chứng:* δ = 0 (giá trị trùng trung bình 3 seed ở bảng trên là ngẫu nhiên) | 25.31 | +0.43 | [−0.42, +1.27] |

Cấu hình đầy đủ so với hai đối chứng: hơn bản xáo trộn +0.83 [−0.07, +1.68]; ngang δ = 0 (−0.13 [−0.57, +0.29]). Thứ tự cả video < cắt
đều < theo bước UniAV < theo đáp án giữ nguyên sau Context Fusion.

**Điều này cho thấy:** Với cách cắt có nhiều đoạn, cấu hình và lượng dữ liệu này, ngữ cảnh không mang thêm thông tin đo được. Không
thành phần nào của Exp 2 tạo khác biệt có ý nghĩa so với không có Context Fusion. Ngoại lệ duy nhất là "cả video", nơi chỉ có một vector.

**Mô-đun chạy đúng, không phải cài sai.** Bằng chứng:
- **Học được:** trên tập nhỏ 32 video, mô-đun đưa kết quả trên chính tập đó từ 70.87 lên 91.74 (kiểm tra "thuộc bài", ngưỡng đặt trước
  là 90).
- **α học ra khác 0** ở cả 3 lần chạy (+0.13, −0.12, −0.37): mô-đun có chỉnh vector, không đứng yên. Dấu của α không có ý nghĩa riêng,
  vì hướng của phần chỉnh do chính mô-đun học (đổi dấu cả α lẫn hướng chỉnh cho cùng kết quả); điều quan trọng là α khác 0, và **độ lớn**
  của nó. Vector Omni có độ dài 1, phần chỉnh cũng được chuẩn hoá về độ dài 1, rồi tổng của chúng được chuẩn hoá lại (phụ lục D). Vì vậy
  |α| = 0.12, 0.13 và 0.37 ở ba lần chạy nghĩa là phần chỉnh dài bằng khoảng 12%, 13% và 37% vector gốc trước khi chuẩn hoá lại. Ngay cả
  mức nhỏ nhất cũng không phải không đáng kể, và lần chỉnh mạnh nhất (37%) cho kết quả cao nhất trong ba lần (25.68) mà vẫn chỉ hơn bản
  không chỉnh khoảng 0.8.
  α khởi tạo bằng 0 và học với tốc độ lớn hơn các tham số khác, nên "khác 0" thôi chưa đủ; độ lớn này mới là bằng chứng mô-đun thật sự
  chỉnh vector. Ba lần chạy
  ra ba giá trị khác nhau nhưng kết quả gần nhau (25.18, 25.08, 25.68): mô-đun tìm ra các cách chỉnh khác nhau mà không cách nào giúp.
- **Chú ý có cấu trúc, nhưng ít nhìn ra ngoài đoạn:** *attention* (sự chú ý) là trọng số cho biết mỗi đoạn "nhìn" vào lát thời gian nào
  của chuỗi ngữ cảnh. Ở 3 video có nhiều đoạn nhất, 82–86% sự chú ý của bước 1 rơi vào chính khoảng thời gian của đoạn. Mô-đun không
  nhìn bừa, nhưng cũng ít dùng ngữ cảnh **ngoài** đoạn mà thiết kế nhắm tới; điều này khớp với việc δ = 0 cho kết quả ngang cấu hình đầy
  đủ (hình
  `results/056d388e/qualitative/attention_step1.png` trên Drive).
- **Làm được khi có chỗ để làm:** với "cả video", cùng mô-đun tăng +3.42. Số này chỉ đo trên bản nhỏ, bản được chọn sau khi đã thấy
  tập kiểm tra (ngoại lệ ở mục 4.2, 4.13).
- **Dev có tăng:** ở lần chạy seed 0, từ 49.81 lên 52.39 (+2.59), nhưng phần tăng không chuyển sang tập kiểm tra. Mốc 49.81 là dev khi
  chưa chỉnh vector nhưng đã thêm vector cả video mà cấu hình đầy đủ dùng; mốc 49.68 ở mục 4.10 là dev khi không có cả hai. Hai mốc lệch
  nhau đúng 1 câu trên 773.
- **Đường dữ liệu đúng:** khi không chỉnh vector, kết quả tính qua dữ liệu của Context Fusion trùng đúng số không có Context Fusion ở cả 4
  cách cắt (8.32, 18.02, 24.88, 37.52).

**Vì sao lại như vậy:** Giả thuyết: chuỗi ngữ cảnh của UniAV bên trong đoạn mang **cùng thông tin** mà Omni đã mã hoá từ chính clip đó,
nên không có gì mới để thêm. Hai dấu hiệu ủng hộ: xáo trộn chuỗi ngữ cảnh chỉ làm kém đi 0.83 (khoảng tin cậy chứa 0), và không cho nhìn
ra ngoài đoạn (δ = 0) cũng không làm kém đi. Một cách giải thích cho ngoại lệ "cả video": vector duy nhất bỏ qua chi tiết từng bước, nên
chuỗi ngữ cảnh theo thời gian bổ sung được; đó không phải "ngữ cảnh giữa các bước nấu" như thiết kế nhắm tới.

**Hội đồng có thể hỏi:**
- *"Context Fusion có quá nhỏ không?"* → **Bản lớn hơn khoảng 35 lần đã được thử và kém hơn.** Bản đầu tiên (135 triệu tham số, chạy ở
  độ rộng đầy đủ của Omni) không hơn bản không có Context Fusion trên dev (cấu hình đầy đủ: 49.42 so với 49.68), và làm tập kiểm tra giảm
  1.6 qua 3 seed (23.27 so với 24.88). Lưu ý: bản lớn còn khác ở hai điểm (phần chỉnh không được chuẩn hoá, tốc độ học của α), nên đây
  không phải phép so thuần về kích thước. Thêm các bằng chứng **gián tiếp**: mô-đun nhỏ vẫn thuộc được tập nhỏ (70.87 → 91.74); dev đạt
  đỉnh ngay ở epoch 2 rồi dừng sớm, tức là học nhanh rồi khớp quá vào train; cùng kích thước thì làm được với "cả video" (+3.4); dev tăng
  +2.6 mà tập kiểm tra chỉ +0.4, nên vấn đề nằm ở tổng quát hoá. Chưa thử các kích thước ở giữa (ví dụ 128 / 512).
- *"Có cài sai không?"* → Xem đoạn "Mô-đun chạy đúng" ở trên.
- *"Không có lợi ích thì đóng góp là gì?"* → Kết quả "không có lợi ích" (gọi là **kết quả âm**) **có đối chứng**: 3 seed, đối chứng xáo trộn, δ = 0, tháo từng phần. Nó trả lời
  RQ3 một cách có kiểm soát. Đóng góp 3 được viết lại cho đúng (mục 4.3).

Thêm một kiểm tra: *hàm mất mát* là thước đo sai mà mô-đun cố giảm khi train; cấu hình đầy đủ dùng một phần ở mức video và một phần ở mức
đoạn. Để xem phần mức đoạn có phải lý do mô-đun không giúp, ta train chỉ với phần mức video. Kết quả: không cách cắt nào tăng (không chạy
cho "cả video"; theo bước UniAV 24.49, cắt đều 17.72, theo đáp án 36.86;
khoảng tin cậy đều chứa 0).

**Trạng thái:** Tạm thời (Context Fusion dùng chuỗi ngữ cảnh của UniAV, kể cả ở dòng "cả video"); dòng theo đáp án là oracle.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c25 (kiểm tra tập nhỏ, dev, train lại seed 0), c26 (cột (c), Exp 2, đối chứng, chỉ
mức video), c30 (attention), c23 (trùng số của đường dữ liệu); 3 giá trị seed: `configs/final_protocol.json` `reference_current_uniav`,
khớp phép kiểm tra số tham chiếu (regression) ở c26.

---

## 4.8 Ranh giới bước nấu phải chính xác tới mức nào?  (RQ4)

**Trả lời ngắn:** Khi đoạn chồng với bước thật từ **khoảng một nửa** trở lên, đoạn theo bước thắng rõ rệt cắt đều theo số bước thật;
ở mức 0.3 chỉ còn thắng sát biên; giữa 0.3 và 0.5 chưa đo. UniAV hiện tại chồng trung bình 0.51 nhưng chỉ hiệu quả như mức 0.41.

**Đã kiểm tra bằng cách nào:** **tIoU** đo hai khoảng thời gian chồng lên nhau bao nhiêu, từ 0 (không chạm nhau) tới 1 (trùng khít). Lấy
ranh giới của người gán nhãn, dịch và co giãn ngẫu nhiên từng bước cho tới khi tIoU trung bình so với đáp án đạt 0.9 / 0.7 / 0.5 / 0.3
(sai lệch cho phép 0.02). Số bước và mọi bước vẫn giữ nguyên; chỉ ranh giới bị lệch. So mỗi mức với **cắt đều theo số bước thật** (cùng
số đoạn, không theo bước). Sau đó đặt các cách cắt thật lên cùng thang tIoU.

**Kết quả** (kho 394 video; chênh lệch so với cắt đều theo số bước thật, 18.91; cột chênh lệch in đúng 3 chữ số thập phân như log):

| tIoU trung bình | Đúng hạng 1 | Chênh lệch | Khoảng tin cậy 95% |
|---|---|---|---|
| 1.0 (đáp án) | 37.52 | +18.614 | [+16.49, +20.95] |
| 0.9 | 36.73 | +17.822 | [+15.58, +20.17] |
| 0.7 | 34.29 | +15.380 | [+13.37, +17.33] |
| 0.5 | 28.42 | +9.505 | [+7.66, +11.36] |
| 0.3 | 20.66 | +1.749 | [+0.00, +3.45] |

tIoU trung bình của một cách cắt: với mỗi bước thật, lấy tIoU lớn nhất giữa bước đó và các đoạn của cùng video, rồi lấy trung bình trên
mọi bước. Cách tính này áp dụng như nhau cho cách có 1 đoạn hay 22.66 đoạn mỗi video.

| Cách cắt thật | tIoU trung bình với đáp án | Đúng hạng 1 |
|---|---|---|
| Theo bước UniAV | 0.511 | 24.88 |
| Cắt mỗi 14 giây (22.66 đoạn) | 0.467 | 27.16 |
| Cắt đều theo số bước UniAV (R3) | 0.341 | 18.02 |
| Cả video | 0.064 | 8.32 |

**Điều này cho thấy:**
- Khi ranh giới lệch tới mức chồng 0.5, đoạn theo bước vẫn hơn cắt đều theo số bước thật chắc chắn (+9.5). Ở mức 0.3, lợi thế còn nhỏ
  (+1.75, khoảng tin cậy chạm 0). Ước tính R@1 của ranh giới bị làm nhiễu rơi xuống bằng mức của cắt đều theo số bước thật (18.91) khi tIoU còn khoảng 0.26. Đây
  là **ngoại suy** (kéo dài đoạn 0.3–0.5 xuống dưới), vì 0.26 thấp hơn mức nhiễu thấp nhất đã đo (0.3).
- UniAV chồng trung bình 0.511, nhưng kết quả của nó (24.88) chỉ ngang ranh giới đáp án bị làm nhiễu tới 0.41, tức nằm **dưới** đường
  cong khoảng 3.9 câu trên 100. Cắt đều R3 cũng nằm dưới đường cong khoảng 4.2 (tIoU 0.341: đường cong cho khoảng 22.2, thực tế 18.02).
  Vì vậy "nằm dưới đường cong" **không phải dấu hiệu riêng** của UniAV (giải thích ở dưới). Cả hai số là ước lượng thô (nội suy tuyến
  tính trên đường cong lõm, không có khoảng tin cậy), chỉ dùng để thấy chúng cùng bậc độ lớn. Cắt mỗi 14 giây nằm đúng trên đường (0.467:
  khoảng 27.1 so với 27.16), nhưng nó có số đoạn gấp khoảng 3 lần số bước, nên không cùng điều kiện với đường cong.
- Khoảng cách từ UniAV tới mức trần (đáp án): 12.64 [+10.55, +14.57] câu trên 100.

**Vì sao lại như vậy:** Đường cong **lõm**: độ dốc khoảng 38.8 ở đoạn 0.3–0.5, 29.4 ở 0.5–0.7 và 12.2 ở 0.7–0.9 (R@1 trên một đơn vị
tIoU). Nhiễu của thí nghiệm này làm tIoU từng bước tập trung quanh mức đích, và chỉ làm **lệch** ranh giới, không làm **mất** bước nào.
Các cách cắt thật cho tIoU từng bước phân tán hơn nhiều: có bước gần 0, có bước gần 1. Một cách giải thích (giả thuyết): nếu R@1 của mỗi câu chủ
yếu phụ thuộc vào tIoU của bước đó theo một hàm lõm, thì cùng tIoU trung bình mà phân tán hơn sẽ cho R@1 thấp hơn (bất đẳng thức
Jensen), nên cách cắt nào có tIoU từng bước phân tán hơn nhiễu cũng được dự đoán nằm dưới đường cong. Giả thuyết này cần ba điều chưa
đo trực tiếp: đường cong đo theo tIoU **trung bình** (và chỉ từ 0.3 trở lên), không theo tIoU từng câu; nhiễu tập trung quanh mức đích;
R@1 của một câu chỉ phụ thuộc tIoU của bước đó. Bỏ sót bước (27.8% với UniAV) là trường hợp cực của sự
phân tán, nên con số 3.9 **nhất quán** với giả thuyết bỏ sót nhưng **không tự chứng minh** giả thuyết đó. Bằng chứng trực tiếp là bảng
theo nhóm câu ở mục 4.9. Bảng theo nhóm tIoU từng bước bên dưới kiểm tra trực tiếp cách giải thích này.

Mức nhiễu và các cách cắt thật dùng **cùng một cách tính** tIoU trung bình (với mỗi bước thật, lấy max theo các đoạn của cùng video;
hàm `best_tiou` trong `omni_retrieval/partitions_uemr.py`), nên hai trục x cùng thang.

**Kiểm tra theo nhóm tIoU từng bước.** Chia 3.030 câu theo tIoU của bước cần tìm với đoạn chồng nhiều nhất của từng cách cắt, rồi so
với cắt theo đáp án trên **cùng nhóm câu**. Hai cách cắt có nhóm câu khác nhau (ví dụ nhóm 0.5–0.7 có 673 câu với R6 nhưng 466 câu với
R3), nên không so R@1 thô giữa hai cột; so **chênh lệch tới cắt theo đáp án**, vì nó khử phần độ khó riêng của từng nhóm câu. Chênh lệch có khoảng tin cậy 95% (bootstrap theo video).

| tIoU của bước | Theo bước UniAV: số câu | R@1 | Theo đáp án − theo bước UniAV | Cắt đều R3: số câu | R@1 | Theo đáp án − cắt đều R3 |
|---|---|---|---|---|---|---|
| < 0.3 (bỏ sót) | 843 (27.8%) | 7.00 | +32.74 [+29.03, +36.79] | 1.433 (47.3%) | 8.72 | +31.12 [+28.29, +34.01] |
| 0.3–0.5 | 586 (19.3%) | 22.35 | +12.29 [+7.67, +16.90] | 930 (30.7%) | 23.01 | +13.55 [+9.78, +17.72] |
| 0.5–0.7 | 673 (22.2%) | 35.36 | +3.42 [−0.48, +7.70] | 466 (15.4%) | 29.61 | +5.15 [+0.20, +10.05] |
| ≥ 0.7 | 928 (30.6%) | 35.13 | +1.29 [−1.76, +4.19] | 201 (6.6%) | 34.33 | −2.49 [−9.05, +4.59] |

Quy tắc đọc được ghi trong notebook trước khi chạy: cách giải thích được ủng hộ nếu (1) chênh với đáp án giảm dần khi tIoU tăng, ở cả
hai cách cắt, và (2) trong cùng nhóm tIoU, khoảng tin cậy của chênh lệch ở hai cách cắt chồng nhau. **Cả hai điều đều đúng.** Đây là
phép kiểm tra **yếu**: các khoảng tin cậy rộng, và "chồng nhau" chỉ cho thấy dữ liệu **không phân biệt được** hai cách cắt, không chứng
minh chúng bằng nhau. Vậy: **ở cùng mức chồng với bước, không thấy đoạn UniAV khác đoạn cắt đều R3.** Cắt theo bước UniAV hơn R3 chủ
yếu vì nhiều bước của nó rơi vào nhóm tIoU cao (30.6% bước chồng từ 0.7, so với 6.6%; 27.8% bị bỏ sót, so với 47.3%). Kết quả này phù
hợp với cách giải thích bằng sự phân tán của tIoU từng bước, và không cho thấy dấu hiệu của một nguyên nhân riêng của UniAV. Ngoài ra,
khi đoạn chồng bước từ 0.5 trở lên, chưa phát hiện được khác biệt giữa cắt theo bước UniAV và cắt theo đáp án (hai khoảng tin cậy chứa 0).

**Hội đồng có thể hỏi:**
- *"Nhiễu ngẫu nhiên khác lỗi thật của detector?"* → Đúng, hai loại lỗi khác nhau. UniAV nằm dưới đường cong 3.9 câu trên 100,
  nhưng cắt đều R3 cũng nằm dưới khoảng 4.2, nên khoảng này không đo riêng lỗi của detector. Bảng theo nhóm tIoU không thấy đoạn UniAV
  kém đoạn cắt đều ở cùng mức chồng; khác biệt nằm ở chỗ detector bỏ sót bước nào và chồng mỗi bước bao nhiêu. Phần nằm ở các câu bị
  bỏ sót được đo ở mục 4.9: khoảng 72% khoảng cách tới mức trần.
- *"Một mẫu nhiễu có đủ không?"* → Mỗi mức chỉ dùng một seed nhiễu; đây là hạn chế (mục 4.13). Khoảng tin cậy theo video vẫn tính cho
  từng mức.
- *"Thiết kế có hai đường, có và không có Context Fusion?"* → Chỉ chạy đường không có Context Fusion, vì Context Fusion không tạo lợi ích
  ở mọi cách cắt có nhiều đoạn (mục 4.7). Câu hỏi "Context Fusion có giúp bền hơn với ranh giới kém không" vì vậy chỉ được trả lời gián
  tiếp: không có lợi ích ở cả cắt theo bước UniAV (tIoU 0.51) lẫn theo đáp án (tIoU 1.0).

**Trạng thái:** Đường cong: oracle. Điểm của cắt theo bước UniAV và cắt đều: tạm thời. Cả video, cắt mỗi 14 giây: chính thức.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c21 (bảng mức nhiễu, chênh lệch, `significant down to tIoU`, `R6 equivalent tIoU`,
`break-even`, `R6 below curve`), c12 (tIoU trung bình từng cách cắt), c18 (R7 − R6); bảng theo nhóm tIoU: cell B3, output ở
`notebooks/log/uemr/uemr_ket_qua_bo_sung_B3.md` (Drive: `results/056d388e/gap_by_tiou_bin.csv`). Điểm của cắt đều và mỗi 14 giây so với đường
cong, độ dốc từng đoạn: nội suy tuyến tính từ c21 và c12 (cách tính như số 3.9).

---

## 4.9 Lỗi nằm ở đâu: phía UniAV hay phía Omni?

**Trả lời ngắn:** Cả hai, và không tách bạch. **UniAV còn bỏ sót bước nấu, và Omni (bản đã dạy thêm) nhạy với ranh giới.** Cắt theo bước
UniAV đạt khoảng 2/3 mức trần mà cắt theo đáp án đạt được, và khoảng 72% khoảng cách tới mức trần (12.64) nằm ở các câu có bước bị
UniAV bỏ sót.

**Mốc so sánh.** Cắt theo đáp án (oracle) đúng hạng 1 ở 37.5 trên 100 câu. Đó là **mức trần của Omni trên dữ liệu này khi biết đúng
ranh giới** theo người gán nhãn: một mốc oracle, không phải cận trên tuyệt đối, vì có câu cắt theo bước UniAV thắng (mục 4.11). Phần còn thiếu tới 100 đến từ nhiều nguồn: dữ liệu mơ hồ, độ khó của câu tìm kiếm, kích thước kho video và giới hạn của
Omni. Ba nguồn đo được (đây là các mối liên hệ quan sát được, không cho biết mỗi nguồn chiếm bao nhiêu phần của khoảng thiếu):
- **Bước gần trùng ở nhiều video.** Kết quả theo đáp án giảm mạnh theo số video khác có bước gần trùng với câu tìm kiếm ("gần trùng":
  mô tả của hai bước có cosine ≥ 0.8 theo một bộ mã hoá câu độc lập, không phải Omni; `omni_retrieval/analysis.py`): 45.81 khi không
  có video nào (1.419 câu), 36.55 khi có 1–2 (829 câu), 26.13 khi có 3–9 (643 câu), 12.21 khi có 10–29 (131 câu), và 0 khi có từ 30
  trở lên (8 câu, mọi cách cắt đều 0).
- **Bước lặp lại trong cùng video: nguồn nhỏ.** 298 lượt tìm đúng video ở hạng 1 nhưng đoạn khớp nhất chồng với bước cần tìm dưới 0.5 (gộp hai
  cách cắt: 57 lượt với cắt theo đáp án, tức 1.137 câu đúng video trừ 1.080 câu chỉ đúng khoảnh khắc; 241 lượt với cắt theo bước UniAV,
  tức 754 − 513; một câu có thể được đếm ở cả hai). Với mỗi lượt, lấy bước thật chồng nhiều nhất với đoạn bị chọn, rồi xem bước đó
  là chính bước cần tìm hay một bước khác, và mô tả của nó có gần trùng chữ với câu tìm kiếm không (tỉ lệ từ chung Jaccard ≥ 0.5):

  | Cách cắt | Lượt | Chính bước cần tìm, ranh giới lệch | Bước khác, mô tả gần trùng | Bước khác, mô tả không gần trùng |
  |---|---|---|---|---|
  | Theo đáp án | 57 | 0 | 12 | 45 |
  | Theo bước UniAV | 241 | 148 | 13 | 80 |
  | Tổng | 298 | 148 | 25 | 125 |

  Nếu chỉ đếm "mô tả gần trùng chữ" thì được 173 lượt, nhưng 148 trong số đó là cắt theo bước UniAV chọn **đúng bước** với đoạn lệch
  (tIoU trung bình 0.354 với bước); mô tả trùng vì đó chính là câu tìm kiếm. Số lượt thật sự nhầm sang một bước khác có mô tả gần trùng
  chỉ là **25 trên 298** (12 trên 57 với cắt theo đáp án). Với cắt theo bước UniAV, phần lớn "đúng video, sai đoạn" (148 trên 241) là
  lỗi ranh giới, không phải bước lặp. Ví dụ đoạn bị chọn trùng với một bước khác của cùng video, có mô tả gần trùng chữ với câu tìm kiếm: "remove and drain the potatoes from the fryer" bị khớp vào bước "remove and drain the fish from the
  fryer" của cùng video; "continue chopping the cabbage" xuất hiện hai lần trong cùng một video.
- **Kho lớn hơn** (ghi chú độ bền, không phải nguồn của khoảng thiếu trong kho 394): cắt theo đáp án còn 34.16 với kho 498 video.

**Khoảng cách tới mức trần: 12.64 câu trên 100** [+10.55, +14.57], tức khoảng cách giữa cắt theo đáp án và cắt theo bước UniAV. Khoảng
cách này sinh ra vì UniAV tìm bước chưa tốt, nhưng độ lớn của nó còn phụ thuộc việc Omni nhạy với ranh giới (đoạn "Hai phía không tách
bạch" bên dưới), nên không gán trọn 12.64 cho riêng UniAV.

**Bỏ sót là dạng lỗi rõ nhất.** Bằng chứng chính là bảng theo nhóm câu bên dưới: khoảng 72% của 12.64 nằm ở các câu có bước bị bỏ sót.
- UniAV bỏ sót 27.8% bước ở tập kiểm tra.
- Ở các câu bị **lật** (cách khác đúng, cắt theo bước UniAV sai), tỉ lệ bước bị bỏ sót cao hơn nhiều so với chiều ngược lại. Nhưng đây
  phần lớn là **hiệu ứng chọn mẫu**: khi bước bị bỏ sót, cắt theo bước UniAV gần như luôn sai (R@1 7.00), nên mọi nhóm có điều kiện
  "theo bước UniAV đúng" tự nhiên có ít bước bị bỏ sót (khoảng 7.8%, = 59 / 754). So với mốc công bằng hơn là mọi câu mà cắt theo bước
  UniAV sai (khoảng 34.4%, = 784 / 2.276), các nhóm bị lật **không** cao hơn. Bảng này vì vậy chỉ mô tả, không phải bằng chứng riêng:

  | Nhóm câu | Bước bị UniAV bỏ sót (%) |
  |---|---|
  | cả video đúng, theo bước UniAV sai / ngược lại | 32.21 / 7.68 |
  | mỗi 14 giây đúng, theo bước UniAV sai / ngược lại | 34.73 / 9.40 |
  | cắt đều đúng, theo bước UniAV sai / ngược lại | 25.27 / 8.11 |

- Bước ngắn dưới 10 giây chịu thiệt nhiều: theo đáp án 44.29, theo bước UniAV 21.05 (bảng đủ theo độ dài ở mục 4.6).

**Hai phía không tách bạch.** Omni (bản đã dạy thêm) nhạy với ranh giới. Ở mục 4.8, giữ đủ mọi bước và chỉ làm lệch ranh giới tới mức
chồng 0.7 đã mất 3.23 câu, tới 0.5 mất 9.11 câu so với ranh giới đáp án (trùng số với phần đóng góp ở bảng theo nhóm câu bên dưới
chỉ là ngẫu nhiên). Một dấu hiệu nữa: khoảng cách giữa theo đáp án
và theo bước UniAV là 6.37 với Omni chưa dạy thêm (25.31 − 18.94) và 12.64 với Omni đã dạy thêm. Dạy thêm làm cả hai tăng (theo đáp án +12.21, theo bước
UniAV +5.94), nên đây là "lợi ích của dạy thêm tập trung ở đoạn đúng ranh giới", không phải "Omni phạt đoạn lệch". Một cách giải thích
(giả thuyết): Omni được dạy thêm trên clip cắt theo đáp án nên khớp tốt nhất với đoạn đúng ranh giới. Đường cong nhiễu chỉ đo với Omni
đã dạy thêm, nên chưa biết bản chưa dạy thêm nhạy với ranh giới tới đâu.

**Phần của 12.64 nằm ở câu có bước bị bỏ sót.** Chia 3.030 câu thành hai nhóm theo việc bước cần tìm có bị UniAV bỏ sót
không (không đoạn nào trùng từ 0.3 trở lên), rồi tính khoảng cách trong từng nhóm. Số trong hai cột giữa là R@1 (đúng hạng 1 trên 100 câu
của nhóm).

| Nhóm câu | Số câu (tỉ lệ) | Theo bước UniAV | Theo đáp án | Chênh trong nhóm (khoảng tin cậy 95%) | Đóng góp vào 12.64 (khoảng tin cậy 95%) |
|---|---|---|---|---|---|
| Bước bị UniAV bỏ sót | 843 (27.82%) | 7.00 | 39.74 | +32.74 [+29.03, +36.79] | **+9.11** [+7.87, +10.36], tức 72% |
| Bước không bị bỏ sót | 2.187 (72.18%) | 31.78 | 36.67 | +4.89 [+2.78, +7.21] | +3.53 [+1.82, +5.11], tức 28% |

"Đóng góp" của một nhóm = tỉ lệ câu của nhóm × chênh lệch trong nhóm; hai dòng cộng lại bằng 12.64 (9.11 + 3.53).

**Quy tắc đọc.** Trước khi chạy, quy tắc chỉ được ghi ở dạng định tính: "nhóm bỏ sót đóng góp **phần lớn** của 12.64" thì kết luận là
tăng recall. Ngưỡng bằng số (đóng góp trên 6.32, một nửa của 12.64; tương đương chênh trong nhóm trên khoảng 22.7) được đặt **sau** khi
chạy. Kết luận không phụ thuộc vào ngưỡng này: cả khoảng tin cậy của phần đóng góp [+7.87, +10.36] nằm trên 6.32, và chênh trong nhóm
[+29.03, +36.79] nằm trên 22.7.

**Đọc bảng:** khi UniAV bỏ sót bước cần tìm, cắt theo bước UniAV gần như mất hẳn câu đó: chỉ đúng hạng 1 ở 7 trên 100 câu, so với khoảng
40 khi cắt theo đáp án. Khi UniAV không bỏ sót, nó chỉ kém đáp án khoảng 4.9 câu trên 100. Vì vậy khoảng 72% khoảng cách tới mức trần
nằm ở 28% số câu có bước bị bỏ sót. Phần còn lại (3.53, khoảng tin cậy không chứa 0) nằm ở các câu có đoạn chồng với bước nhưng chưa
khớp, tức gắn với ranh giới chưa chính xác. Câu có bước bị bỏ sót không khó hơn với Omni: theo đáp án, nhóm này đạt 39.74, cao hơn mức
chung 37.52, nên khoảng cách lớn trong nhóm không phải vì nhóm gồm câu khó hơn. Giới hạn của phép chia: nó chỉ xét bước cần tìm trong
**video đúng**. Cách cắt của 393 video còn lại trong kho cũng khác giữa hai bên, và một câu có thể bị mất vì đoạn của một video khác
chấm cao hơn; phần này không tách được ở đây.

Bảng theo nhóm tIoU ở mục 4.8 tách tiếp phần 3.53: 2.38 [+1.51, +3.24] nằm ở các câu có bước chồng 0.3–0.5 (gần bỏ sót); hai nhóm từ
0.5 trở lên đóng góp 0.76 [−0.17, +1.70] và 0.40 [−0.54, +1.44]; mỗi khoảng tin cậy chứa 0, còn tổng 1.16 chưa có khoảng tin cậy riêng. Cộng lại, khoảng 91% khoảng cách tới mức
trần (9.11 + 2.38 = 11.49 trên 12.64) nằm ở các câu có bước chồng dưới 0.5.

Ngưỡng "bỏ sót" 0.3 là một lựa chọn: một
đoạn chồng 0.3–0.5 với bước vẫn được tính là "không bỏ sót"; những trường hợp gần bỏ sót này chiếm 2.38 trong 3.53 (đoạn trên).

**Kết luận:** tăng **recall** của UniAV, tức bỏ sót ít bước hơn, là hướng có tiềm năng lớn nhất theo ước lượng: khoảng 72% khoảng cách tới mức trần (9.11 trên
12.64) nằm ở các câu có bước bị bỏ sót. Độ chính xác ranh giới vẫn có giá trị: nhóm không bỏ sót còn đóng góp 3.53 [+1.82, +5.11], phần lớn
ở các bước chỉ chồng 0.3–0.5 (2.38); khi đoạn đã chồng bước từ 0.5, chưa phát hiện được phần còn lại. Ranh giới quan trọng nhất
với bước ngắn dưới 10 giây (mục 4.6). Lưu ý: 9.11 là khoảng cách tới mức trần, chưa phải phần lấy lại được nếu chỉ tăng recall.
Nếu các bước đang bị bỏ sót được tìm ra và đạt mức như nhóm không bỏ sót hiện nay (31.78), lợi ích khoảng 0.278 × (31.78 − 7.00) ≈
**6.9** câu trên 100 (ước lượng, không có khoảng tin cậy). Con số này vẫn lớn hơn toàn bộ phần nằm ở nhóm không bỏ sót (3.53), nhưng nó giả định các đoạn
mới không làm video nhiễu chấm cao hơn. Báo cáo chưa thử hạ ngưỡng của UniAV để đổi lấy recall; bằng chứng thực nghiệm duy nhất về
việc thêm đoạn là gộp bước UniAV với mỗi 14 giây (29.77, mục 4.10). Con số 12.64
đo với UniAV hiện tại; nó có thể thay đổi khi UniAV được train lại đúng quy trình.

**Trạng thái:** Tạm thời (mọi số có cắt theo bước UniAV); mức trần và các số theo đáp án là oracle.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c18 (R7 − R6, R7 kho 498, cột (a)), c19 (theo số bước gần trùng, theo độ dài), c21
(mức nhiễu, R6 dưới đường cong), c30 (nhóm câu, đúng video sai đoạn và ví dụ), c10 (tỉ lệ bỏ sót); tách 298 lượt và nhóm tIoU: cell B3, output ở
`notebooks/log/uemr/uemr_ket_qua_bo_sung_B3.md` (Drive: `results/056d388e/same_video_wrong_split.csv`, `gap_by_tiou_bin.csv`); bảng theo nhóm bỏ sót: cell B2, output ở `notebooks/log/uemr/uemr_ket_qua_bo_sung_B2.md` (Drive:
`results/056d388e/r7_r6_by_miss.csv`).

---

## 4.10 Cách chấm điểm và chi phí  (Exp 5)

**Trả lời ngắn:** Cách chấm đơn giản nhất, "đoạn khớp nhất", là tốt nhất cho cắt theo bước UniAV. Dùng độ tin cậy của UniAV không cho cải thiện
phát hiện được. Lưu vector ở dạng nén 8 bit không cho mất mát phát hiện được (chênh tối đa 0.10, các khoảng tin cậy chứa 0). So với cắt mỗi 14 giây, cắt theo bước UniAV cần ít hơn khoảng 3 lần bộ nhớ và thời
gian tìm trên CPU, và ít hơn khoảng 2 lần thời gian chuẩn bị mỗi video khi tính cả thời gian chạy UniAV. Đổi lại, R@1 kém mỗi 14 giây
2.28 (24.88 so với 27.16), nhưng chỉ ra đúng khoảnh khắc gấp khoảng 1.7 lần (16.93 so với 9.87).

Dev là kho 104 video nên số trên dev (khoảng 50) cao hơn nhiều so với tập kiểm tra (kho 394 video); không so trực tiếp hai thang. Lưu
ý: UniAV (Athena) đã học dev khi train (mục 4.2), nên trên dev ranh giới và độ tin cậy của UniAV là trong mẫu; việc chọn β và cách gộp
trên dev vì vậy có thể bị lệch, và nên chọn lại sau khi train lại UniAV.

**Độ tin cậy của UniAV làm điểm cộng.** Điểm của đoạn i = cosine(câu, đoạn i) + β · log(độ tin cậy UniAV của đoạn i); cosine nằm trong
[−1, 1], độ tin cậy trong (0, 1]. β được chọn trên dev (hoà thì lấy β nhỏ hơn); tập kiểm tra chỉ dùng để báo cáo.

| β | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|
| Đúng hạng 1 trên dev | 49.68 | 49.81 | **50.06** | 50.06 | 48.90 | 46.05 |

Với β = 0.01, tập kiểm tra tăng +0.23 [−0.17, +0.64] (kho 498: +0.20 [−0.23, +0.62]): không có ý nghĩa. β lớn hơn làm kém đi ngay trên dev.

**Cách gộp điểm của các đoạn trong một video.** Ngoài "đoạn khớp nhất" (max), thử "trung bình k đoạn khớp nhất" (top-k) và "max mềm"
(LSE: điểm video = τ · log((1/K) · Σ exp(s_i / τ)) với K đoạn có điểm s_i; chia cho K nên không thiên vị video nhiều đoạn; gần với max
khi τ nhỏ).

| Cách gộp (cắt theo bước UniAV) | Dev | Kiểm tra |
|---|---|---|
| **max** | **49.68** | **24.88** |
| LSE τ = 0.01 | 49.42 | 24.26 |
| LSE τ = 0.02 | 47.74 | 22.44 |
| LSE τ = 0.05 | 36.87 | 15.18 |
| LSE τ = 0.1 | 28.85 | 11.12 |
| trung bình 2 đoạn tốt nhất | 41.66 | 18.55 |
| trung bình 3 đoạn tốt nhất | 37.00 | 16.11 |

Với cắt theo bước UniAV và theo đáp án, max tốt nhất trên cả dev và tập kiểm tra. Hai trường hợp khác: với cắt đều, LSE τ = 0.01 nhỉnh hơn
max trên dev (36.61 so với 36.35); với cắt mỗi 14 giây, LSE τ = 0.01 nhỉnh hơn max trên tập kiểm tra (27.95 so với 27.16) nhưng thua trên
dev (48.77 so với 49.42), nên theo quy tắc chọn trên dev vẫn dùng max. Một cách giải thích cho việc top-k kém: câu tìm kiếm chỉ tả một
bước, nên đoạn khớp thứ hai thường là bước khác và chỉ làm loãng điểm.

**Chi phí** (498 video; đo trên máy Colab có GPU A100; độ trễ CPU phụ thuộc máy nên chỉ so tương đối giữa các dòng):

| Cách cắt | Số vector | Bộ nhớ fp16 / 8 bit (MB) | 1 câu trên CPU (ms) | 1 câu trên GPU (ms) | Thời gian Omni mỗi video (giây) |
|---|---|---|---|---|---|
| Cả video | 498 | 3.4 / 1.7 | 0.11 | 0.12 | 0.26 |
| Theo bước UniAV | 3.579 | 24.5 / 12.2 | 0.82 | 0.12 | 1.87 |
| Theo đáp án (oracle) | 3.803 | 26.0 / 13.0 | 0.85 | 0.12 | 1.99 |
| Mỗi 14 giây | 11.316 | 77.4 / 38.7 | 2.63 | 0.12 | 5.91 |
| Gộp bước UniAV và mỗi 14 giây | 14.895 | 101.8 / 51.0 | 3.43 | 0.13 | 7.78 |

**Nén 8 bit:** lưu mỗi số bằng 1 byte thay vì 2 byte, giảm một nửa bộ nhớ.

| Cách cắt | Chênh lệch khi nén | Khoảng tin cậy 95% |
|---|---|---|
| Theo bước UniAV | −0.03 | [−0.16, +0.07] |
| Mỗi 14 giây | −0.10 | [−0.30, +0.07] |
| Theo đáp án | −0.03 | [−0.23, +0.16] |
| Gộp hai cách | −0.07 | [−0.29, +0.13] |

**Thời gian chạy UniAV mỗi video: chưa có số đo riêng.** Log cũ ghi khoảng 0.55 giây (tìm bước và lưu chuỗi ngữ cảnh) cộng 0.47 giây (hậu
xử lý) mỗi video trên GPU T4, chưa tách khỏi thời gian đọc ghi Drive.

**Chi phí chuẩn bị khi tính cả UniAV.** Cắt theo bước UniAV phải chạy UniAV, còn cắt mỗi 14 giây thì không. Cộng khoảng 1.0 giây UniAV (số
log cũ ở trên) với 1.87 giây Omni, cắt theo bước UniAV cần khoảng 2.9 giây mỗi video, so với 5.91 giây của cắt mỗi 14 giây, tức vẫn ít
hơn khoảng 2 lần. Đây là ước lượng: thời gian UniAV đo trên GPU T4 và gồm cả đọc ghi Drive, thời gian Omni đo trên A100.

**Chỉ ra đúng khoảnh khắc (phần RQ5 đã có trên YouCook2).** Cùng một index vừa xếp hạng video vừa trả về đoạn khớp nhất, nên đo được luôn
mức chỉ đúng khoảnh khắc. "Đúng cả video lẫn đoạn" nghĩa là video đúng ở hạng 1 và đoạn khớp nhất chồng ít nhất 0.5 với bước cần tìm.

| Cách cắt | Đúng cả video lẫn đoạn (trên 100 câu) | Khi video đúng ở hạng 1: đoạn đúng (%) |
|---|---|---|
| Theo bước UniAV | 16.93 | 68.04 |
| Mỗi 14 giây | 9.87 | 36.33 |
| Cắt đều | 5.81 | 32.23 |
| Gộp hai cách | 15.38 | 51.66 |
| *Theo đáp án (oracle)* | *35.64* | *94.99* |

**Trạng thái:** Tạm thời (mọi dòng có UniAV); cả video, mỗi 14 giây: chính thức; theo đáp án: oracle.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c28 (β, cách gộp, chi phí, nén 8 bit), c18 (chỉ đúng khoảnh khắc), c10 (thời gian
UniAV trong log cũ).

---

## 4.11 Phân tích định tính

Ba câu hỏi của `UEMR_FINAL.md` §8.

**1. Những câu mà "cả video" thắng "theo bước UniAV" có gì chung?** Bước bị UniAV bỏ sót hay gặp ở các câu này, dù chưa phải đa số. Cắt theo bước UniAV đúng
mà cả video sai ở 651 câu; chiều ngược lại chỉ 149 câu. Trong 149 câu đó, 32.2% là bước UniAV bỏ sót, so với 7.7% ở 651 câu kia; nhưng
khác biệt này phần lớn do chọn mẫu (mục 4.9): ở mọi câu mà cắt theo bước UniAV sai, tỉ lệ bỏ sót là khoảng 34%. Bước cần tìm cũng hơi
dài hơn (trung vị 17 so với 16 giây). Cả video không có khoảng trống giữa các đoạn, dù Omni chỉ lấy mẫu thưa (8 khung hình, mục 4.1).

Ví dụ: "cut up the banana", bước dài 6 giây ở giây 341–347. Bước này nằm trong khoảng trống giữa một đoạn UniAV kết thúc khoảng giây 320 và
đoạn sau bắt đầu khoảng giây 362. Cắt theo bước UniAV xếp video đúng ở hạng 236; cắt mỗi 14 giây và cắt theo đáp án đều xếp hạng 1. Trong
10 ví dụ "cả video hoặc mỗi 14 giây thắng" được chọn tự động theo hạng chênh nhiều nhất, 8 ví dụ có bước nằm hoàn toàn ngoài mọi đoạn
UniAV. Cách chọn này thiên về đúng các ca bỏ sót (hạng tụt sâu), nên 8/10 không phải tần suất. Lưu ý: 2 trong 8 ví
dụ đó thuộc video `wii9jNiNl9Y`, video có chuỗi ngữ cảnh lệch 9.84 giây so với file video (mục 4.13), nên ở video này đoạn UniAV có thể
bị lệch thời gian chứ không chỉ bỏ sót. Bỏ hai ví dụ đó, còn 6 trong 8 ví dụ còn lại.

**2. Các bước giống nhau có tách được không?** Một phần: mới đo được quy mô vấn đề, chưa đo việc tách. Quy mô **trong cùng video**
nhỏ: trong 298 lượt tìm đúng
video nhưng chọn sai đoạn (gộp hai cách cắt, mục 4.9), chỉ 25 lượt nhầm sang một bước khác có mô tả gần trùng chữ với câu tìm kiếm
(12 với cắt theo đáp án, 13 với cắt theo bước UniAV). 148 lượt khác trông giống vậy thật ra là cắt theo bước UniAV chọn đúng bước nhưng
ranh giới lệch. Ví dụ "remove and drain the potatoes from the fryer" và
"remove and drain the fish from the fryer" trong cùng một video. Giữa các video, kết quả giảm mạnh khi có nhiều video chứa bước gần trùng
(mục 4.9). **Chưa đo riêng** Context Fusion có tách được các cặp này không; ở mức tổng thể Context Fusion không tạo khác biệt (mục 4.7).

**3. Context Fusion nhìn vào đâu?** Ở 3 video có nhiều đoạn nhất, 82–86% sự chú ý của bước 1 của mô-đun (nhìn chuỗi ngữ cảnh, mục 4.7)
rơi vào chính khoảng thời gian của đoạn (0.824, 0.856, 0.857). Chưa có mốc so sánh: vùng mô-đun được nhìn chỉ rộng hơn đoạn 5 giây mỗi
bên, nên chú ý chia đều cũng dồn phần lớn vào trong đoạn; với 3 video này đây chỉ là minh hoạ. Mô-đun nhìn chủ yếu vào bên trong đoạn, ít dùng ngữ cảnh ngoài đoạn mà thiết kế nhắm tới; theo giả thuyết ở mục
4.7, phần bên trong đoạn lại là thông tin Omni đã có.

Một ví dụ ngược chiều đáng ghi: với câu "sprinkle green onions on top", cắt theo bước UniAV xếp video đúng hạng 1, còn cắt theo đáp án xếp
hạng 65. Đoạn UniAV dài hơn bước thật một chút nhưng khớp câu tìm kiếm tốt hơn. Ranh giới của người gán nhãn không phải lúc nào cũng tối
ưu cho tìm kiếm, dù trung bình nó tốt hơn nhiều.

Các ví dụ được chọn tự động (hạng chênh nhiều nhất), không chọn tay; số lượng nhỏ, chỉ để minh hoạ.

**Trạng thái:** Tạm thời.
**Nguồn:** `notebooks/log/uemr/uemr_final_AZ.ipynb` c30 (nhóm câu, đúng video sai đoạn, attention); ví dụ "cut up the banana", "sprinkle
green onions on top", 8/10 ví dụ: `notebooks/log/uemr/NHAN_XET_KET_QUA_A4.md` §2.2, §2.3 (log `notebooks/log/uemr/A2_A4/uemr_qualitative_colab.ipynb`).

---

## 4.12 Vì sao kết quả đáng tin (phần dành cho hội đồng)

### (a) Các biện pháp kiểm soát

| Biện pháp | Loại bỏ cách giải thích nào | Bằng chứng |
|---|---|---|
| Cùng Omni, cùng câu tìm kiếm, cùng kho, cùng cách chấm cho mọi cách cắt | khác biệt do mô hình hay cách chấm | thiết kế, mục 4.1 |
| Cắt đều, ngẫu nhiên, đa dạng nội dung dùng đúng số đoạn của UniAV; cắt đều theo số bước thật cho đáp án | "nhiều vector hơn thì dễ khớp hơn" | mục 4.6 |
| Trung bình các bước (cùng thông tin, nén lại) | "lợi ích do có thông tin bước, không do giữ riêng" | mục 4.5 |
| Cột (a): Omni chưa dạy thêm | "cả video thua vì Omni dạy thêm trên clip ngắn" | mục 4.5: +11.06 [+9.21, +12.89] |
| Trung bình các bước và cắt đều xem đúng số khung hình như theo bước UniAV | "theo bước UniAV hơn chỉ vì Omni được xem nhiều khung hình hơn (khoảng 58 so với 8)" | mục 4.5: trung bình các bước chỉ ngang cả video (+0.92); cắt đều thua theo bước UniAV 6.86 |
| Bước bị dời chỗ (giữ độ dài, đổi vị trí) | "lợi ích do độ dài đoạn" | mục 4.6: +7.52 |
| Hồi quy độ phủ | "lợi ích do phủ nhiều thời lượng hơn" | mục 4.6: +8.99 khi độ phủ bằng nhau |
| Khi train Context Fusion, mỗi **lô** (nhóm câu học cùng một lúc) chỉ có một câu của mỗi video | hai câu cùng video bị coi là "sai" của nhau | danh sách kiểm tra cuối notebook A-Z (`UEMR_FINAL.md` §14) |
| Ngưỡng UniAV, β, cách gộp, thời điểm dừng đều chọn trên dev; không lựa chọn nào dựa vào tập kiểm tra | chọn thiết lập trên tập test | mục 4.2; ba ngoại lệ (hai ở mô hình nền, một là kích thước Context Fusion): mục 4.13 |
| Khoảng tin cậy bootstrap theo video, 1.000 lần | chênh lệch do may rủi; câu cùng video không độc lập | mục 4.2 |
| 3 seed cho Context Fusion | kết quả do một lần khởi tạo may mắn | mục 4.7 |
| Đối chứng xáo trộn chuỗi ngữ cảnh, δ = 0 | phần tăng do thêm tham số | mục 4.7 |
| Kho phụ 498 video | kết quả chỉ đúng với một kho | mục 4.5, 4.6, phụ lục C |
| Tái lập trọn vẹn: một notebook, một phiên bản code, 79 số khớp, 89 kiểm thử tự động | kết quả không chạy lại được | dựng lại mọi cách cắt từ đầu cho ra file trùng hệt; 39 clip lấy mẫu (trong 55.557 clip của cache) mã hoá lại cho vector trùng (độ tương đồng thấp nhất 1.00000); train lại Context Fusion seed 0 được 25.21 so với 25.18 |
| Báo cáo kết quả âm và trái giả thuyết | chọn lọc kết quả đẹp | mục 4.6 (ba kết quả), 4.7, 4.10 |

### (b) Câu hỏi phản biện dự kiến

| # | Câu hỏi | Trả lời ngắn | Chi tiết |
|---|---|---|---|
| 1 | Nhiều vector thì dễ khớp trùng hợp, sao biết là nhờ bước nấu? | Cùng số đoạn, cắt theo bước UniAV vẫn hơn cắt đều +6.86, ngẫu nhiên +8.71, đa dạng nội dung +10.69; dời chính các đoạn đó đi thì mất 7.52 | 4.6 |
| 2 | Cả video thua vì Omni được dạy thêm trên clip ngắn? | Với Omni chưa dạy thêm, cắt theo bước UniAV vẫn hơn cả video +11.06 [+9.21, +12.89]; thứ tự không đổi | 4.5 |
| 3 | UniAV có thấy dữ liệu đánh giá khi train không? Nếu có thì số bị thổi phồng theo hướng nào? | Có: UniAV (Athena) đã thấy toàn bộ dev khi train và được chọn phiên bản theo tập kiểm tra. Hướng: số của cắt theo bước UniAV có thể cao hơn thực tế. Cắt đều và cắt ngẫu nhiên cũng dùng UniAV, nhưng chỉ dùng số bước; rò rỉ chủ yếu làm **ranh giới** của UniAV tốt hơn, nên các khoảng cách của RQ2 nhiều khả năng là **cận trên** (giới hạn trên của khoảng cách thật; chưa kiểm chứng). RQ1 khó đảo chiều vì khoảng cách rất lớn (+16.6). Cần train lại UniAV để có số chính thức | 4.13 |
| 4 | Cắt mỗi 14 giây cao hơn, sao vẫn nói cắt theo bước có giá trị? | Claim là "tốt nhất khi cùng số đoạn". Cắt mỗi 14 giây dùng gấp khoảng 3 lần vector, chỉ ra khoảnh khắc kém hơn nhiều (9.87 so với 16.93), và ở kho 498 không còn hơn có ý nghĩa; gộp hai cách cho kết quả cao nhất | 4.6 |
| 5 | Context Fusion không có lợi ích, vậy đóng góp là gì? | Nghiên cứu có kiểm soát (RQ1, RQ2), phân tích độ nhạy ranh giới (RQ4), và một kết quả âm có đối chứng cho RQ3. Đóng góp 3 được viết lại đúng với số liệu | 4.3, 4.7 |
| 6 | Chênh lệch có ý nghĩa thống kê không? | Mọi chênh lệch có khoảng tin cậy 95% bootstrap theo video; các kết luận chính đều có khoảng không chứa 0; các kết quả âm có khoảng chứa 0 và được gọi đúng là "không đo được" | 4.2 |
| 7 | Có chọn thiết lập trên tập test không? | Mọi thiết lập của UEMR chọn trên dev, không lựa chọn nào dựa vào tập kiểm tra, trừ ba ngoại lệ được ghi rõ: hai ở mô hình nền (UniAV; lịch dạy thêm Omni) và một ở UEMR (kích thước Context Fusion, thu nhỏ sau khi thấy bản lớn làm tập kiểm tra giảm). Ngoại lệ thứ ba không làm thổi phồng kết luận về các cách cắt nhiều đoạn, vì ở cả hai kích thước Context Fusion đều không có lợi ích đo được với chúng; riêng +3.42 ở cả video chỉ đo trên bản nhỏ (bản lớn chưa chạy cho cả video), nên đọc kèm ngoại lệ này | 4.2, 4.13 |
| 8 | Một bộ dữ liệu nấu ăn có tổng quát không? | Chưa biết. Đây là hạn chế; lặp lại trên ActivityNet Captions là việc của bản tạp chí | 4.13, 4.14 |
| 9 | Khác gì MeVTR? | Đơn vị là đoạn bước nấu có giờ bắt đầu và kết thúc, không phải khung hình đại diện; chọn bằng bộ tìm bước, không theo độ đa dạng nội dung; dùng cả hình lẫn tiếng; trả về luôn khoảnh khắc. Cách chọn kiểu MeVTR trên cùng mô hình nền thua 10.69 | 4.6 |
| 10 | Âm thanh có đóng góp gì không? | Chưa đo (Exp 4 chưa chạy). Omni có nhận 8 giây âm thanh mỗi đoạn, nhưng chưa có thí nghiệm bỏ âm thanh để so | 4.13 |
| 11 | Chạy lại có ra cùng số không? | Có: 79 số khớp, cách cắt dựng lại trùng hệt, vector mã hoá lại trùng, train lại Context Fusion lệch 0.03 | 4.2, 4.12 (a) |
| 12 | Vì sao không fine-tune chung Omni và UniAV? | Nằm ngoài phạm vi theo thiết kế (`UEMR_FINAL.md` §10 "Không làm"). Giữ hai mô hình cố định là điều kiện để so các cách cắt trên cùng mô hình nền | 4.3 |
| 13 | Ranh giới của người gán nhãn có phải tốt nhất cho tìm kiếm? | Trung bình là tốt hơn nhiều (+12.64 so với UniAV), nhưng không phải luôn luôn: có câu UniAV xếp hạng 1 còn đáp án xếp hạng 65 | 4.11 |
| 14 | Omni có thấy dữ liệu kiểm tra không? | Không thấy khi train: Omni chỉ được dạy thêm trên 1.002 video train. Bằng chứng: lệnh dạy thêm đọc `train_omni_video.trainsplit.jsonl`, log ghi 7.787 câu, batch 8 trên 1 GPU, 2 epoch, tổng 1.948 bước (974 × 2), khớp tên `checkpoint-1948`; nếu thêm video kiểm tra thì số bước sẽ khác. Nhưng checkpoint lấy theo lịch, không chọn trên dev, và theo ghi chép của notebook dạy thêm, lịch này được đặt sau khi xem kết quả trên tập kiểm tra của lần dạy thêm trước, nên đây là một ngoại lệ (mục 4.2) | 4.13 |
| 15 | Vì sao chọn "đoạn khớp nhất" để chấm? | Thử 7 cách gộp (bảng ở mục 4.10) và độ tin cậy của UniAV; chọn trên dev, max tốt nhất cho cắt theo bước UniAV ở cả dev và tập kiểm tra | 4.10 |
| 16 | Báo cáo có rất nhiều phép so; có hiệu chỉnh cho việc so nhiều lần không? | Không hiệu chỉnh chính thức. Các kết luận chính có cận dưới của khoảng tin cậy cách 0 rất xa (RQ1: +14.69; RQ2: +4.97; RQ4 ở mức chồng 0.5: +7.66), nên vẫn đứng nếu dùng ngưỡng chặt hơn. Các kết quả sát ngưỡng được gọi rõ, không dùng để kết luận mạnh: cắt theo bước UniAV − mỗi 14 giây ở kho 394 (−2.28 [−4.33, −0.27]) và kho 498 (−1.68 [−3.68, +0.17], không có ý nghĩa); trung bình các bước − cả video ở kho 498 (+1.29 [−0.07, +2.75]) và với Omni chưa dạy thêm (+1.72 [+0.19, +3.30]; kho 498: +1.85 [+0.53, +3.52]); ranh giới nhiễu ở mức 0.3 (+1.749 [+0.00, +3.45]); Context Fusion so với chuỗi ngữ cảnh xáo trộn (+0.83 [−0.07, +1.68]) | 4.5, 4.6, 4.7, 4.8 |

---

## 4.13 Hạn chế

| Hạn chế | Mức ảnh hưởng | Hướng ảnh hưởng | Bằng chứng |
|---|---|---|---|
| **UniAV (Athena) đã thấy toàn bộ dev khi train và được chọn phiên bản theo tập kiểm tra** | Lớn: mọi số tạm thời | Số của cắt theo bước UniAV có thể cao hơn thực tế; khoảng cách RQ2 là cận trên; thông số chọn trên dev (ngưỡng UniAV, β, thời điểm dừng) có thể lệch nhẹ vì dev "dễ" bất thường | F1 của UniAV: dev 65.37 > train 63.49 > kiểm tra 54.40; bỏ sót dev 22.0% so với kiểm tra 27.8% |
| **Omni: checkpoint cố định theo lịch, không chọn trên dev**; lịch dạy thêm (2 vòng qua dữ liệu, tốc độ học 5e-5) được đặt dựa trên lần dạy thêm trước đó. Lần trước đó chọn phiên bản theo tập kiểm tra và thấy kết quả "đỉnh ở 0.5 vòng rồi tụt dần" (theo ghi chép; báo cáo giữ ngoại lệ này). Dữ liệu dạy thêm chỉ gồm 1.002 video train: 7.787 câu, batch 8, 2 vòng, 1.948 bước, khớp tên `checkpoint-1948` | Trung bình | Mọi số, kể cả số chính thức, có thể cao hơn thực tế một chút; ít khả năng đổi **thứ tự** giữa các cách cắt vì mọi cách dùng cùng Omni. Nhãn "chính thức" nghĩa là không phụ thuộc UniAV, không có nghĩa là không có rò rỉ | `Code/Omni/colab/Omni_training_uemr.ipynb` ô 0 và ô 10 (`BATCH_SIZE = 8`, `GRAD_ACCUM = 1`, `TRAIN` = trainsplit); `Code/Omni/log.md` (lệnh train, `Num examples = 7,787`, `Total optimization steps = 1,948`); `docs/UEMR_TONG_KET.md` §3 |
| **Kích thước Context Fusion** (thu nhỏ từ 135 triệu xuống 3.8 triệu tham số) được quyết định khi bản lớn đã được chấm trên tập kiểm tra (giảm 1.6). Ghi chép không cho biết thứ tự thời gian, nên tính là ngoại lệ (cách tính thận trọng) | Nhỏ | Không làm thổi phồng kết luận về các cách cắt nhiều đoạn: ở cả hai kích thước, Context Fusion không có lợi ích đo được với chúng, nên RQ3 vẫn là "không đo được lợi ích". Riêng +3.42 ở cả video chỉ đo trên bản nhỏ (bản lớn chưa chạy cho cả video), tức bản được chọn sau khi đã thấy tập kiểm tra, nên đọc kèm ngoại lệ này. Tốc độ học riêng của α được đổi cùng lúc nên thuộc cùng ngoại lệ | `notebooks/log/uemr/NHAN_XET_KET_QUA_P6.md` §2.4, §3.2, §4 |
| Một bộ dữ liệu, một miền (nấu ăn) | Chưa biết | Bước lặp lại và bước gần trùng là đặc thù của YouCook2; nên kết quả trên YouCook2 có thể không đại diện cho miền khác (ở miền ít bước lặp hơn, mọi số có thể cao hơn, và khoảng cách giữa các cách cắt có thể khác) | mục 4.9 |
| Exp 4 (âm thanh) chưa làm; Exp 6 thuộc phạm vi tạp chí | – | Không biết âm thanh đóng góp bao nhiêu | mục 4.3 |
| Video `wii9jNiNl9Y` lệch thời gian giữa chuỗi ngữ cảnh UniAV và file video 9.84 giây | Nhỏ (1/394 video) | Có thể làm đoạn UniAV của video này lệch, hạ nhẹ số của cắt theo bước UniAV; video này xuất hiện hai lần trong các ví dụ "UniAV bỏ sót" | `uemr_final_AZ.ipynb` c10; `NHAN_XET_KET_QUA_A4.md` §2.3 |
| Độ trễ CPU phụ thuộc máy; thời gian UniAV mỗi video chưa đo | Nhỏ | Chỉ so tương đối giữa các cách cắt | mục 4.10 |
| Exp 3 chỉ dùng một seed nhiễu mỗi mức | Nhỏ | Vị trí đường cong có thể dịch nhẹ | mục 4.8 |
| Cột (a) chỉ có cả video, trung bình các bước, cắt đều, theo bước UniAV, theo đáp án; cột (c) chỉ có 4 cách cắt | Nhỏ | Không đổi kết luận vì Context Fusion ≈ 0 ở mọi cách cắt nhiều đoạn đã chạy | mục 4.3 |

**Các chỗ lệch so với `UEMR_FINAL.md`**, mỗi chỗ kèm lý do:

| Chỗ | `UEMR_FINAL.md` | Đã dùng | Lý do |
|---|---|---|---|
| Số bước tối đa của UniAV | 16 | 30 | Không video nào có quá 15 bước, nên kết quả như nhau |
| Kích thước Context Fusion | chỉnh ở độ rộng đầy đủ của Omni | độ rộng bên trong còn 256 (không liên quan tới 256 lát của chuỗi ngữ cảnh), chuẩn hoá phần chỉnh | Bản lớn (135 triệu tham số) khớp quá vào train, tức học thuộc dữ liệu train mà không tổng quát sang dữ liệu mới: ở bản lớn, dev đạt đỉnh ngay vòng train (epoch) đầu tiên ở 4 trong 7 lần chạy trong khi mất mát trên train còn giảm tới hơn 20 epoch, và dev không hơn bản không có Context Fusion (cấu hình đầy đủ: 49.42 so với 49.68). Quyết định thu nhỏ được đưa ra khi **đã thấy** tập kiểm tra giảm 1.6 (3 seed), nên báo cáo tính đây là một thiết lập có dùng tới tập kiểm tra (cách tính thận trọng, mục 4.2) |
| Tốc độ học của α | như mọi tham số | lớn hơn 10 lần | đổi cùng lúc với việc thu nhỏ (cùng ngoại lệ); giá trị cụ thể không chọn trên dev hay tập kiểm tra mà đặt theo thiết kế để α tăng được (α nhân với một vector độ dài 1 nên cần bước lớn hơn), kiểm tra trên bài "thuộc bài" 32 video train |
| Nhiệt độ τ_s của hàm điểm khi train | 0.1 | 0.05 | 0.1 làm điểm gần như trung bình các đoạn; chọn trên dev |
| Exp 3 | hai đường, có và không Context Fusion | chỉ đường không có | Context Fusion ≈ 0 ở mọi cách cắt nhiều đoạn |
| Exp 1 cột (c) | mọi dòng | 4 cách cắt | để train Context Fusion cho các cách cắt còn lại cần vector của chúng trên **dữ liệu train** (khoảng 20.000 clip chưa mã hoá; cột (b) chỉ cần dữ liệu kiểm tra); không bắt buộc vì Context Fusion ≈ 0 |
| Cắt mỗi L giây | thêm L = 8 giây | chỉ L = 14 | chưa làm |
| Phạm vi δ | 0 / 5 / 10 | 0 / 5 | δ = 0 ngang δ = 5 |
| Chuỗi ngữ cảnh Z | một tầng, bước 1–2 giây | tầng 0, nội suy về 256 lát (mỗi lát dài video/256 giây) | theo cách Athena tự đánh giá |
| R1 (trung bình các bước) | trung bình vector của R6 (theo bước UniAV) | đúng như vậy; thêm R1_gt (trung bình vector theo đáp án) làm dòng phụ | – |
| Thước đo video → câu (theo MeVTR) | có | chưa làm | thuộc phần tạp chí |

---

## 4.14 Hướng tiếp theo

Theo thứ tự lợi ích dự kiến:

1. **Train lại UniAV đúng quy trình** (chỉ trên phần train, chọn phiên bản trên dev), rồi chạy lại notebook A-Z với đầu ra mới. Đây là
   việc duy nhất có thể đổi kết luận của RQ2 và RQ4, và đổi nhãn "tạm thời" thành "chính thức". Nó cũng cho luôn thời gian chạy UniAV mỗi
   video.
2. **Tăng recall của UniAV hoặc lấp khoảng trống** giữa các đoạn (`uniav_gapfill`, chưa duyệt), vì khoảng 72% khoảng cách tới mức
   trần (9.11 trên 12.64) nằm ở các câu có bước bị bỏ sót (mục 4.9). Kết quả gộp bước UniAV với mỗi 14 giây (29.77) gợi ý lấp khoảng trống
   có ích. Ước lượng thực tế: nếu các bước bị bỏ sót được tìm ra và đạt mức như nhóm không bỏ sót, lợi ích khoảng **6.9** câu trên 100
   (không phải đủ 9.11; ước lượng, không có khoảng tin cậy; mục 4.9). Làm ranh giới chính xác hơn là bước sau, nhắm vào phần còn lại (3.53), phần lớn ở các bước chỉ chồng 0.3–0.5 (2.38).
   Với cắt theo bước UniAV, 148 trên 241 lượt đúng video sai đoạn là đúng bước nhưng ranh giới lệch (mục 4.9).
3. **Exp 4 (âm thanh):** code đã sẵn trong notebook A-Z (`RUN_OPTIONAL = True`), chỉ phía Omni, bằng cách thay âm thanh bằng im lặng. Lưu
   ý: im lặng là đầu vào Omni chưa từng thấy khi train, nên chênh lệch đo được là cận trên của đóng góp âm thanh.
4. **Làm vector bước của UniAV khớp với vector Omni** (hướng song song do thành viên phụ trách UniAV làm): **hướng khác, chưa có kết quả**. Theo
   mục 4.9, hướng này có thể giảm độ nhạy ranh giới của Omni, nhưng không giải quyết bước bị bỏ sót và giới hạn do dữ liệu. Không dùng
   hướng này để claim gì.
5. **ActivityNet Captions và định vị khoảnh khắc đầy đủ** (bản tạp chí).

---

## Phụ lục

### A. Bảng tra cứu

| Kết quả | Log + cell | Đầu ra trên Drive (`MyDrive/Colab Notebooks/code KL/Fushion/uemr/`) | Nhận xét |
|---|---|---|---|
| UniAV: F1, AP, bỏ sót | `notebooks/log/uemr/uemr_final_AZ.ipynb` c10 | `results/056d388e/uniav_stats.csv` | `NHAN_XET_KET_QUA_AZ.md` §1 |
| Các cách cắt: số đoạn, độ dài, độ phủ, tIoU | c12 | `omni_cache/common/partitions/SOURCE.json` | – |
| Exp 1 cột (a), (b), bootstrap | c18 | `results/056d388e/exp1_b.csv`, `exp1_b_bootstrap.csv`, `exp1_a_pt.csv` | `NHAN_XET_KET_QUA_AZ.md` §2 |
| Cột (a): khoảng tin cậy | `notebooks/log/uemr/uemr_ket_qua_bo_sung_B1.md` | `results/056d388e/exp1_a_pt_bootstrap.csv` | mục 4.5 |
| Theo độ dài, bước gần trùng, hồi quy độ phủ | c19 | `results/056d388e/exp1_by_gt_length.csv`, `exp1_by_near_dup.csv`, `rq2_coverage_regression.csv` | – |
| Exp 3 | c21 | `results/056d388e/p5_table.csv`, `p5_summary.json`, `p5_r1_vs_tiou.png` | `NHAN_XET_KET_QUA_P5.md` |
| Context Fusion: cột (c), Exp 2, đối chứng | c25, c26 | `results/056d388e/exp1_c.csv`, `exp2.csv`, `exp2_pairs.csv`, `lvid_only.csv`; `fusion_runs_v2/` | `NHAN_XET_KET_QUA_P6.md` |
| Exp 5 | c28 | `results/056d388e/exp5_*.csv`, `exp5_beta.json` | `NHAN_XET_KET_QUA_EXP5.md` |
| Định tính | c30 | `results/056d388e/qualitative/` | `NHAN_XET_KET_QUA_A4.md` |
| Năng lực Omni | `notebooks/log/omni/new_log_ft.ipynb` c25; `notebooks/log/uemr/P6/uemr_p6_colab_v2.ipynb` c10 | `MyDrive/uemr/omni_cache/ft/results_gt.json` | `docs/UEMR_TONG_KET.md` §5.1 |
| Tổng hợp mọi bảng | c34, c38 | `results/056d388e/so_lieu.md`, `run_manifest.json` | – |

### B. Bảng thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Vector | dãy số tóm tắt nội dung một đoạn video hoặc một câu; hai thứ càng giống thì hai vector càng gần |
| Đa vector | mỗi đoạn của video một vector, thay vì một vector cho cả video |
| Truy hồi video | cho một câu, xếp hạng mọi video trong kho theo mức khớp |
| Kho (gallery) | tập video được xếp hạng: 394 video kiểm tra (chính), 498 video (phụ) |
| Fine-tune (dạy thêm) | train tiếp một mô hình có sẵn trên dữ liệu của mình; Omni `ft` đã dạy thêm, Omni `pt` chưa |
| Oracle | dùng đáp án của người gán nhãn; không dùng được trong thực tế, chỉ để biết mức trần |
| train / dev / val | phần dữ liệu để train / để chọn thiết lập / để kiểm tra (val đóng vai tập test) |
| R@1, R@5, R@10 | trên 100 câu, bao nhiêu câu có video đúng ở hạng 1 / trong top 5 / trong top 10 |
| MedR | hạng trung vị của video đúng; càng nhỏ càng tốt |
| Khoảng tin cậy 95% (CI) | khoảng chứa chênh lệch thật với độ tin cậy 95%; không chứa 0 thì chênh lệch gần như chắc chắn không do may rủi |
| Bootstrap theo video | rút ngẫu nhiên lại các video 1.000 lần để tính khoảng tin cậy; các câu cùng video đi cùng nhau |
| tIoU | hai khoảng thời gian chồng nhau bao nhiêu, từ 0 (không chạm) tới 1 (trùng khít) |
| Bỏ sót | không đoạn nào của UniAV chồng với bước thật từ 0.3 trở lên |
| F1, AP | thước đo mức UniAV tìm đúng các bước (F1: đúng và không thừa; AP: có xét độ tin cậy), cao nhất là 100 |
| Độ phủ | phần trăm thời lượng video nằm trong ít nhất một đoạn |
| MaxSim (max) | điểm video = mức khớp của đoạn khớp nhất |
| LSE, top-k | hai cách gộp điểm khác: "max mềm" và trung bình k đoạn tốt nhất |
| Chuỗi ngữ cảnh (Z) | 256 lát thời gian của video, mỗi lát một vector hình và tiếng do UniAV tạo ra |
| Context Fusion | mô-đun nhỏ chỉnh vector mỗi đoạn dựa trên chuỗi ngữ cảnh và các đoạn khác |
| Epoch | một vòng train qua toàn bộ dữ liệu train |
| Checkpoint (lần lưu) | bản lưu của mô hình tại một thời điểm trong quá trình train |
| Tốc độ học (lr) | mỗi bước train thay đổi tham số nhiều hay ít |
| Hàm mất mát | thước đo "sai" mà mô hình cố giảm khi train; mức video: sai khi xếp video; mức đoạn: sai khi chọn đoạn trong video |
| Lô (batch) | nhóm câu được học cùng một lúc trong một bước train |
| Khớp quá vào train (overfit) | học thuộc dữ liệu train nhưng không làm tốt hơn trên dữ liệu mới |
| Tổng quát hoá | làm tốt trên dữ liệu chưa thấy khi train |
| Attention (sự chú ý), head | trọng số cho biết mỗi đoạn "nhìn" vào lát thời gian nào; head là một nhóm trọng số như vậy, mô-đun có 8 nhóm |
| Nội suy tuyến tính | ước lượng giá trị ở giữa hai điểm đã đo bằng cách nối chúng bằng một đường thẳng |
| τ, τ_s (nhiệt độ) | tham số điều chỉnh độ "gắt" của điểm, khi train (Context Fusion) và khi chấm bằng LSE (mục 4.10); τ nhỏ gần với chọn max |
| RQ1–RQ5 | năm câu hỏi nghiên cứu của `UEMR_FINAL.md` (mục 4.0) |
| Exp 1–5 | năm thí nghiệm của `UEMR_FINAL.md`: Exp 1 bảng chính, Exp 2 tháo từng phần Context Fusion, Exp 3 làm nhiễu ranh giới, Exp 4 âm thanh, Exp 5 cách chấm và chi phí |
| Cột (a) / (b) / (c) | ba cột của bảng chính: (a) Omni chưa dạy thêm, (b) Omni đã dạy thêm, (c) thêm Context Fusion |
| R0–R7 | tên các cách cắt (bảng ở mục 4.1): R0 cả video, R2 mỗi 14 giây, R3 cắt đều theo số bước UniAV, R6 theo bước UniAV, R7 theo đáp án |
| Bước gần trùng | hai bước có mô tả cosine ≥ 0.8 theo một bộ mã hoá câu độc lập (không phải Omni) |
| Cận trên | giá trị mà số thật không vượt quá |
| Kết quả âm | kết quả "không có lợi ích đo được", không phải số âm |
| Regression (trong phần Nguồn) | phép kiểm tra số tính lại có khớp số đã báo trước đó không; khác với "hồi quy" ở mục 4.6 |
| α | mức Context Fusion được phép chỉnh vector; bắt đầu bằng 0 |
| δ | Context Fusion được nhìn ra ngoài đoạn bao nhiêu giây (δ = 0: không nhìn ra ngoài) |
| Seed | số khởi tạo ngẫu nhiên; chạy nhiều seed để chắc kết quả không do may mắn |
| Chỉ ra đúng khoảnh khắc (VCMR R@1) | video đúng ở hạng 1 **và** đoạn khớp nhất chồng ít nhất 0.5 với bước cần tìm |
| Nén 8 bit (int8) | lưu mỗi số của vector bằng 1 byte thay vì 2 byte |
| Chính thức / Tạm thời / Oracle | nhãn trạng thái của kết quả, định nghĩa ở mục 4.1 |
| R0–R7 và các đối chứng | tên các cách cắt, bảng ở mục 4.1 |

### C. Bảng số đầy đủ

**C1. Cột (b), Omni đã dạy thêm** (MedR: hạng trung vị)

| Cách cắt | Kho 394: R@1 / R@5 / R@10 / MedR | Kho 498: R@1 / R@5 / R@10 / MedR |
|---|---|---|
| R0 cả video | 8.32 / 26.93 / 38.75 / 18.5 | 6.93 / 23.10 / 34.82 / 23.0 |
| R1_gt trung bình các bước theo đáp án (oracle) | 10.30 / 32.90 / 48.35 / 11.0 | 8.65 / 28.71 / 42.41 / 14.0 |
| R1 trung bình các bước UniAV | 9.24 / 30.30 / 43.73 / 14.0 | 8.22 / 26.70 / 39.14 / 18.0 |
| R2 mỗi 14 giây | 27.16 / 58.88 / 73.40 / 4.0 | 24.36 / 54.49 / 68.91 / 5.0 |
| R3 cắt đều | 18.02 / 43.66 / 57.36 / 8.0 | 16.20 / 39.21 / 52.41 / 9.0 |
| R4 cắt ngẫu nhiên | 16.17 / 37.76 / 50.89 / 10.0 | 14.39 / 33.89 / 46.01 / 13.0 |
| R5 đa dạng nội dung | 14.19 / 34.39 / 46.63 / 12.0 | 12.41 / 31.16 / 42.97 / 15.0 |
| R6 theo bước UniAV | 24.88 / 52.05 / 64.72 / 5.0 | 22.67 / 48.78 / 60.76 / 6.0 |
| R7 theo đáp án (oracle) | 37.52 / 69.80 / 80.96 / 2.0 | 34.16 / 66.17 / 77.52 / 3.0 |
| cắt đều theo số bước thật (oracle) | 18.91 / 46.47 / 61.29 / 6.0 | 16.60 / 42.48 / 56.60 / 8.0 |
| bước UniAV bị dời chỗ | 17.36 / 39.80 / 50.53 / 10.0 | 15.31 / 36.01 / 46.90 / 13.0 |
| bước thật bị dời chỗ (oracle) | 17.72 / 40.76 / 52.90 / 9.0 | 15.84 / 37.52 / 49.04 / 11.0 |
| gộp R6 và R2 | 29.77 / 62.97 / 76.40 / 3.0 | 27.26 / 57.85 / 72.77 / 4.0 |

**C2. Cột (a), Omni chưa dạy thêm**

| Cách cắt | Kho 394: R@1 / R@5 / R@10 / MedR | Kho 498: R@1 |
|---|---|---|
| R0 cả video | 7.89 / 24.03 / 35.78 / 23.0 | 6.40 |
| R1_gt (oracle) | 10.66 / 32.54 / 47.36 / 12.0 | – |
| R1 trung bình các bước UniAV | 9.60 / 29.41 / 44.22 / 14.0 | 8.25 |
| R3 cắt đều | 15.15 / 38.91 / 53.60 / 9.0 | – |
| R6 theo bước UniAV | 18.94 / 45.12 / 58.18 / 7.0 | 17.26 |
| R7 theo đáp án (oracle) | 25.31 / 54.52 / 68.12 / 4.0 | – |

**C3. Cột (c), có Context Fusion** (kho 394, R@1)

| Cách cắt | Số seed | Không có | Có (trung bình) | Độ lệch chuẩn giữa các seed | Chỉ ra đúng khoảnh khắc (có) |
|---|---|---|---|---|---|
| R0 cả video | 3 | 8.32 | 11.74 | 0.43 | 0.01 |
| R3 cắt đều | 3 | 18.02 | 18.10 | 0.11 | 5.80 |
| R6 theo bước UniAV | 3 | 24.88 | 25.31 | 0.32 | 17.37 |
| R7 theo đáp án (oracle) | 1 | 37.52 | 37.72 | – | 35.87 |

### D. Tham số đã chốt (trích `configs/final_protocol.json`)

| Nhóm | Tham số | Giá trị |
|---|---|---|
| UniAV | ngưỡng θ | 0.4, chọn trên dev (số bước dự đoán trung bình gần số bước thật nhất) |
| | số bước tối đa / độ dài tối thiểu / gộp đoạn chồng | 30 / 2 giây / tIoU > 0.7 |
| | chuỗi ngữ cảnh | tầng 0, 256 lát, 1.024 chiều, lưu fp16 |
| Cách cắt | mỗi L giây | L = 14 |
| | độ dài tối thiểu (cắt ngẫu nhiên) | 2 giây |
| | đa dạng nội dung | K-Medoids trên chuỗi ngữ cảnh, cửa sổ 2 giây quanh điểm đại diện |
| Omni | `ft` / `pt` | `checkpoint-1948` (lịch train cố định) / `YunzeLiu/OmniRetriever-7B` |
| | mỗi đoạn | 8 khung hình, 50.176 điểm ảnh, 8 giây âm thanh |
| Chấm điểm | cách gộp / β | max / 0 |
| Context Fusion (D3) | kích thước / đầu ra | 256; c = chuẩn hoá(e + α · chuẩn hoá(W·u)), α bắt đầu bằng 0 |
| | attention | 8 head, dropout 0.1, δ = 5 giây |
| | hàm mất mát | τ = 0.05, τ_s = 0.05, λ = 0.5, có vector cả video |
| | tối ưu | lr 1e-4 (α: 1e-3), wd 0.01, warmup 5%, cosine, lô 128 (một câu mỗi video), tối đa 30 epoch, dừng sau 5 epoch không tăng trên dev |
| | seed | 0, 1, 2 |
| Exp 3 | mức tIoU / sai lệch cho phép / seed | 0.9, 0.7, 0.5, 0.3 / 0.02 / 0 |
| Đánh giá | bootstrap | 1.000 lần, seed 0, theo video, 95% |
| | hạng | 1 + số video có điểm lớn hơn video đúng |

### E. Công thức

- **Điểm video (MaxSim):** S(q, V) = max_i cos(q, e_i), với q là vector câu tìm kiếm và e_1…e_K là vector các đoạn của video V. Đoạn đạt
  max cho luôn khoảnh khắc [t_s, t_e].
- **LSE:** S = τ · log( (1/K) · Σ_i exp(cos(q, e_i) / τ) ). τ nhỏ thì gần max; τ lớn thì gần trung bình.
- **Top-k mean:** trung bình k giá trị cos lớn nhất.
- **Prior độ tin cậy:** s_i = cos(q, e_i) + β · log conf_i.
- **InfoNCE** (họ hàm mất mát khi dạy thêm Omni và train Context Fusion): với câu q, ứng viên đúng x⁺ và các ứng viên x trong cùng lô,
  mất mát là −log( exp(s(q, x⁺)/τ) / Σ_x exp(s(q, x)/τ) ). Khi dạy thêm Omni, ứng viên là đoạn clip và s là cos; khi train Context
  Fusion, ứng viên là video và s là LSE với nhiệt độ τ_s (mức video), cộng thêm một số hạng mức đoạn có trọng số λ.
- **tIoU** của hai khoảng [a, b] và [c, d]: độ dài phần giao chia độ dài phần hợp.
- **Hồi quy độ phủ (mục 4.6):** với mỗi video v, y_v = chênh lệch R@1 của hai cách cắt trên các câu của v, x_v = chênh lệch độ phủ; khớp
  y = a + b · x bằng bình phương tối thiểu có trọng số theo số câu của video. Hệ số a là chênh lệch khi độ phủ bằng nhau; khoảng tin cậy
  lấy bằng bootstrap theo video (`omni_retrieval/evaluate.py` `coverage_regression`).
- **Bootstrap theo video:** rút lại 394 video có lặp, tính chênh lệch R@1 trên mọi câu của các video được rút; lặp 1.000 lần; khoảng tin
  cậy là phân vị 2.5% và 97.5%.
