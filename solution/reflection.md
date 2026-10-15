# Reflection — Data Siege (Day 27, Track 2)

**Học viên:** Phùng Thanh An — 2A202603006 (K4)

## 1. Loại lỗi khó phát hiện nhất

Khó nhất là các lỗi **biên độ tinh vi (subtle)** đứng ngay trên bao lọc sạch, chia làm hai dạng:

- **Đa chiều (multi-signal) trong `data_batch`**: `volume_spike` ở dạng "nhẹ" làm `row_count`,
  `mean_amount` và `staleness_min` cùng lệch, nhưng *mỗi chiều vẫn nằm trong biên 3σ công bố*.
  Kiểm tra từng chiều riêng lẻ không thấy gì, nên tôi phải chuẩn hóa từng chiều về thang
  "biên 3σ = 1.0" rồi lấy chuẩn Euclid (`norm2`). Trên 45 mẫu sạch của practice + public,
  `norm2` tối đa chỉ 1.138, còn ca lỗi đa chiều tinh vi thấp nhất là 1.546 → đặt ngưỡng 1.20
  (thay vì 1.30 như gợi ý) vì 1 lỗi bỏ sót đắt gấp ~4.5 lần 1 báo động giả.
- **Trôi dạt/ô nhiễm sát ngưỡng ở `ai_infra`**: `centroid_shift` (sạch ≤ 0.0388, lỗi ≥ 0.0400)
  và `avg_doc_age_days` (sạch ≤ 41 ngày, lỗi ≥ 48). Khe hở giữa sạch và lỗi ở đây chỉ ~3–12%,
  trong khi **biên 3σ công bố lại nằm sai chỗ**: `feature_mean_shift_sigma_max = 0.4095` thấp
  hơn cả dữ liệu sạch (tới 0.472 → dùng nguyên biên là False Positive ngay), còn
  `embedding_centroid_shift_max = 0.0435` và `corpus_avg_doc_age_days_max = 49.8` lại cao hơn
  ca lỗi tinh vi thấp nhất (0.040 và 48.3 → dùng nguyên biên là bỏ sót). Vì vậy với 3 tín hiệu
  này tôi **không dùng biên công bố** mà đặt ngưỡng vào giữa khe hở (0.65 / 0.0395 / 44.5 ngày).

Bài học lớn nhất: trên phase Private, phần lớn số lỗi bị bỏ sót thuộc nhóm **nằm sát phân phối
chuẩn** (ngoài khoảng "thấy được" bằng mọi ngưỡng còn an toàn). Khi tôi siết cả 10 ngưỡng vào
sát bao lọc sạch (row_count 556, null 0.0093, staleness 7.5, delay 10.0, duration 4880…), kết quả
chỉ thêm **1 TP nhưng mất 2 FP** — đúng như cảnh báo của bài lab: lỗi sát phân phối chuẩn thì
không thể bắt thêm mà không bán rẻ FPR. Việc siết ngưỡng này vẫn có lợi ròng (+0.5 điểm) và giữ
FPR ở mức 1.4%, nên tôi giữ lại.

## 2. Quyết định cân bằng chi phí gọi tool / TPR

Phễu lọc của tôi: `payload` (miễn phí) → `ctx.baseline` + ngưỡng hiệu chuẩn (miễn phí) →
`ctx.state` (miễn phí) → **đúng 1 lệnh gọi tool cho mỗi sự kiện**, không bao giờ gọi trùng lặp
hoặc gọi thêm để "xác nhận". Vì mỗi tool đã trả về nhiều chiều trong 1 lần gọi
(`batch_profile` trả 5 chỉ số, `embedding_drift` trả 2 chỉ số), chi phí đạt **mức sàn lý thuyết**:

| Phase | Sự kiện | Chi phí | Ngân sách | Overage |
|---|---|---|---|---|
| practice | 120 | 180.0 | 220 | 0 → **50.0 điểm** (TPR 1.0, FPR 0) |
| public | 160 | 240.0 | 220 | 0.0909 → **48.18 điểm** (TPR 1.0, FPR 0) |
| private | 200 | 300.0 | 320 | 0 → **32.92 điểm** (TPR 0.667, FPR 0.0137) |

Ở public tôi **cố tình chấp nhận vượt ngân sách 20 credit** (−1.82 điểm): bỏ 1 lệnh gọi để tiết
kiệm 1–2 credit nghĩa là tự nguyện mù trước 1 sự kiện, tức đánh đổi tối đa 0.93 điểm lấy 0.006
điểm tiết kiệm — lỗ ròng theo mọi hướng. Trần phạt overage là 20 điểm nhưng thực tế chỉ mất 1.82.
Đây là đánh đổi có lợi nhất trong cả ba mục tiêu của bài toán.

## 3. Nếu có thêm thời gian — cải tiến thống kê trong `ctx.state`

Detector hiện tại **không nhớ gì giữa các sự kiện**: mọi phán quyết chỉ dựa trên ngưỡng tĩnh đã
hiệu chuẩn trước. Ba hướng cải tiến tôi sẽ làm tiếp, đều dùng `ctx.state` (miễn phí):

1. **Chuẩn hóa theo chính run đang chạy (robust running statistics)**: giữ median/MAD trượt của
   từng tín hiệu trên các sự kiện *chưa từng bị cảnh báo* để phát hiện phase/seed nào có phương
   sai rộng hơn dữ liệu lịch sử — ngưỡng tĩnh hiện tại sẽ vỡ nếu phân phối sạch của phase đó khác đi.
2. **Bộ lọc tuần tự CUSUM/EWMA**: các lỗi dạng "trôi chậm" (corpus mục dần, feature skew tăng dần
   qua từng batch) nằm dưới ngưỡng từng sự kiện nhưng tích lũy lại rất rõ. Một thống kê tích lũy
   trong `ctx.state` sẽ bắt được nhóm này mà không hạ ngưỡng đơn lẻ (nhờ đó không tăng FPR).
3. **Ngưỡng theo hàm tổn thất thay vì hằng số**: với `ctx.state` đếm số sự kiện/số cảnh báo đã
   phát, tôi có thể chạy luật quyết định Bayes `alert ⇔ P(lỗi | bằng chứng) ≥ 0.18`
   (từ tỷ lệ đánh đổi 0.926 điểm/TP so với 0.205 điểm/FP), tức hiệu chỉnh ngưỡng theo đúng
   lịch trình sự kiện còn lại thay vì cố định như hiện nay.
