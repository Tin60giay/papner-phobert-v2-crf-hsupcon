# Báo cáo lần chạy 16/10/2026

## Phạm vi và nguồn số liệu

Một lần chạy PhoBERT-v2 + CRF + Hierarchical SupCon, seed 42, RTX 3090 24 GB. Dữ liệu lấy trực tiếp từ [train.log](../logs/train.log); cấu hình tại [training-config.json](training-config.json), bảng epoch tại [history.csv](history.csv), số liệu máy đọc được tại [metrics.json](metrics.json).

## Lựa chọn checkpoint

Chạy epoch 0–11 rồi early stopping. Checkpoint tốt nhất được lưu tại epoch 7, theo **BIO token macro-F1 trên dev = 0.987413**. Đây là macro-F1 của `sklearn.classification_report` trên các nhãn BIO, không phải entity strict micro-F1.

![Huấn luyện theo epoch](figures/training-curves.png)

## Test checkpoint tốt nhất

10,278 câu, 243,616 token. Bộ chấm chunk CoNLL trong notebook ghi precision **0.9938**, recall **0.9966**, micro-F1 **0.9952**. Code `validate()` gộp các chuỗi token trước khi gọi bộ chấm; repo này giữ nguyên hành vi và số liệu đó.

| Loại | Precision (%) | Recall (%) | F1 (%) | Số entity gold |
| --- | ---: | ---: | ---: | ---: |
| CQ | 99.74 | 99.88 | 99.81 | 7,321 |
| NG | 100.00 | 100.00 | 100.00 | 4,021 |
| SL | 97.12 | 98.66 | 97.88 | 1,666 |
| VBPL | 97.51 | 99.15 | 98.32 | 842 |
| ĐT | 99.40 | 99.54 | 99.47 | 7,189 |

BIO token accuracy: **0.9992**. Token macro-F1 trong classification report: **0.9959**. Không trộn hai chỉ số này với entity micro-F1.

## Trạng thái cell báo cáo paper

Sau test, cell dùng `nervaluate==1.2.1` dừng khi truy cập `overall[scenario]`: `TypeError: string indices must be integers, not 'str'`. Xem đầy đủ [runner.log](../logs/runner.log). Bảng Strict/Partial/Entity Type/Exact chưa được tạo; không có các chỉ số đó trong báo cáo này. Notebook được lưu nguyên bản, không sửa lỗi API trong quá trình đóng gói.

## Giới hạn tái lập

- Train/dev dùng revision cố định `e4af303d22f878e5827c94861f50f747a6b3ed45` của `levandai1704/PAP_NER_dedup_70_15_15`.
- Test lấy từ `Leekien0108/PAP_NER`. Notebook lấy SHA tại runtime; log đã xuất không lưu SHA đó. Chưa kiểm chứng overlap giữa hai nguồn dữ liệu.
- Kết quả là một seed và dùng scorer đi kèm notebook; chưa phải phép so sánh kiểm soát với paper hoặc các artifacts trong repo tham khảo.
- Chưa xác minh chạy trên Kaggle. Runtime PyTorch và thư viện không pin toàn bộ phiên bản; danh sách cài đặt nguyên bản ở [requirements.txt](../requirements.txt).
- Cell inference mẫu nằm sau cell bị lỗi nên chưa được chạy trong lần thực thi này.
