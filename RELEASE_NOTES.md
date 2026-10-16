# PhoBERT-v2 + CRF + Hierarchical SupCon · 16/10/2026

Checkpoint tốt nhất từ lần chạy Vast.ai RTX 3090 24 GB, seed 42.

- Encoder: `vinai/phobert-base-v2`; CRF + Hierarchical SupCon.
- Best checkpoint: epoch 7 (zero-based), dev BIO token macro-F1 0.987413.
- Test scorer CoNLL trong notebook: precision 0.9938, recall 0.9966, entity micro-F1 0.9952 trên 10,278 câu.
- Model: 538,671,007 byte. SHA-256: `449743cbb5b89ab8860d46875577a170600208a00120150772b799cfbd351bab`.
- `train.log`, `runner.log`, `SHA256SUMS.txt` đính kèm để đối chiếu.

Notebook gốc giữ nguyên trong repo. Huấn luyện và test đã hoàn tất; cell báo cáo `nervaluate` phía sau dừng vì `TypeError`. Các chỉ số ở đây không phải bảng Strict/Partial/Entity Type/Exact của cell bị lỗi. Chưa xác minh chạy trên Kaggle.

Xem README và `reports/experiment.md` để biết nguồn dữ liệu, cấu hình và giới hạn của phép đánh giá.
