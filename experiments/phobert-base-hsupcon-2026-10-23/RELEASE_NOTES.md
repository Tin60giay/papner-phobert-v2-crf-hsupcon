# PhoBERT base + CRF + Hierarchical SupCon · 23/10/2026

Backbone `vinai/phobert-base`, seed 42, Vast RTX 3090. Train 17/20 epoch, early stopping sau epoch 16; best epoch 12, dev BIO token macro-F1 98,8903%.

Test dedup 5.162 câu: CoNLL precision 98,21%, recall 98,88%, micro-F1 98,55%. Đây là scorer trong notebook; bảng bốn tiêu chí nervaluate chưa tạo do lỗi API ở cell báo cáo cuối. Train và checkpoint đã hoàn tất; toàn notebook exit code 1. Notebook không bị sửa khi đóng gói.

[Code, log, báo cáo và cấu hình](https://github.com/Tin60giay/papner-phobert-v2-crf-hsupcon/tree/main/experiments/phobert-base-hsupcon-2026-10-23). Checkpoint: 538.671.071 byte, SHA-256 `6f54f28f4c185541ba1e458b04354a2b7d4f3a80769f9072d76750be34df7892`.
