# Checkpoint PhoBERT-v2 + CRF + Hierarchical SupCon

[Tải best_model.pt](https://github.com/Tin60giay/papner-phobert-v2-crf-hsupcon/releases/download/v1.0.0/best_model.pt) · [Trang release](https://github.com/Tin60giay/papner-phobert-v2-crf-hsupcon/releases/tag/v1.0.0)

- Kích thước: **538,671,007 byte** (513.72 MiB).
- SHA-256: `449743cbb5b89ab8860d46875577a170600208a00120150772b799cfbd351bab`.
- Checkpoint tốt nhất theo dev BIO token macro-F1: epoch **7** (đếm từ 0).
- Lưu bằng `torch.save`, gồm `model` (state dict), `classes` và `args`.
- Dùng class `PhoBertV2Crf` trong notebook để nạp; đây không phải thư mục `save_pretrained`.

Model được đính kèm Release để clone repo không phải tải trọng số lớn. Không cần Git LFS.

```bash
gh release download v1.0.0 --repo Tin60giay/papner-phobert-v2-crf-hsupcon --pattern best_model.pt --dir artifacts
```

Hash và kích thước từng file nằm trong [manifest.json](manifest.json). File model tải từ Vast đã được kiểm tra ZIP CRC trước khi phát hành.
