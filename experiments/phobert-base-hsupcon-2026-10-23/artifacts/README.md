# Checkpoint PhoBERT · 23/10/2026

[Tải best_model.pt](https://github.com/Tin60giay/papner-phobert-v2-crf-hsupcon/releases/download/phobert-base-2026-10-23/best_model.pt) · [Release](https://github.com/Tin60giay/papner-phobert-v2-crf-hsupcon/releases/tag/phobert-base-2026-10-23)

- Backbone: `vinai/phobert-base`; CRF + Hierarchical SupCon.
- Best epoch: 12 (đếm từ 0), chọn bằng dev BIO token macro-F1 0.988903.
- Kích thước: **538.671.071 byte**.
- SHA-256: `6f54f28f4c185541ba1e458b04354a2b7d4f3a80769f9072d76750be34df7892`.
- Checkpoint `torch.save` gồm `model` (state dict), `classes`, `args`; nạp bằng class và hàm test trong notebook. Không phải thư mục `save_pretrained`.

```bash
gh release download phobert-base-2026-10-23 --repo Tin60giay/papner-phobert-v2-crf-hsupcon --pattern best_model.pt --dir artifacts/phobert-base-20261023
```

Model nằm trong Release, tách khỏi lịch sử Git. [manifest.json](manifest.json) và [SHA256SUMS.txt](SHA256SUMS.txt) dùng để đối chiếu file.
