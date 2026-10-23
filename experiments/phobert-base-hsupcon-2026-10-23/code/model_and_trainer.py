# Export of the model/trainer definitions cell from the actual run notebook.
# Workflow and dependency installation remain in the notebook.
# Mã Python được lấy trực tiếp từ PhoBERT-Legal-NER/vphoberttagger.
# Các class loss và hàm train được giữ nguyên; train/test chọn PhoBertV2Crf.
# CRF/SupCon/forward giữ nguyên theo PhoBertCrf của baseline.
import argparse
from argparse import ArgumentParser
import datetime
import itertools
import logging
import math
import os
import random
import re
import sys
import time
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from prettytable import PrettyTable
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchcrf import CRF
from tqdm import tqdm
from vncorenlp import VnCoreNLP
from transformers import (AutoConfig, AutoTokenizer, RobertaForSequenceClassification,
                          RobertaForTokenClassification, get_linear_schedule_with_warmup)

transformers.logging.set_verbosity_error()
work_root = Path(os.environ.get("NER_WORK_ROOT", "/workspace/phobert_base_hsupcon_20261023" if Path("/workspace").is_dir() else str(Path.cwd() / "phobert_base_hsupcon_20261023"))).resolve()
project_dir = work_root
project_dir.mkdir(parents=True, exist_ok=True)
os.chdir(project_dir)
get_ipython().run_line_magic("cd", str(project_dir))
print("Thư mục kết quả:", project_dir)



# ========================= Helper functions =========================
def safe_torch_load(f, map_location=None, **kwargs):
    """Safely loads torch files supporting PyTorch 2.6+ where weights_only defaults to True."""
    try:
        return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(f, map_location=map_location, **kwargs)

def set_ramdom_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        try:
            if not os.path.isdir('./logs'):
                os.makedirs('./logs', exist_ok=True)
            log_path = os.path.join('./logs', log_file)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_file_level)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Notice: Could not write log file ({e}), logging to console only.")
    return logger

def get_total_model_parameters(model):
    total_params, trainable_params = 0, 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            trainable_params += params
        total_params += params
    return total_params,

def normalize_text(txt: str) -> str:
    # Remove special character
    txt = re.sub("\xad|\u200b|\ufeff", "", txt)
    # Normalize vietnamese accents
    txt = re.sub(r"òa", "oà", txt)
    txt = re.sub(r"óa", "oá", txt)
    txt = re.sub(r"ỏa", "oả", txt)
    txt = re.sub(r"õa", "oã", txt)
    txt = re.sub(r"ọa", "oạ", txt)
    txt = re.sub(r"òe", "oè", txt)
    txt = re.sub(r"óe", "oé", txt)
    txt = re.sub(r"ỏe", "oẻ", txt)
    txt = re.sub(r"õe", "oẽ", txt)
    txt = re.sub(r"ọe", "oẹ", txt)
    txt = re.sub(r"ùy", "uỳ", txt)
    txt = re.sub(r"úy", "uý", txt)
    txt = re.sub(r"ủy", "uỷ", txt)
    txt = re.sub(r"ũy", "uỹ", txt)
    txt = re.sub(r"ụy", "uỵ", txt)
    txt = re.sub(r"Ủy", "Uỷ", txt)

    txt = re.sub(r'"', '”', txt)

    # Remove multi-space
    txt = re.sub(" +", " ", txt)
    return txt.strip()

def _get_tags(sents):
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}
        for i, tag in enumerate(iob_tags):
            if tag == 'O' and curr_tag['type']:
                tags.append(tuple(curr_tag.values()))
                curr_tag = {'type': None, 'start_idx': None,
                            'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B'):
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I'):
                curr_tag['end_idx'] = i
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))
    tags = set(tags)
    return tags

def plot_confusion_matrix(y_true, y_pred, classes, labels,
                          normalize=False,
                          title=None,
                          output_dir='./',
                          cmap=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # Only use the labels that appear in the data

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        print("Normalized confusion matrix")
    else:
         print('Confusion matrix, without normalization')

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(fig)
    return ax


# ========================= NerOutput =========================
class NerOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = torch.FloatTensor([0.0])
    tags: Optional[List[int]] = []
    loss_crf: Optional[torch.FloatTensor] = None
    loss_supcon: Optional[torch.FloatTensor] = None

    def __init__(self, *args, **kwargs):
        super(NerOutput, self).__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())


# ========================= CRF + Hierarchical SupCon: nguyên bản =========================
def _disable_amp(device_type: str = 'cuda'):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type, enabled=False)
    return torch.cuda.amp.autocast(enabled=False)

class ProjectionHead(nn.Module):
    """
    Projection Head for Supervised Contrastive Learning (CUA-MNER design)
    Projects 768-dim contextual representations to a 128-dim normalized contrastive space.
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class TokenSupConLoss(nn.Module):
    """
    Token-level Supervised Contrastive Loss for NER entities.
    Supports both Exact BIO Matching and Hierarchical Entity-Type Matching.
    """
    def __init__(self, temperature: float = 0.07, mode: str = 'exact', alpha: float = 0.3):
        super(TokenSupConLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.alpha = alpha

    def forward(self, features, labels):
        # features: [N_entity, 128]
        # labels:   [N_entity]
        if features.size(0) < 2:
            return features.new_tensor(0.0)

        # 1. Ep kieu sang float32 de dam bao on dinh so hoc tuyet doi trong moi truong AMP
        features = features.float()
        features = F.normalize(features, p=2, dim=1)

        # 2. Cosine similarity scaled by temperature
        similarity = torch.matmul(features, features.T) / self.temperature
        N = features.size(0)

        # 3. Self-mask
        self_mask = torch.eye(N, device=features.device, dtype=torch.bool)

        # 4. Weight mask (Exact BIO vs Hierarchical Matching)
        labels_col = labels.contiguous().view(-1, 1)
        same_bio = torch.eq(labels_col, labels_col.T)

        if self.mode == 'exact' or self.alpha == 0.0:
            weight_mask = (same_bio & (~self_mask)).float()
        else:
            # Map BIO label ID sang Entity Type ID:
            # ID 1,2 -> 1 (CQ); 3,4 -> 2 (ĐT); 5,6 -> 3 (VBPL); 7,8 -> 4 (NG); 9,10 -> 5 (SL)
            entity_types = (labels_col + 1) // 2
            same_type = torch.eq(entity_types, entity_types.T)
            strong_pos = same_bio.float()
            weak_pos = (same_type & (~same_bio)).float() * self.alpha
            weight_mask = (strong_pos + weak_pos) * (~self_mask).float()

        # 5. Denominator mask (all candidates except self)
        # Sử dụng -1e9 thay cho -inf để tránh hoàn toàn phép toán 0.0 * (-inf) = NaN
        similarity = similarity.masked_fill(self_mask, -1e9)

        # 6. Numerical stability (subtract max)
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # 7. Log probability over denominator
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # 8. Gán đường chéo bằng 0.0 để triệt tiêu hoàn toàn nguy cơ NaN khi nhân ma trận
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        # 9. Filter anchors that have positive weight > 0
        weight_count = weight_mask.sum(dim=1)
        valid_anchor = weight_count > 0

        if not valid_anchor.any():
            return features.new_tensor(0.0)

        # 10. Mean positive log-likelihood per valid anchor (Weighted Expectation)
        mean_log_prob_pos = (weight_mask * log_prob).sum(dim=1) / weight_count.clamp_min(1e-12)

        # 11. Scalar SupCon loss
        loss = -mean_log_prob_pos[valid_anchor].mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return features.new_tensor(0.0)
        return loss

class PhoBertCrf(RobertaForTokenClassification):
    """
    PhoBERT + CRF (+ optional SupCon) Architecture for Vietnamese Named Entity Recognition (NER)
    
    Pipeline:
      1. Input tokens (word-segmented) -> PhoBERT Tokenizer (BPE subwords)
      2. PhoBERT Encoder -> Contextual representations H in R^(T x 768) (aligned via valid_ids)
      3. Dropout + Linear Classifier -> Emission scores (768 -> N_tags)
      4. CRF Layer -> Transition scores & Negative Log-Likelihood Loss
      5. (Optional) Projection Head + SupCon Loss -> Entity representation learning
      6. Viterbi Decoding -> Optimal BIO entity tag sequence
    """
    def __init__(self, config, use_supcon=None, lambda_supcon=None, supcon_temp=None, supcon_mode=None, supcon_alpha=None, o_label_id=None):
        super(PhoBertCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.crf = CRF(config.num_labels, batch_first=True)

        self.use_supcon = use_supcon if use_supcon is not None else getattr(config, 'use_supcon', False)
        self.lambda_supcon = lambda_supcon if lambda_supcon is not None else getattr(config, 'lambda_supcon', 0.3)
        self.supcon_temp = supcon_temp if supcon_temp is not None else getattr(config, 'supcon_temp', 0.07)
        self.supcon_mode = supcon_mode if supcon_mode is not None else getattr(config, 'supcon_mode', 'exact')
        self.supcon_alpha = supcon_alpha if supcon_alpha is not None else getattr(config, 'supcon_alpha', 0.3)
        self.o_label_id = o_label_id if o_label_id is not None else getattr(config, 'o_label_id', 0)

        if self.use_supcon:
            self.projection_head = ProjectionHead(input_dim=config.hidden_size, hidden_dim=256, output_dim=128)
            self.supcon_loss = TokenSupConLoss(temperature=self.supcon_temp, mode=self.supcon_mode, alpha=self.supcon_alpha)
        else:
            self.projection_head = None
            self.supcon_loss = None

        self.init_weights()
        # Đảm bảo CRF transition parameters được khởi tạo chuẩn xác trong [-0.1, 0.1]
        self.crf.reset_parameters()

    def _init_weights(self, module):
        super(PhoBertCrf, self)._init_weights(module)
        if isinstance(module, CRF):
            module.reset_parameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        # 1. PhoBERT contextual representations
        seq_outputs = self.roberta(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   head_mask=None)[0]

        # 2. Subword alignment to word-level representations H in R^(T x 768)
        batch_size, max_len, feat_dim = seq_outputs.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        word_representations = seq_outputs[range_vector, valid_ids]
        
        # 3. Dropout + Linear emission layer: 768 -> N_tags
        seq_outputs_dropped = self.dropout(word_representations)
        logits = self.classifier(seq_outputs_dropped)  # Emission scores

        # 4. Viterbi decoding via CRF (chỉ giải mã khi eval/predict, bỏ qua lúc train để tăng tốc tối đa)
        # Ép kiểu logits sang FP32 và tắt autocast để chống tràn số cực đại FP16 trong CRF
        logits_fp32 = logits.float()
        dev_type = 'cuda' if logits.is_cuda else 'cpu'
        if labels is None or not self.training:
            with _disable_amp(dev_type):
                seq_tags = self.crf.decode(logits_fp32, mask=label_masks != 0)
        else:
            seq_tags = []

        # 5. Training loss
        if labels is not None:
            with _disable_amp(dev_type):
                log_likelihood = self.crf(logits_fp32, labels, mask=label_masks.bool(), reduction='mean')
                loss_crf = -1.0 * log_likelihood

            if self.use_supcon and self.projection_head is not None:
                # Representation before dropout passed into projection head
                projected = self.projection_head(word_representations)
                valid_entity_mask = (label_masks != 0) & (labels != self.o_label_id)
                entity_feats = projected[valid_entity_mask]
                entity_tags = labels[valid_entity_mask]
                with _disable_amp(dev_type):
                    loss_supcon = self.supcon_loss(entity_feats.float(), entity_tags)
                    if torch.isnan(loss_supcon) or torch.isinf(loss_supcon):
                        loss_supcon = loss_crf.new_tensor(0.0)
                loss_total = loss_crf + self.lambda_supcon * loss_supcon
                return NerOutput(loss=loss_total, tags=seq_tags, loss_crf=loss_crf, loss_supcon=loss_supcon)
            else:
                return NerOutput(loss=loss_crf, tags=seq_tags)
        else:
            return NerOutput(tags=seq_tags)


# ========================= Feature processing =========================
class NerFeatures(object):
    def __init__(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None, labels=None, label_masks=None, **kwargs):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.token_type_ids = (torch.as_tensor(token_type_ids, dtype=torch.long)
                               if token_type_ids is not None
                               else torch.zeros_like(self.input_ids))
        self.attention_mask = (torch.as_tensor(attention_mask, dtype=torch.long)
                               if attention_mask is not None
                               else torch.ones_like(self.input_ids))
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long) if valid_ids is not None else None
        self.labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long) if label_masks is not None else None

def convert_word_segment_examples_features(data_path: Union[str, os.PathLike],
                                          tokenizer,
                                          label2id,
                                          header_names: List[str],
                                          max_seq_len: int = 256,
                                          use_crf: bool = True) -> List[NerFeatures]:
    """
    Process Vietnamese word-segmented dataset into PhoBERT features with subword alignment.
    Supports whitespace-delimited (spaces or tabs) CoNLL format files.
    """
    features = []
    tokens = []
    tag_ids = []
    
    data = pd.read_csv(data_path,
                       sep=r'\s+',
                       encoding='utf-8',
                       skip_blank_lines=False,
                       names=header_names)
    data.fillna(method="ffill")
    
    for row_idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Load dataset {data_path}..."):
        if row.notna().token:
            tokens.append(row.token.strip().replace(' ', '_'))
            tag_ids.append(label2id.index(row.ner.strip()))
            if not row_idx == len(data) - 1:
                continue
                
        seq_len = len(tokens)
        if seq_len == 0:
            continue
            
        sentence = ' '.join(tokens)
        encoding = tokenizer(sentence,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len)
        subwords = tokenizer.tokenize(sentence)
        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        label_marks = np.zeros(len(encoding.input_ids), dtype=int)
        valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
        i = 1
        for idx, subword in enumerate(subwords[:max_seq_len - 2]):
            if idx != 0 and subwords[idx - 1].endswith("@@"):
                continue
            if use_crf:
                valid_ids[i - 1] = idx + 1
            else:
                valid_ids[idx + 1] = 1
            valid_labels[idx + 1] = tag_ids[i - 1]
            i += 1
            
        if max_seq_len >= seq_len:
            label_padding_size = (max_seq_len - seq_len)
            label_marks[:seq_len] = [1] * seq_len
            tag_ids.extend([0] * label_padding_size)
        else:
            tag_ids = tag_ids[:max_seq_len]
            label_marks[:-2] = [1] * (max_seq_len - 2)
            tag_ids[-2:] = [0] * 2
            
        if use_crf and label_marks[0] == 0:
            raise ValueError(f"{sentence} - {tag_ids} has label_mark == 0 at index 0!")

        items = {key: val for key, val in encoding.items()}
        if 'token_type_ids' not in items:
            items['token_type_ids'] = [0] * max_seq_len
        items['labels'] = tag_ids if use_crf else valid_labels
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_marks if use_crf else valid_ids
        features.append(NerFeatures(**items))

        for k, v in items.items():
            assert len(v) == max_seq_len, f"Expected length of {k} is {max_seq_len} but got {len(v)}"

        tokens = []
        tag_ids = []
        
    return features


# ========================= Labels and mappings =========================
LOGGER = init_logger(datetime.datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))

LABEL2ID_PAP_NER = [
    'O',
    'B-CQ', 'I-CQ',
    'B-ĐT', 'I-ĐT',
    'B-VBPL', 'I-VBPL',
    'B-NG', 'I-NG',
    'B-SL', 'I-SL'
]

PROCESSOR_MAPPING = {
    'vinai/phobert-base': convert_word_segment_examples_features,
    'vinai/phobert-base-v2': convert_word_segment_examples_features,
    'vinai/phobert-large': convert_word_segment_examples_features,
    'trungbb8/distilphobert': convert_word_segment_examples_features,
}

MODEL_MAPPING = {
    'vinai/phobert-base': {
        'crf': PhoBertCrf,
    },
    'vinai/phobert-base-v2': {
        'crf': PhoBertCrf,
    },
    'vinai/phobert-large': {
        'crf': PhoBertCrf,
    },
    'trungbb8/distilphobert': {
        'crf': PhoBertCrf,
    }
}

LABEL_MAPPING = {
    'pap_ner': {
        'label2id': LABEL2ID_PAP_NER,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_PAP_NER)},
        'header': ['token', 'pos', 'chunk', 'ner', 'tmp']
    }
}


# ========================= Dataset =========================
def _safe_load_torch(f, **kwargs):
    try:
        return torch.load(f, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(f, **kwargs)

class NerDataset(Dataset):
    def __init__(self, features: List[NerFeatures], device: str = 'cpu'):
        self.examples = features
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val for key, val in self.examples[index].__dict__.items()}

def build_dataset(data_dir: Union[str, os.PathLike],
                  tokenizer,
                  label2id: List[str],
                  header: List[str],
                  dtype: str = 'train',
                  max_seq_len: int = 256,
                  device: str = 'cpu',
                  use_crf: bool = True,
                  overwrite_data: bool = False) -> NerDataset:
    data_path = Path(data_dir)
    # Support both train_data.txt and train.txt naming conventions
    dfile_candidates = [
        data_path / f"{dtype}_data.txt",
        data_path / f"{dtype}.txt"
    ]
    dfile_path = None
    for candidate in dfile_candidates:
        if candidate.exists():
            dfile_path = candidate
            break
            
    if dfile_path is None:
        raise FileNotFoundError(f"Could not find {dtype} dataset file in {data_dir}. Looked for {[str(c) for c in dfile_candidates]}")

    process_key = tokenizer.name_or_path
    processor_fn = PROCESSOR_MAPPING.get(process_key, None)
    if processor_fn is None:
        if 'phobert' in str(process_key).lower():
            processor_fn = PROCESSOR_MAPPING.get('vinai/phobert-base', convert_word_segment_examples_features)
        else:
            raise KeyError(f"Processor not found for tokenizer '{process_key}'. Supported: {list(PROCESSOR_MAPPING.keys())}")

    model_tag = str(tokenizer.name_or_path).replace('/', '_').replace('\\', '_')
    cached_path = dfile_path.with_name(f"{dfile_path.stem}_{model_tag}.cached")
    legacy_cached_path = dfile_path.with_suffix('.cached')

    if not overwrite_data and cached_path.exists():
        features = _safe_load_torch(cached_path)
    elif not overwrite_data and legacy_cached_path.exists() and ('phobert-base' in model_tag):
        features = _safe_load_torch(legacy_cached_path)
    else:
        features = processor_fn(dfile_path, tokenizer, label2id, header, max_seq_len, use_crf=use_crf)
        torch.save(features, cached_path)
        
    return NerDataset(features=features, device=device)


# ========================= CoNLL evaluation =========================
def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1

def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)

def get_result(correct_chunks, true_chunks, pred_chunks,
               correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks, percent=False)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type
    LOGGER.info(f"Processed {sum_true_counts} tokens with {sum_true_chunks} phrases; "
                f"Found: {sum_pred_chunks} phrases; correct: {sum_correct_chunks}.")
    LOGGER.info("Accuracy: %0.4f; (without `O` tag)" % (nonO_correct_counts / nonO_true_counts))
    LOGGER.info("Accuracy: %0.4f;  Precision: %0.4f; Recall: %0.4f; F1-score: %0.4f"
                % (sum_correct_counts / sum_true_counts, prec, rec, f1))
    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t], percent=False)
        LOGGER.info("%17s: Precision: %0.4f; Recall: %0.4f; F1-score: %0.4f  %d" % (t, prec, rec, f1, pred_chunks[t]))
    return res

def evaluate(true_seqs, pred_seqs, verbose=True):
    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts = count_chunks(true_seqs,
                                                                                                      pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts,
                        verbose=verbose)
    return result

def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []

    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)


# ========================= CLI arguments =========================
def get_predict_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What processs to be run')
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()

def get_test_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What process to be run')
    parser.add_argument("--data_dir", default='Dataset', type=str,
                        help="The input data dir containing dataset files.")
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwrite splitted dataset")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--num_worker', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in "
                             "the main process.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()

def get_train_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What process to be run')
    parser.add_argument("--task", default='pap_ner', type=str, choices=['pap_ner'],
                        help="Training task: pap_ner (Vietnamese e-Government / Legal NER)")
    parser.add_argument("--data_dir", default='Dataset', type=str,
                        help="The input data dir. Should contain train_data.txt, dev_data.txt, test_data.txt")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwrite splitted dataset")
    parser.add_argument("--load_weights", default=None, type=str,
                        help='Path of pretrained file.')
    parser.add_argument("--model_name_or_path", default='vinai/phobert-base', type=str,
                        help="Pre-trained model selected in the list: vinai/phobert-base, trungbb8/distilphobert, vinai/phobert-large...")
    parser.add_argument("--model_arch", default='crf', type=str, choices=['crf'],
                        help="Model architecture: PhoBERT + CRF for sequence tagging")
    parser.add_argument("--use_supcon", action='store_true', default=False,
                        help="Whether to enable Supervised Contrastive Loss (SupCon) for entity representations")
    parser.add_argument("--lambda_supcon", default=0.3, type=float,
                        help="Weight lambda for SupCon loss (default: 0.3)")
    parser.add_argument("--supcon_temp", default=0.07, type=float,
                        help="Temperature tau for SupCon loss (default: 0.07)")
    parser.add_argument("--supcon_mode", default='exact', choices=['exact', 'hierarchical'], type=str,
                        help="Matching strategy for SupCon loss: 'exact' (Exact BIO matching) or 'hierarchical' (Entity-type weighted matching).")
    parser.add_argument("--supcon_alpha", default=0.3, type=float,
                        help="Weak positive weight for same entity-type but different BIO prefix in hierarchical SupCon (default: 0.3).")
    parser.add_argument("--output_dir", default='outputs/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for PhoBERT (paper: 2e-5).")
    parser.add_argument("--classifier_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for CRF and Linear layer (paper: 1e-3).")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--early_stop", default=3.0, type=float,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--run_test", action='store_true', default=False,
                        help="Whether not to run test after train")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded "
                             "in the main process.")
    parser.add_argument("--fp16", action='store_true', default=True,
                        help="Whether to use 16-bit (mixed) precision (through torch.cuda.amp)")
    parser.add_argument("--no_fp16", dest='fp16', action='store_false',
                        help="Disable 16-bit (mixed) precision")
    parser.add_argument('--save_step', type=int, default=20000,
                        help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()


# ========================= Training and testing =========================
def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'classes': args.label2id,
        'args': args
    }
    torch.save(saved_data, saved_file)

def validate(model, task, iterator, cur_epoch: int, output_dir: Union[str, os.PathLike] = './', is_test=False, device='cuda', use_amp=False):
    start_time = time.time()
    model.eval()
    eval_loss = 0.0
    eval_golds, eval_preds = [], []
    # Run one step on sub-dataset
    with torch.no_grad():
        tqdm_desc = f'[EVAL- Epoch {cur_epoch}]'
        eval_bar = tqdm(enumerate(iterator), total=len(iterator), desc=tqdm_desc)
        for idx, batch in eval_bar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if use_amp and str(device).startswith('cuda'):
                amp_ctx = torch.amp.autocast('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
                with amp_ctx:
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            eval_loss_val = outputs.loss.detach().item()
            if not math.isnan(eval_loss_val) and not math.isinf(eval_loss_val):
                eval_loss += eval_loss_val
            active_accuracy = batch['label_masks'].view(-1) != 0
            labels = torch.masked_select(batch['labels'].view(-1), active_accuracy)
            eval_golds.extend(labels.detach().cpu().tolist())
            if isinstance(outputs.tags[-1], list):
                eval_preds.extend(list(itertools.chain(*outputs.tags)))
            else:
                eval_preds.extend(outputs.tags)
    epoch_loss = eval_loss / len(iterator)
    if is_test:
        evaluate([LABEL_MAPPING[task]["id2label"][tag] for tag in eval_golds],
                 [LABEL_MAPPING[task]["id2label"][tag] for tag in eval_preds])
        reports: dict = classification_report(eval_golds, eval_preds,
                                              output_dict=False,
                                              zero_division=0,
                                              digits=4,
                                              target_names=LABEL_MAPPING[task]["label2id"])
        LOGGER.info(reports)
        label_index_to_print = list(range(len(LABEL_MAPPING[task]["label2id"])))
        plot_confusion_matrix(eval_golds, eval_preds,
                              classes=LABEL_MAPPING[task]["label2id"],
                              labels=label_index_to_print,
                              output_dir=output_dir,
                              title=f'Normalized confusion matrix of {task.upper()}',
                              normalize=True)
    else:
        reports: dict = classification_report(eval_golds, eval_preds,
                                              output_dict=True,
                                              zero_division=0)
        epoch_avg_f1 = reports['macro avg']['f1-score']
        epoch_avg_acc = reports['accuracy']
        LOGGER.info(f"\t{'*' * 20}Validate Summary{'*' * 20}")
        LOGGER.info(f"\tValidation Loss: {epoch_loss:.4f};\n"
                    f"\tBIO-Accuracy: {epoch_avg_acc:.4f};\n"
                    f"\tBIO-Macro-F1 score: {epoch_avg_f1:.4f}; "
                    f"\tSpend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
        return epoch_loss, epoch_avg_acc, epoch_avg_f1

def train_one_epoch(model, iterator, optim, cur_epoch: int, max_grad_norm: float = 1.0, scheduler=None, scaler=None, device='cuda'):
    start_time = time.time()
    tr_loss = 0.0
    tr_crf = 0.0
    tr_supcon = 0.0
    has_supcon = False
    model.train()
    use_amp = (scaler is not None and str(device).startswith('cuda'))
    tqdm_bar = tqdm(enumerate(iterator), total=len(iterator), desc=f'[TRAIN-EPOCH {cur_epoch}]')
    for idx, batch in tqdm_bar:
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optim.zero_grad()
        if use_amp:
            amp_ctx = torch.amp.autocast('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
            with amp_ctx:
                outputs = model(**batch)
            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            scale_after = scaler.get_scale()
            if scheduler and (scale_after >= scale_before):
                scheduler.step()
        else:
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optim.step()
            if scheduler:
                scheduler.step()
        loss_val = outputs.loss.detach().item()
        if not math.isnan(loss_val) and not math.isinf(loss_val):
            tr_loss += loss_val
        if hasattr(outputs, 'loss_crf') and outputs.loss_crf is not None:
            has_supcon = True
            crf_val = outputs.loss_crf.detach().item()
            if not math.isnan(crf_val) and not math.isinf(crf_val):
                tr_crf += crf_val
            if hasattr(outputs, 'loss_supcon') and outputs.loss_supcon is not None:
                supcon_val = outputs.loss_supcon.detach().item()
                if not math.isnan(supcon_val) and not math.isinf(supcon_val):
                    tr_supcon += supcon_val

    epoch_loss = tr_loss / len(iterator)
    epoch_crf = tr_crf / len(iterator) if has_supcon else epoch_loss
    epoch_supcon = tr_supcon / len(iterator) if has_supcon else 0.0

    LOGGER.info(f"\t{'*' * 20}Train Summary{'*' * 20}")
    log_msg = f"\tTraining Lr: {optim.param_groups[0]['lr']}; Loss: {epoch_loss:.4f}; "
    if has_supcon:
        log_msg += f"CRF Loss: {epoch_crf:.4f}; SupCon Loss: {epoch_supcon:.4f}; "
    log_msg += f"Spend time: {datetime.timedelta(seconds=(time.time() - start_time))}"
    LOGGER.info(log_msg)
    return epoch_loss, epoch_crf, epoch_supcon

def train():
    args = get_train_argument()
    LOGGER.info(f"Arguments: {args}")
    set_ramdom_seed(args.seed)
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    use_crf = True if 'crf' in args.model_arch else False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found!'
    setattr(args, 'label2id', LABEL_MAPPING[args.task]['label2id'])
    setattr(args, 'id2label', LABEL_MAPPING[args.task]['id2label'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = build_dataset(args.data_dir,
                                  tokenizer,
                                  label2id=args.label2id,
                                  header=LABEL_MAPPING[args.task]['header'],
                                  dtype='train',
                                  max_seq_len=args.max_seq_length,
                                  device='cpu',
                                  use_crf=use_crf,
                                  overwrite_data=args.overwrite_data)
    eval_dtype = 'dev' if (os.path.exists(os.path.join(args.data_dir, 'dev_data.txt')) or os.path.exists(os.path.join(args.data_dir, 'dev.txt'))) else 'test'
    eval_dataset = build_dataset(args.data_dir,
                                 tokenizer,
                                 label2id=args.label2id,
                                 header=LABEL_MAPPING[args.task]['header'],
                                 dtype=eval_dtype,
                                 max_seq_len=args.max_seq_length,
                                 device='cpu',
                                 use_crf=use_crf,
                                 overwrite_data=args.overwrite_data)

    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=len(args.label2id),
                                        finetuning_task=args.task)
    setattr(config, 'use_supcon', getattr(args, 'use_supcon', False))
    setattr(config, 'lambda_supcon', getattr(args, 'lambda_supcon', 0.3))
    setattr(config, 'supcon_temp', getattr(args, 'supcon_temp', 0.07))
    setattr(config, 'supcon_mode', getattr(args, 'supcon_mode', 'exact'))
    setattr(config, 'supcon_alpha', getattr(args, 'supcon_alpha', 0.3))
    o_id = args.label2id.index('O') if isinstance(args.label2id, list) else args.label2id.get('O', 0)
    setattr(config, 'o_label_id', o_id)

    model_clss = PhoBertV2Crf
    model = model_clss.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,
                                       config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.load_weights is not None:
        LOGGER.info(f'Load pretrained model weights from "{args.load_weights}"')
        if device == 'cpu':
            checkpoint_data = safe_torch_load(args.load_weights, map_location='cpu')
        else:
            checkpoint_data = safe_torch_load(args.load_weights)
        model.load_state_dict(checkpoint_data['model'], strict=False)
        checkpoint_data = None

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    if hasattr(model, 'roberta'):
        bert_param_optimizer = list(model.roberta.named_parameters())
    elif hasattr(model, 'bert'):
        bert_param_optimizer = list(model.bert.named_parameters())
    else:
        raise AttributeError(f"Unsupported model architecture: cannot find encoder attribute ('roberta' or 'bert') on {type(model)}.")
    ner_param_optimizer = list(model.classifier.named_parameters())
    if 'lstm' in args.model_arch:
        ner_param_optimizer.extend(list(model.lstm.named_parameters()))
    if 'crf' in args.model_arch:
        ner_param_optimizer.extend(list(model.crf.named_parameters()))
    if getattr(model, 'projection_head', None) is not None:
        ner_param_optimizer.extend(list(model.projection_head.named_parameters()))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in ner_param_optimizer],
         'lr': args.classifier_learning_rate, 'weight_decay': args.weight_decay}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    train_steps_per_epoch = len(train_dataset) // args.train_batch_size
    total_train_steps = (train_steps_per_epoch * int(args.epochs)) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_proportion * total_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_train_steps)
    use_amp = (not args.no_cuda and torch.cuda.is_available() and getattr(args, 'fp16', True))
    scaler = (torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()) if use_amp else None
    pin_memory = (str(device).startswith('cuda'))
    LOGGER.info(f"Using Mixed Precision (AMP FP16): {use_amp}, Pin Memory: {pin_memory}, Num Workers: {args.num_workers}")
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                num_workers=args.num_workers,
                                pin_memory=pin_memory)
    eval_iterator = DataLoader(eval_dataset,
                               batch_size=args.eval_batch_size,
                               num_workers=args.num_workers,
                               pin_memory=pin_memory)
    best_score = 0.0
    best_loss = float('inf')
    cumulative_early_steps = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Early stopping. Check your saved model.")
            break
        LOGGER.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        # Fit model with dataset
        tr_loss, tr_crf, tr_supcon = train_one_epoch(model=model,
                                                     optim=optimizer,
                                                     iterator=train_iterator,
                                                     cur_epoch=epoch,
                                                     max_grad_norm=args.max_grad_norm,
                                                     scheduler=scheduler,
                                                     scaler=scaler,
                                                     device=device)

        # Validate trained model on dataset
        eval_loss, eval_acc, eval_f1 = validate(model=model,
                                 task=args.task,
                                 iterator=eval_iterator,
                                 cur_epoch=epoch,
                                 is_test=False,
                                 device=device,
                                 use_amp=use_amp)

        LOGGER.info(f"\t{'*' * 20}Epoch Summary{'*' * 20}")
        LOGGER.info(f"\tEpoch Loss = {eval_loss:.6f} ; Best loss = {best_loss:.6f}")
        LOGGER.info(f"\tEpoch BIO-F1 score = {eval_f1:.6f} ; Best score = {best_score:.6f}")

        if eval_loss < best_loss:
            best_loss = eval_loss
        if eval_f1 > best_score:
            cumulative_early_steps = 0
            best_score = eval_f1
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***New best model, saving to {saved_file}...***")
            save_model(args, saved_file, model)
        else:
            cumulative_early_steps += 1
    if args.run_test:
        test_dataset = build_dataset(args.data_dir,
                                     tokenizer,
                                     label2id=args.label2id,
                                     header=LABEL_MAPPING[args.task]['header'],
                                     dtype='test',
                                     max_seq_len=args.max_seq_length,
                                     device='cpu',
                                     use_crf=use_crf,
                                     overwrite_data=args.overwrite_data)
        test_iterator = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        if device == 'cpu':
            checkpoint_data = safe_torch_load(args.output_dir + f"/best_model.pt", map_location='cpu')
        else:
            checkpoint_data = safe_torch_load(args.output_dir + f"/best_model.pt")
        model.load_state_dict(checkpoint_data['model'], strict=False)
        validate(model=model,
                 task=args.task,
                 iterator=test_iterator,
                 cur_epoch=0,
                 is_test=True,
                 output_dir=args.output_dir,
                 device=device,
                 use_amp=use_amp)

def test():
    args = get_test_argument()
    if not args.no_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    LOGGER.info(f"Loading checkpoint from '{args.model_path}' on device '{device}'...")
    checkpoint = safe_torch_load(args.model_path, map_location=device)
    train_args = checkpoint.get('args', None)

    task = getattr(train_args, 'task', 'pap_ner') if train_args else 'pap_ner'
    model_name_or_path = getattr(train_args, 'model_name_or_path', 'vinai/phobert-base') if train_args else 'vinai/phobert-base'
    max_seq_length = getattr(train_args, 'max_seq_length', 256) if train_args else 256
    classes = checkpoint.get('classes', LABEL_MAPPING[task]['label2id'])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path,
                                        num_labels=len(classes),
                                        finetuning_task=task)
    use_supcon = getattr(train_args, 'use_supcon', False) if train_args else False
    setattr(config, 'use_supcon', use_supcon)
    model = PhoBertV2Crf.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                       config=config)
    model.resize_token_embeddings(len(tokenizer))
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        LOGGER.warning(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        LOGGER.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    model.to(device)

    num_workers = getattr(args, 'num_workers', getattr(args, 'num_worker', 0))
    pin_memory = (str(device).startswith('cuda'))
    use_amp = (str(device).startswith('cuda'))
    test_dataset = build_dataset(args.data_dir,
                                 tokenizer,
                                 label2id=classes,
                                 header=LABEL_MAPPING[task]['header'],
                                 dtype='test',
                                 max_seq_len=max_seq_length,
                                 device='cpu',
                                 use_crf=True,
                                 overwrite_data=args.overwrite_data)
    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory)

    output_dir = os.path.dirname(args.model_path) or './'
    LOGGER.info(f"Running evaluation on test set ({len(test_dataset)} samples)...")
    validate(model=model,
             task=task,
             iterator=test_iterator,
             cur_epoch=0,
             is_test=True,
             output_dir=output_dir,
             device=device,
             use_amp=use_amp)
    return model, test_iterator, classes, device, use_amp


# ========================= Prediction =========================
class ViTagger(object):
    def __init__(self, model_path: Union[str or os.PathLike],  no_cuda=False):
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        print("[ViTagger] VnCoreNLP loading ...")
        self.rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        print("[ViTagger] Model loading ...")
        self.model, self.tokenizer,  self.max_seq_len, self.label2id, self.use_crf = self.load_model(model_path, device=self.device)
        self.id2label = {idx: label for idx, label in enumerate(self.label2id)}
        print("[ViTagger] All ready!")

    @staticmethod
    def load_model(model_path: Union[str or os.PathLike],  device='cpu'):
        if device == 'cpu':
            checkpoint_data = safe_torch_load(model_path, map_location='cpu')
        else:
            checkpoint_data = safe_torch_load(model_path)
        args = checkpoint_data["args"]
        max_seq_len = args.max_seq_length
        use_crf = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(args.label2id))
        use_supcon = getattr(args, 'use_supcon', False)
        setattr(config, 'use_supcon', use_supcon)
        model_clss = PhoBertCrf
        model = model_clss(config=config)
        model.load_state_dict(checkpoint_data['model'], strict=False)
        model.to(device)
        model.eval()

        return model, tokenizer, max_seq_len, args.label2id, use_crf

    def preprocess(self, in_raw: str):
        norm_text = normalize_text(in_raw)
        sents = []
        sentences = self.rdrsegmenter.tokenize(norm_text)
        for sentence in sentences:
            sents.append(sentence)
        return sents

    def convert_tensor(self, tokens):
        seq_len = len(tokens)
        encoding = self.tokenizer(tokens,
                                  padding='max_length',
                                  truncation=True,
                                  is_split_into_words=True,
                                  max_length=self.max_seq_len)
        is_phobert = ('phobert' in str(self.tokenizer.name_or_path).lower()) or ('Phobert' in self.tokenizer.__class__.__name__)
        if is_phobert:
            subwords = self.tokenizer.tokenize(' '.join(tokens))
            valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
            label_marks = np.zeros(len(encoding.input_ids), dtype=int)
            i = 1
            for idx, subword in enumerate(subwords[:self.max_seq_len - 2]):
                if idx != 0 and subwords[idx - 1].endswith("@@"):
                    continue
                if self.use_crf:
                    valid_ids[i - 1] = idx + 1
                else:
                    valid_ids[idx + 1] = 1
                i += 1
        else:
            valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
            label_marks = np.zeros(len(encoding.input_ids), dtype=int)
            i = 1
            word_ids = encoding.word_ids()
            for idx in range(1, len(word_ids)):
                if word_ids[idx] is not None and word_ids[idx] != word_ids[idx - 1]:
                    if self.use_crf:
                        valid_ids[i - 1] = idx
                    else:
                        valid_ids[idx] = 1
                    i += 1
        if self.max_seq_len >= seq_len + 2:
            label_marks[:seq_len] = [1] * seq_len
        else:
            label_marks[:-2] = [1] * (self.max_seq_len - 2)
        if self.use_crf and label_marks[0] == 0:
            raise ValueError(f"{tokens} have mark == 0 at index 0!")
        item = {key: torch.as_tensor([val]).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['valid_ids'] = torch.as_tensor([valid_ids]).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor([label_marks if self.use_crf else valid_ids]).to(self.device, dtype=torch.long)
        return item

    def extract_entity_doc(self, in_raw: str):
        sents = self.preprocess(in_raw)
        entities_doc = []
        for sent in sents:
            item = self.convert_tensor(sent)
            with torch.no_grad():
                outputs = self.model(**item)
            entity = None
            if isinstance(outputs.tags[0], list):
                tags = list(itertools.chain(*outputs.tags))
            else:
                tags = outputs.tags
            for w, l in list(zip(sent, tags)):
                w = w.replace("_", " ")
                tag = self.id2label[l]
                if not tag == 'O':
                    prefix, tag = tag.split('-')
                    if entity is None:
                        entity = (w, tag)
                    else:
                        if entity[-1] == tag:
                            if prefix == 'I':
                                entity = (entity[0] + f' {w}', tag)
                            else:
                                entities_doc.append(entity)
                                entity = (w, tag)
                        else:
                            entities_doc.append(entity)
                            entity = (w, tag)
                elif entity is not None:
                    entities_doc.append(entity)
                    entities_doc.append((w, 'O'))
                    entity = None
                else:
                    entities_doc.append((w, 'O'))
                    entity = None
        return entities_doc

    def __call__(self, in_raw: str):
        sents = self.preprocess(in_raw)
        entites = []
        for sent in sents:
            item = self.convert_tensor(sent)
            with torch.no_grad():
                outputs = self.model(**item)
            entity = None
            if isinstance(outputs.tags[0], list):
                tags = list(itertools.chain(*outputs.tags))
            else:
                tags = outputs.tags
            for w, l in list(zip(sent, tags)):
                w = w.replace("_", " ")
                tag = self.id2label[l]
                if not tag == 'O':
                    prefix, tag = tag.split('-')
                    if entity is None:
                        entity = (w, tag)
                    else:
                        if entity[-1] == tag:
                            if prefix == 'I':
                                entity = (entity[0] + f' {w}', tag)
                            else:
                                entites.append(entity)
                                entity = (w, tag)
                        else:
                            entites.append(entity)
                            entity = (w, tag)
                elif entity is not None:
                    entites.append(entity)
                    entity = None
                else:
                    entity = None
        return entites


# ========================= PhoBERT-base adaptation =========================
MODEL_ID = "vinai/phobert-base"
OUTPUT_DIR = "outputs/phobert_base_crf_hierarchical_supcon_20261023"


class PhoBertV2Crf(PhoBertCrf):
    # Kế thừa toàn bộ CRF, SupCon, projection head và forward từ PhoBertCrf.
    # Giữ gradient checkpointing như notebook ELECTRA trên RTX 3090.
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.gradient_checkpointing_enable()


def encode_phobert_words(tokens, tag_ids, tokenizer, max_seq_len=256):
    # PhoBERT dùng BPE với ký hiệu @@ cho các subword còn tiếp diễn.
    # Không dùng offset_mapping / fast tokenizer riêng của ELECTRA.
    words = [str(token).strip().replace(" ", "_") for token in tokens]
    sentence = " ".join(words)
    encoding = tokenizer(sentence, padding="max_length", truncation=True,
                         max_length=max_seq_len)
    subwords = tokenizer.tokenize(sentence)
    usable = subwords[:max_seq_len - 2]
    valid_ids = np.zeros(max_seq_len, dtype=np.int64)
    label_masks = np.zeros(max_seq_len, dtype=np.int64)
    labels = np.zeros(max_seq_len, dtype=np.int64)
    kept = 0
    for idx, subword in enumerate(usable):
        # Chỉ chọn subword đầu tiên của từng từ được tách bằng khoảng trắng.
        if idx > 0 and usable[idx - 1].endswith("@@"):
            continue
        # Nếu truncate ở giữa một từ, bỏ từ chưa có đủ subword.
        last = idx
        while last + 1 < len(usable) and usable[last].endswith("@@"):
            last += 1
        if usable[last].endswith("@@"):
            break
        if kept >= len(tag_ids):
            break
        valid_ids[kept] = idx + 1  # Vị trí 0 dành cho <s>.
        label_masks[kept] = 1
        labels[kept] = tag_ids[kept]
        kept += 1
    if kept == 0:
        raise ValueError(f"Không căn được từ đầu tiên: {words[:1]}")
    values = dict(encoding)
    values.update(valid_ids=valid_ids, label_masks=label_masks, labels=labels)
    return NerFeatures(**values)


def convert_phobert_examples_features(data_path, tokenizer, label2id, header_names,
                                      max_seq_len=256, use_crf=True):
    if not use_crf:
        raise ValueError("Notebook này giữ nguyên kiến trúc CRF của baseline")
    features, tokens, tag_ids = [], [], []
    with open(data_path, encoding="utf-8-sig") as counter:
        total_lines = sum(1 for _ in counter)
    with open(data_path, encoding="utf-8-sig") as stream:
        progress = tqdm(stream, total=total_lines, desc=f"Tokenize {Path(data_path).name}",
                        unit="lines", mininterval=2, file=sys.stdout)
        for line_number, line in enumerate(progress, 1):
            fields = line.split()
            if not fields:
                if tokens:
                    features.append(encode_phobert_words(tokens, tag_ids, tokenizer, max_seq_len))
                    tokens, tag_ids = [], []
                continue
            if len(fields) < 4:
                raise ValueError(f"{data_path}:{line_number}: cần ít nhất 4 cột CoNLL")
            token, ner = fields[0], fields[3]
            if ner not in label2id:
                raise ValueError(f"{data_path}:{line_number}: nhãn NER không hợp lệ: {ner}")
            tokens.append(token)
            tag_ids.append(label2id.index(ner))
    if tokens:
        features.append(encode_phobert_words(tokens, tag_ids, tokenizer, max_seq_len))
    print(f"Đã căn nhãn {len(features):,} câu từ {Path(data_path).name}", flush=True)
    return features


PROCESSOR_MAPPING[MODEL_ID] = convert_phobert_examples_features
print("Đã đăng ký PhoBERT bản gốc (vinai/phobert-base); CRF/SupCon và mọi tham số train gốc được giữ nguyên.")
