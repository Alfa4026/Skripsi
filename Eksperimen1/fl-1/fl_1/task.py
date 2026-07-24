# fl_1/task.py

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from sklearn.metrics import f1_score

# =========================
# PATH SETUP (PAKAI DATA LAMA)
# =========================
ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "indobenchmark/indobert-base-p1"
WARMSTART_PATH = ROOT / "../models/model_warmstart2.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ["input_ids", "attention_mask"]

# =========================
# DATASET
# =========================
class HateSpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols):
        self.input_ids = df["input_ids"].tolist()
        self.attention_mask = df["attention_mask"].tolist()
        self.labels = df[label_cols].values.astype("float32")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }

# =========================
# MODEL
# =========================
class IndoBertMultiLabel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, pos_weight=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))

        loss = None
        if labels is not None:
            if pos_weight is not None:
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return loss, logits

# =========================
# UTILS
# =========================
def infer_label_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in FEATURE_COLS]

def compute_pos_weight(df: pd.DataFrame, label_cols):
    counts = df[label_cols].sum().values
    pos_weight = (len(df) - counts) / (counts + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float, device=DEVICE)

# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(
    model,
    loader,
    optimizer,
    pos_weight=None,
):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        loss, _ = model(ids, mask, labels, pos_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(
    model,
    loader,
    threshold=0.4,
    pos_weight=None,
) -> Dict[str, float]:
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0

    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        loss, logits = model(ids, mask, labels, pos_weight)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()

        all_y.append(labels.cpu())
        all_p.append(preds.cpu())

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()

    return {
        "loss": total_loss / len(loader),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

# =========================
# FACTORY (DIPAKAI CLIENT)
# =========================
def build_model(num_labels: int) -> nn.Module:
    model = IndoBertMultiLabel(num_labels).to(DEVICE)

    if WARMSTART_PATH.exists():
        model.load_state_dict(torch.load(WARMSTART_PATH, map_location=DEVICE))

    return model
