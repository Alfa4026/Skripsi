# fl_1/client_app.py

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl
from flwr.common import Context



from fl_1.task import (
    HateSpeechDataset,
    infer_label_cols,
    compute_pos_weight,
    build_model,
    train_one_epoch,
    evaluate,
    DEVICE,
)

# =========================
# PATH
# =========================
ROOT = Path(__file__).resolve().parents[1]

# =========================
# UTILS PARAM
# =========================
def get_parameters(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]

def set_parameters(model, parameters):
    params = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
    model.load_state_dict(state_dict, strict=True)

# =========================
# CLIENT CLASS
# =========================
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data_dir: Path):
        self.cid = int(cid)
        self.data_dir = data_dir

        df = pd.read_pickle(self.data_dir / f"client_{self.cid:02d}.pkl")
        self.label_cols = infer_label_cols(df)
        self.num_labels = len(self.label_cols)

        # split local
        idx = np.random.permutation(len(df))
        split = int(0.9 * len(df))
        df_tr = df.iloc[idx[:split]]
        df_va = df.iloc[idx[split:]]

        self.train_loader = DataLoader(
            HateSpeechDataset(df_tr, self.label_cols),
            batch_size=8,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            HateSpeechDataset(df_va, self.label_cols),
            batch_size=8,
            shuffle=False,
        )

        self.pos_weight = compute_pos_weight(df_tr, self.label_cols)
        self.model = build_model(self.num_labels).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_one_epoch(
            self.model,
            self.train_loader,
            self.optimizer,
            pos_weight=self.pos_weight,
        )
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = evaluate(
            self.model,
            self.val_loader,
            threshold=0.4,
            pos_weight=self.pos_weight,
        )
        return (
            float(metrics["loss"]),
            len(self.val_loader.dataset),
            {
                "micro_f1": float(metrics["micro_f1"]),
                "macro_f1": float(metrics["macro_f1"]),
            },
        )

# client_fn HARUS terima Context
def client_fn(context: Context):
    cid = context.node_config["partition-id"]
    data_dir = ROOT / "../fl_clients_iid_10"
    return FLClient(str(cid), data_dir)

# =========================
# CLIENT APP (WAJIB)
# =========================
app = fl.client.ClientApp(client_fn=client_fn)

