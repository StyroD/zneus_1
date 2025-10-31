"""
sweep_train.py — clean refactored version for W&B sweeps
--------------------------------------------------------
- Loads configs (default.yaml + wandb.yaml)
- Builds and trains MLP
- Handles W&B sweep overrides
- Clean logging and evaluation
"""

import os
import yaml
import wandb
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# =============================
# === 1. CONFIGS & SETUP ===
# =============================

yaml_path = "configs/default.yaml"
wandb_path = "configs/wandb.yaml"

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

with open(wandb_path, "r") as f:
    wandb_config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# === 2. MODEL DEFINITION ===
# =============================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim,
                 dropout=0.0, batch_norm=False, activation="relu"):
        super().__init__()
        layers = []
        act_fn = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }[activation]

        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h

        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =============================
# === 3. DATA PREPARATION ===
# =============================

df = pd.read_csv("data/bioresponse_filtered.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1111, random_state=42, stratify=y_train_full
)

print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

def to_tensor(data, device):
    return torch.tensor(data.values, dtype=torch.float32).to(device)

X_train_t, y_train_t = to_tensor(X_train, device), to_tensor(y_train, device)
X_val_t, y_val_t = to_tensor(X_val, device), to_tensor(y_val, device)
X_test_t, y_test_t = to_tensor(X_test, device), to_tensor(y_test, device)

# =============================
# === 4. TRAIN FUNCTION ===
# =============================

def train():
    # === Load and merge configs ===
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Start W&B run
    wandb.init(
        project=wandb_config["wandb"]["project"],
        entity=wandb_config["wandb"]["entity"],
        name=wandb_config["wandb"].get("run_name", wandb_config["wandb"]["name"]),
        group=config["experiment"].get("group", None),
        notes=wandb_config["wandb"].get("notes", ""),
        config=wandb_config,
    )

    # Merge sweep parameters into config if any
    if wandb.run is not None:
        for key, value in dict(wandb.config).items():
            parts = key.split(".")
            sub = config
            for p in parts[:-1]:
                sub = sub[p]
            sub[parts[-1]] = value

    # === Build model, optimizer, loss ===
    model = MLP(
        input_dim=X_train_t.shape[1],
        hidden_layers=config["model"]["hidden_layers"],
        output_dim=config["model"]["output_dim"],
        dropout=config["model"]["dropout"],
        batch_norm=config["model"]["batch_norm"],
        activation=config["model"]["activation"]
    ).to(device)

    optimizer_dict = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
    }
    optimizer_class = optimizer_dict[config["training"]["optimizer"]]
    optimizer = optimizer_class(model.parameters(), lr=float(config["training"]["learning_rate"]))
    loss_fn = nn.BCELoss()

    wandb.watch(model, log="all", log_freq=100)

    # === DataLoaders ===
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    epochs = config["training"]["epochs"]
    log_interval = config["experiment"]["log_interval"]

    # === Training Loop ===
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        avg_train_loss = training_loss / len(train_loader)

        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend((outputs.cpu().numpy() >= 0.5).astype(int))

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)

        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} "
                  f"| Train Loss: {avg_train_loss:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f} "
                  f"| Val Acc: {val_acc:.4f}")

        if wandb_config["wandb"]["enabled"]:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            })

    # === Final Confusion Matrix ===
    if wandb_config["wandb"]["enabled"]:
        wandb.log({
            "final_conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=["negative", "positive"]
            )
        })

# =============================
# === 5. MAIN / SWEEP ===
# =============================

if __name__ == "__main__":
    # ---- NORMAL TRAIN ----
    # train()

    # ---- OR SWEEP MODE ----
    with open("configs/sweep.yaml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project=wandb_config["wandb"]["project"])
    wandb.agent(sweep_id, function=train, count=10)
