from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error, r2_score
)
from src.evaluation.scm_metrics import evaluate_scm_economics


class EdgePredictor(nn.Module):
    def __init__(self, node_feat_dim: int, edge_attr_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Encodes 6D OEM echelon features (echelon flags, stage processing time, holding cost)
        self.node_emb = nn.Linear(node_feat_dim, hidden_dim)

        # 2-hop GraphSAGE message passing propagates Tier-1 -> OEM Plant -> PDC states
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Prediction head: [Source PDC, Target Dealer, Shipment Edge Features] -> Delay
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.node_emb(x))
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.predictor(edge_repr).squeeze(-1)


def train_and_evaluate(task_name: str, train_data, val_data, test_data, save_dir: Path = Path("checkpoints")):
    print(f"\n{'='*55}\nSTATIC GNN (4-ECHELON OEM) | TASK: {task_name.upper()}\n{'='*55}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = EdgePredictor(
        node_feat_dim=train_data.x.size(1),
        edge_attr_dim=train_data.edge_attr.size(1),
        hidden_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    is_clf = task_name == "classification"
    criterion = nn.BCEWithLogitsLoss() if is_clf else nn.MSELoss()

    train_mask = train_data.is_shipment_edge
    val_mask = val_data.is_shipment_edge
    test_mask = test_data.is_shipment_edge

    y_train = (train_data.y_cls.float() if is_clf else train_data.y_reg)[train_mask]
    y_val = (val_data.y_cls.float() if is_clf else val_data.y_reg)[val_mask]
    y_test = (test_data.y_cls.float() if is_clf else test_data.y_reg)[test_mask]

    best_val_loss = float("inf")
    best_model_state = None
    patience, patience_counter = 20, 0

    print("[*] Training 4-Echelon Static GNN (GraphSAGE)...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = criterion(out[train_mask], y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.edge_index, val_data.edge_attr)
            val_loss = criterion(val_out[val_mask], y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[*] Early stopping triggered at epoch {epoch + 1} (Best Val Loss: {best_val_loss:.4f})")
            break

    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"static_gnn_oem_{task_name}.pt"
    torch.save(best_model_state, ckpt_path)
    print(f"[✓] Saved checkpoint -> {ckpt_path}")

    # Evaluation on Held-Out Test Set (Shipment Edges Only)
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x, test_data.edge_index, test_data.edge_attr)[test_mask]

    print("\n--- Static GNN OEM Results (Held-Out Test Set) ---")
    preds_raw = test_out.cpu().numpy()
    y_true = y_test.cpu().numpy()

    if is_clf:
        preds_prob = torch.sigmoid(test_out).cpu().numpy()
        preds_bin = (preds_prob > 0.5).astype(int)

        print(f"Accuracy : {accuracy_score(y_true, preds_bin):.4f}")
        print(f"Precision: {precision_score(y_true, preds_bin, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_true, preds_bin):.4f}")
        print(f"Macro F1 : {f1_score(y_true, preds_bin, average='macro'):.4f}")
        print(f"ROC-AUC  : {roc_auc_score(y_true, preds_prob):.4f}")
    else:
        print(f"MAE  : {mean_absolute_error(y_true, preds_raw):.4f} days")
        print(f"RMSE : {root_mean_squared_error(y_true, preds_raw):.4f} days")
        print(f"R²   : {r2_score(y_true, preds_raw):.4f}")

        # Industrial Engineering / SCM Economic Evaluation
        sched_lt = test_data.edge_attr[test_mask, 1].cpu().numpy() * 7.0
        vor_pen = test_data.vor_penalty[test_mask].cpu().numpy()
        hold_cost = test_data.holding_cost[test_mask].cpu().numpy()
        ved_num = test_data.edge_attr[test_mask, 2].cpu().numpy()
        ved_flags = np.where(ved_num > 0.8, "V", np.where(ved_num > 0.3, "E", "D"))

        scm_res = evaluate_scm_economics(
            actual_delay_days=y_true,
            pred_delay_days=preds_raw,
            scheduled_lt_days=sched_lt,
            vor_penalty_day=vor_pen,
            holding_cost_day=hold_cost,
            ved_flags=ved_flags,
        )
        print("\n--- SCM Economic Impact (Static GNN) ---")
        for k, v in scm_res.items():
            print(f"  {k:<30}: {v}")


if __name__ == "__main__":
    print("[*] Loading 4-Echelon OEM PyG Graph Partitions...")
    train_graph = torch.load("data/processed/train_oem_graph.pt", weights_only=False)
    val_graph = torch.load("data/processed/val_oem_graph.pt", weights_only=False)
    test_graph = torch.load("data/processed/test_oem_graph.pt", weights_only=False)

    train_and_evaluate("classification", train_graph, val_graph, test_graph)
    train_and_evaluate("regression", train_graph, val_graph, test_graph)