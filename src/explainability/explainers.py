import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import torch

from src.models.tabular_baseline import prepare_features
from src.models.temporal_gnn import TemporalAttentionPredictor


def _attach_oem_tabular_features(raw_df: pd.DataFrame, base_X: pd.DataFrame) -> pd.DataFrame:
    """Appends point-in-time OEM spare-part criticality and upstream lead-time features."""
    X = base_X.copy()
    ved_map = {"V": 3, "E": 2, "D": 1}
    abc_map = {"A": 3, "B": 2, "C": 1}

    oem_cols = {
        "VED_Criticality": raw_df["ved"].map(ved_map).fillna(2),
        "ABC_Class": raw_df["abc"].map(abc_map).fillna(2),
        "Unit_Cost_USD": raw_df["unit_cost"],
        "VOR_Penalty_Day": raw_df["vor_penalty_day"],
        "Tier1_Proc_Days": raw_df["tier1_proc_days"],
        "Holding_Cost_Day": raw_df["holding_cost_day"],
    }
    for col_name, series in oem_cols.items():
        if col_name not in X.columns:
            X[col_name] = series.values
    return X


def generate_shap_explanations():
    print("[*] Loading 4-Echelon OEM parquet splits for Tabular SHAP...")
    train_df = pd.read_parquet("data/processed/train_oem.parquet")
    test_df = pd.read_parquet("data/processed/test_oem.parquet")

    X_train = _attach_oem_tabular_features(train_df, prepare_features(train_df))
    X_test = _attach_oem_tabular_features(test_df, prepare_features(test_df))
    y_train = train_df["target_is_delayed"]

    clf = lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    print("[*] Calculating TreeSHAP values on 5,000 held-out OEM orders...")
    explainer = shap.TreeExplainer(clf)
    X_test_sample = X_test.sample(n=5000, random_state=42)
    shap_values = explainer.shap_values(X_test_sample)

    if isinstance(shap_values, list):
        plot_values = shap_values[1]
    elif getattr(shap_values, "ndim", 2) == 3:
        plot_values = shap_values[:, :, 1]
    else:
        plot_values = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(plot_values, X_test_sample, show=False)
    plt.title("SHAP Feature Importance: Drivers of Dispatch Delay (4-Echelon OEM)", fontsize=13, pad=18)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("[✓] Saved updated Tabular SHAP plot -> shap_summary.png")


def generate_gnn_attributions():
    print("\n[*] Computing 3-Hop Temporal GATv2 Edge & Echelon Attributions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_graph = torch.load("data/processed/test_oem_graph.pt", weights_only=False).to(device)

    model = TemporalAttentionPredictor(
        node_feat_dim=test_graph.x.size(1),
        edge_attr_dim=test_graph.edge_attr.size(1),
        hidden_dim=64,
        heads=4,
    ).to(device)
    model.load_state_dict(
        torch.load("checkpoints/temporal_gnn_oem_regression.pt", map_location=device, weights_only=True)
    )
    model.eval()

    x_in = test_graph.x.clone().detach().requires_grad_(True)
    edge_in = test_graph.edge_attr.clone().detach().requires_grad_(True)
    ship_mask = test_graph.is_shipment_edge

    preds = model(x_in, test_graph.edge_index, edge_in)[ship_mask]
    preds.sum().backward()

    # 1. Edge Feature Attribution (Gradient x Input + Raw Gradient Sensitivity)
    edge_grad = edge_in.grad[ship_mask].detach().cpu().numpy()
    edge_val = edge_in[ship_mask].detach().cpu().numpy()
    grad_x_input = np.abs(edge_grad * edge_val).mean(axis=0)
    grad_norm = grad_x_input / max(grad_x_input.sum(), 1e-8) * 100.0

    edge_feature_names = [
        "Shipping Mode (SLA)",
        "Scheduled Lead Time",
        "VED Part Criticality",
        "Unit Cost (Log USD)",
        "VOR Penalty ($/day)",
        "Tier-1 Supplier LT",
        "Order Weekday (Sin)",
        "Order Weekday (Cos)",
    ]

    print("\n--- GATv2 Edge Feature Attribution (4-Echelon OEM Test Graph) ---")
    for name, pct, raw_val in sorted(zip(edge_feature_names, grad_norm, grad_x_input), key=lambda t: -t[1]):
        print(f"  {name:<24}: {pct:6.2f}%  (Mean |Grad*Input| = {raw_val:.4f})")

    # 2. 4-Echelon Node Sensitivity across Graph Topology
    node_grad = x_in.grad.detach().abs().mean(dim=1).cpu().numpy()
    echelon_names = [
        "Echelon 1: Tier-1 Suppliers (N4-N5)",
        "Echelon 2: OEM Central Plant (N3)",
        "Echelon 3: Regional PDCs (23 Hubs)",
        "Echelon 4: Dealerships (44 Markets)",
    ]
    echelon_scores = np.array([
        node_grad[0:2].mean(),
        node_grad[2:3].mean(),
        node_grad[3:26].mean(),
        node_grad[26:70].mean(),
    ])
    echelon_pct = echelon_scores / max(echelon_scores.sum(), 1e-8) * 100.0

    print("\n--- GATv2 Multi-Echelon Structural Sensitivity ---")
    for ename, epct in zip(echelon_names, echelon_pct):
        print(f"  {ename:<36}: {epct:6.2f}%")

    # Plot 2-Panel Attribution Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1.0]})

    order = np.argsort(grad_norm)
    ax1.barh([edge_feature_names[i] for i in order], grad_norm[order], color="#2b6cb0", edgecolor="none")
    ax1.set_xlabel("Relative Attribution Share (%)", fontsize=10)
    ax1.set_title("GATv2 Shipment Edge Feature Importance", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    e_order = np.arange(len(echelon_names))[::-1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][::-1]
    ax2.barh([echelon_names[i] for i in e_order], echelon_pct[e_order], color=colors)
    ax2.set_xlabel("Relative Node Gradient Sensitivity (%)", fontsize=10)
    ax2.set_title("4-Echelon Topology Sensitivity", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Temporal GATv2 Explainability: 4-Echelon Heavy Machinery OEM Network", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("gnn_oem_attribution.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("\n[✓] Saved GATv2 Attribution plot -> gnn_oem_attribution.png")


if __name__ == "__main__":
    generate_shap_explanations()
    generate_gnn_attributions()