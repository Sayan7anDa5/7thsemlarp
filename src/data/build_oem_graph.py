from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from stockpyl.instances import load_instance


def _build_node_registry(df_train: pd.DataFrame) -> tuple[dict[str, int], torch.Tensor]:
    """
    Constructs a deterministic 70-node registry across all 4 OEM echelons from the training split:
      - Echelon 1 (Nodes 0-1): Tier-1 Component Suppliers (GSM Nodes 4 & 5)
      - Echelon 2 (Node 2):    OEM Central Assembly & Parts Depot (GSM Node 3)
      - Echelon 3 (Nodes 3-25): Regional Parts Distribution Centers (23 PDCs)
      - Echelon 4 (Nodes 26-69): Authorized Dealerships & Fleet Customers (44 Dealers)
    Returns (node_to_idx, node_features [N, 6]).
    """
    net = load_instance("problem_6_9")
    gsm_nodes = {n.index: n for n in net.nodes}

    t1_nodes = sorted(df_train["echelon_1_tier1"].unique())
    oem_nodes = sorted(df_train["echelon_2_oem_plant"].unique())
    pdc_nodes = sorted(df_train["echelon_3_pdc"].unique())
    dealer_nodes = sorted(df_train["echelon_4_dealer"].unique())

    all_nodes = t1_nodes + oem_nodes + pdc_nodes + dealer_nodes
    node_to_idx = {name: idx for idx, name in enumerate(all_nodes)}

    # Node feature vector: [ech_1, ech_2, ech_3, ech_4, stage_proc_days, stage_hc_rate]
    feats = []
    for name in all_nodes:
        if name in t1_nodes:
            pt = gsm_nodes[5].processing_time if "N5" in name else gsm_nodes[4].processing_time
            hc = gsm_nodes[5].local_holding_cost if "N5" in name else gsm_nodes[4].local_holding_cost
            feats.append([1.0, 0.0, 0.0, 0.0, float(pt), float(hc)])
        elif name in oem_nodes:
            feats.append([0.0, 1.0, 0.0, 0.0, float(gsm_nodes[3].processing_time), float(gsm_nodes[3].local_holding_cost)])
        elif name in pdc_nodes:
            feats.append([0.0, 0.0, 1.0, 0.0, float(gsm_nodes[1].processing_time), float(gsm_nodes[1].local_holding_cost)])
        else:
            feats.append([0.0, 0.0, 0.0, 1.0, 0.0, float(gsm_nodes[2].local_holding_cost)])

    x = torch.tensor(feats, dtype=torch.float32)
    # Normalize continuous stage processing time (col 4)
    x[:, 4] = x[:, 4] / 21.0
    return node_to_idx, x


def _encode_edge_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Encodes point-in-time operational and heavy-machinery OEM features for each shipment edge.
    """
    ship_mode_map = {
        "Same Day": 0.0,
        "First Class": 1.0,
        "Second Class": 2.0,
        "Standard Class": 3.0,
    }
    ved_map = {"V": 1.0, "E": 0.5, "D": 0.1}

    ship_mode_num = df["Shipping Mode"].map(ship_mode_map).fillna(3.0).to_numpy(dtype=np.float32) / 3.0
    ved_num = df["ved"].map(ved_map).fillna(0.5).to_numpy(dtype=np.float32)
    unit_cost_log = np.log1p(df["unit_cost"].to_numpy(dtype=np.float32)) / 10.0
    vor_log = np.log1p(df["vor_penalty_day"].to_numpy(dtype=np.float32)) / 10.0
    tier1_pt = df["tier1_proc_days"].to_numpy(dtype=np.float32) / 10.0

    # Cyclical timestamp encoding if order_timestamp is present
    if "order_timestamp" in df.columns:
        ts = pd.to_datetime(df["order_timestamp"], errors="coerce")
        dow_sin = np.sin(2 * np.pi * ts.dt.dayofweek.fillna(0) / 7.0).to_numpy(dtype=np.float32)
        dow_cos = np.cos(2 * np.pi * ts.dt.dayofweek.fillna(0) / 7.0).to_numpy(dtype=np.float32)
    else:
        dow_sin = np.zeros(len(df), dtype=np.float32)
        dow_cos = np.ones(len(df), dtype=np.float32)

    # Optional scheduled days column if retained in parquet
    sched_candidates = [c for c in df.columns if "scheduled" in c.lower()]
    if sched_candidates:
        sched_days = df[sched_candidates[0]].to_numpy(dtype=np.float32) / 7.0
    else:
        sched_days = ship_mode_num

    edge_mat = np.column_stack([
        ship_mode_num,
        sched_days,
        ved_num,
        unit_cost_log,
        vor_log,
        tier1_pt,
        dow_sin,
        dow_cos,
    ])
    return torch.tensor(edge_mat, dtype=torch.float32)


def build_pyg_oem_graph(
    df: pd.DataFrame,
    node_to_idx: dict[str, int],
    x: torch.Tensor,
) -> Data:
    """
    Builds a PyG Data object combining structural upstream OEM supply links
    (Tier-1 -> OEM Plant -> Regional PDCs) with transactional shipment edges (PDC -> Dealer).
    """
    # 1. Upstream structural supply edges (T1 -> OEM Plant, OEM Plant -> PDCs)
    structural_pairs = set()
    for t1 in df["echelon_1_tier1"].unique():
        structural_pairs.add((node_to_idx[t1], node_to_idx["OEM_Central_Assembly_N3"]))
    for pdc in df["echelon_3_pdc"].unique():
        if pdc in node_to_idx:
            structural_pairs.add((node_to_idx["OEM_Central_Assembly_N3"], node_to_idx[pdc]))

    struct_src = [u for u, _ in sorted(structural_pairs)]
    struct_dst = [v for _, v in sorted(structural_pairs)]
    n_struct = len(struct_src)

    # 2. Downstream shipment edges (PDC -> Dealer)
    valid_mask = df["echelon_3_pdc"].isin(node_to_idx) & df["echelon_4_dealer"].isin(node_to_idx)
    df_valid = df.loc[valid_mask].reset_index(drop=True)

    ship_src = df_valid["echelon_3_pdc"].map(node_to_idx).to_numpy(dtype=np.int64)
    ship_dst = df_valid["echelon_4_dealer"].map(node_to_idx).to_numpy(dtype=np.int64)
    n_ship = len(df_valid)

    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([struct_src, ship_src]),
            np.concatenate([struct_dst, ship_dst]),
        ]),
        dtype=torch.long,
    )

    ship_edge_attr = _encode_edge_features(df_valid)
    struct_edge_attr = torch.zeros((n_struct, ship_edge_attr.shape[1]), dtype=torch.float32)
    edge_attr = torch.cat([struct_edge_attr, ship_edge_attr], dim=0)

    # Mask distinguishing evaluated shipment edges from structural upstream corridors
    is_shipment_edge = torch.cat([
        torch.zeros(n_struct, dtype=torch.bool),
        torch.ones(n_ship, dtype=torch.bool),
    ])

    # Targets & SCM economic vectors (padded with 0 on structural edges)
    y_reg = torch.cat([
        torch.zeros(n_struct, dtype=torch.float32),
        torch.tensor(df_valid["target_delay_days"].to_numpy(dtype=np.float32)),
    ])
    y_cls = torch.cat([
        torch.zeros(n_struct, dtype=torch.long),
        torch.tensor(df_valid["target_is_delayed"].to_numpy(dtype=np.int64)),
    ])
    vor_penalty = torch.cat([
        torch.zeros(n_struct, dtype=torch.float32),
        torch.tensor(df_valid["vor_penalty_day"].to_numpy(dtype=np.float32)),
    ])
    holding_cost = torch.cat([
        torch.zeros(n_struct, dtype=torch.float32),
        torch.tensor(df_valid["holding_cost_day"].to_numpy(dtype=np.float32)),
    ])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        is_shipment_edge=is_shipment_edge,
        y_reg=y_reg,
        y_cls=y_cls,
        vor_penalty=vor_penalty,
        holding_cost=holding_cost,
        num_nodes=x.shape[0],
    )


if __name__ == "__main__":
    proc_dir = Path("data/processed")
    df_train = pd.read_parquet(proc_dir / "train_oem.parquet")
    node_to_idx, x = _build_node_registry(df_train)

    print(f"[*] Built 4-Echelon OEM Node Registry: {len(node_to_idx)} nodes | Node features: {tuple(x.shape)}")

    for split in ["train", "val", "test"]:
        df_split = pd.read_parquet(proc_dir / f"{split}_oem.parquet")
        graph = build_pyg_oem_graph(df_split, node_to_idx, x)
        out_path = proc_dir / f"{split}_oem_graph.pt"
        torch.save(graph, out_path)
        n_ship = int(graph.is_shipment_edge.sum().item())
        n_struct = int((~graph.is_shipment_edge).sum().item())
        print(f"[✓] Saved {out_path} | Total Edges: {graph.edge_index.shape[1]:,} (Structural: {n_struct}, Shipments: {n_ship:,}) | Edge Feats: {graph.edge_attr.shape[1]}")