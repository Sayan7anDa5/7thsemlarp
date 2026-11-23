import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def build_pyg_data(parquet_path: str, origin_map=None, dest_map=None):
    df = pd.read_parquet(parquet_path)
    
    # 1. Align Node Entities
    origins = df['Order Region'].astype(str).values
    dests = df['Customer State'].astype(str).values
    
    if origin_map is None:
        unique_origins = np.unique(origins)
        origin_map = {name: idx for idx, name in enumerate(unique_origins)}
        
    if dest_map is None:
        unique_dests = np.unique(dests)
        # Shift dest indices so node IDs do not overlap with origin IDs
        num_origins = len(origin_map)
        dest_map = {name: idx + num_origins for idx, name in enumerate(unique_dests)}
        
    # Map entities to contiguous tensor indices
    src_nodes = [origin_map.get(o, 0) for o in origins]
    dst_nodes = [dest_map.get(d, len(origin_map)) for d in dests]
    
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    # 2. Extract Edge Features (Leakage-Free Predictors)
    # Temporal attributes
    df['order_hour'] = df['order_timestamp'].dt.hour
    df['order_dayofweek'] = df['order_timestamp'].dt.dayofweek
    df['order_month'] = df['order_timestamp'].dt.month
    
    # Encode categorical columns numerically
    cat_cols = ['Shipping Mode', 'Category Name', 'Department Name', 'Customer Segment']
    for c in cat_cols:
        df[c] = df[c].astype('category').cat.codes
        
    numeric_cols = [
        'Days for shipment (scheduled)',
        'order_hour',
        'order_dayofweek',
        'order_month',
        'Order Item Quantity',
        'Order Item Product Price',
        'Order Item Discount Rate'
    ] + cat_cols
    
    edge_attr = torch.tensor(df[numeric_cols].values, dtype=torch.float)
    
    # 3. Targets
    y_clf = torch.tensor(df['target_is_delayed'].values, dtype=torch.float)
    y_reg = torch.tensor(df['target_delay_days'].values, dtype=torch.float)
    
    # 4. Total node count
    total_nodes = len(origin_map) + len(dest_map)
    
    # Simple one-hot / initial identity embedding matrix for nodes
    x = torch.eye(total_nodes, dtype=torch.float)
    
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_clf=y_clf,
        y_reg=y_reg,
        num_nodes=total_nodes
    )
    
    return graph_data, origin_map, dest_map

def generate_graph_partitions():
    print("[*] Building PyG graph structures from chronological partitions...")
    
    # Fit node maps on train to prevent inductive identity leakage
    train_graph, orig_map, dst_map = build_pyg_data("data/processed/train.parquet")
    val_graph, _, _ = build_pyg_data("data/processed/val.parquet", orig_map, dst_map)
    test_graph, _, _ = build_pyg_data("data/processed/test.parquet", orig_map, dst_map)
    
    print(f"[+] Total Network Nodes: {train_graph.num_nodes} (Origins: {len(orig_map)}, Destinations: {len(dst_map)})")
    print(f"[+] Train Edges (Shipments): {train_graph.edge_index.size(1)}")
    print(f"[+] Val Edges (Shipments):   {val_graph.edge_index.size(1)}")
    print(f"[+] Test Edges (Shipments):  {test_graph.edge_index.size(1)}")
    
    torch.save(train_graph, "data/processed/train_graph.pt")
    torch.save(val_graph, "data/processed/val_graph.pt")
    torch.save(test_graph, "data/processed/test_graph.pt")
    print("[✓] Saved graph objects (.pt) to data/processed/")

if __name__ == "__main__":
    generate_graph_partitions()