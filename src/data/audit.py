import pandas as pd
import numpy as np

def run_audit(path: str):
    df = pd.read_csv(path, encoding='latin-1')

    # Temporal anchor (order placement time t_0)
    df['order_timestamp'] = pd.to_datetime(df['order date (DateOrders)'])
    
    # Target formulation
    df['target_delay_days'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
    df['target_is_delayed'] = (df['target_delay_days'] > 0).astype(int)

    # Strict point-in-time filter
    drop_cols = [
        'Late_delivery_risk', 
        'Days for shipping (real)', 
        'Delivery Status', 
        'Order Status',
        'shipping date (DateOrders)',
        'Product Description',
        'Product Image',
        'Customer Email', 
        'Customer Password', 
        'Customer Fname', 
        'Customer Lname',
        'Customer Street',
        'Order Zipcode'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Chronological partition
    df = df.sort_values(by='order_timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test

if __name__ == "__main__":
    train, val, test = run_audit("data/raw/DataCoSupplyChainDataset.csv")
    train.to_parquet("data/processed/train.parquet", index=False)
    val.to_parquet("data/processed/val.parquet", index=False)
    test.to_parquet("data/processed/test.parquet", index=False)
    print("[+] Audit complete. Partitioned parquet files written to data/processed/")