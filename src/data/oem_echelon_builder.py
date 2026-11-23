from pathlib import Path
import numpy as np
import pandas as pd
from stockpyl.instances import load_instance

# Heavy Machinery OEM Spare Parts Catalog (ABC-VED Criticality + Asymmetric VOR Economics)
OEM_PART_CATALOG = [
    {"part_family": "Hydraulic Main Pump",   "abc": "A", "ved": "V", "unit_cost": 4800.0, "vor_penalty_day": 2500.0, "gsm_supplier_node": 5},
    {"part_family": "Transmission Assembly", "abc": "A", "ved": "V", "unit_cost": 9200.0, "vor_penalty_day": 3200.0, "gsm_supplier_node": 5},
    {"part_family": "Final Drive Axle",      "abc": "A", "ved": "V", "unit_cost": 6500.0, "vor_penalty_day": 2800.0, "gsm_supplier_node": 5},
    {"part_family": "Turbocharger Unit",     "abc": "B", "ved": "V", "unit_cost": 1850.0, "vor_penalty_day": 1800.0, "gsm_supplier_node": 5},
    {"part_family": "Fuel Injector Rail",    "abc": "B", "ved": "E", "unit_cost": 950.0,  "vor_penalty_day": 900.0,  "gsm_supplier_node": 4},
    {"part_family": "Brake Caliper Kit",     "abc": "B", "ved": "E", "unit_cost": 620.0,  "vor_penalty_day": 750.0,  "gsm_supplier_node": 4},
    {"part_family": "Heavy-Duty Air Filter", "abc": "C", "ved": "E", "unit_cost": 140.0,  "vor_penalty_day": 250.0,  "gsm_supplier_node": 4},
    {"part_family": "Cab Trim & Mirror Kit", "abc": "C", "ved": "D", "unit_cost": 210.0,  "vor_penalty_day": 40.0,   "gsm_supplier_node": 4},
]


def extract_gsm_oem_parameters(instance_name: str = "problem_6_9") -> dict:
    """
    Extracts upstream stage processing times and holding cost ratios from Snyder & Shen's
    Tree Guaranteed Service Model (problem_6_9 in stockpyl).
    """
    net = load_instance(instance_name)
    node_map = {n.index: n for n in net.nodes}
    return {
        "tier1_heavy_pt": node_map[5].processing_time,      # 8 days (Precision Hydraulics/Powertrain)
        "tier1_std_pt": node_map[4].processing_time,        # 3 days (Standard Chassis/Filtration)
        "oem_assembly_pt": node_map[3].processing_time,     # 21 days (OEM Heavy Assembly & Central Depot)
        "pdc_primary_pt": node_map[1].processing_time,      # 7 days (Primary Regional PDC)
        "pdc_secondary_pt": node_map[2].processing_time,    # 7 days (Secondary Regional PDC)
        "pdc_primary_hc_rate": node_map[1].local_holding_cost,
        "pdc_secondary_hc_rate": node_map[2].local_holding_cost,
    }


def build_oem_echelon_dataset(
    df_clean: pd.DataFrame,
    origin_col: str = "Order Region",
    dest_col: str = "Customer State",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Constructs the 4-Echelon Heavy Machinery OEM Dataset:
    Echelon 1: Tier-1 Component Supplier (GSM Nodes 4 & 5)
    Echelon 2: OEM Manufacturing & Central Parts Depot (GSM Node 3, PT=21d)
    Echelon 3: Regional Parts Distribution Centers / PDCs (GSM Nodes 1 & 2 -> DataCo Hubs)
    Echelon 4: Dealerships & Heavy Fleet Customers (DataCo Destinations)
    """
    rng = np.random.default_rng(seed)
    df = df_clean.copy()
    gsm = extract_gsm_oem_parameters("problem_6_9")

    part_probs = [0.07, 0.05, 0.06, 0.12, 0.18, 0.18, 0.22, 0.12]
    chosen_idx = rng.choice(len(OEM_PART_CATALOG), size=len(df), p=part_probs)
    parts_df = pd.DataFrame([OEM_PART_CATALOG[i] for i in chosen_idx], index=df.index)
    df = pd.concat([df, parts_df], axis=1)

    df["abc_ved_class"] = df["abc"] + df["ved"]

    # 4-Echelon Topology Mapping
    df["echelon_1_tier1"] = np.where(
        df["gsm_supplier_node"] == 5,
        "T1_Powertrain_Hydraulics_N5",
        "T1_Chassis_Filtration_N4",
    )
    df["echelon_2_oem_plant"] = "OEM_Central_Assembly_N3"
    df["echelon_3_pdc"] = "PDC_" + df[origin_col].astype(str)
    df["echelon_4_dealer"] = "Dealer_" + df[dest_col].astype(str)

    # Upstream Manufacturing & Staging Lead Times (from stockpyl GSM)
    df["tier1_proc_days"] = np.where(
        df["gsm_supplier_node"] == 5, gsm["tier1_heavy_pt"], gsm["tier1_std_pt"]
    )
    df["oem_assembly_days"] = gsm["oem_assembly_pt"]
    df["pdc_staging_days"] = gsm["pdc_primary_pt"]

    # Daily Holding Cost (scaled by GSM holding cost rate) & Target Cycle Service Level (CSL)
    hc_rate = np.where(df["ved"] == "V", gsm["pdc_primary_hc_rate"], gsm["pdc_secondary_hc_rate"])
    df["holding_cost_day"] = (df["unit_cost"] * hc_rate) / 365.0

    z_map = {"V": 2.33, "E": 1.645, "D": 1.04}
    csl_map = {"V": 0.99, "E": 0.95, "D": 0.85}
    df["z_score"] = df["ved"].map(z_map)
    df["target_csl"] = df["ved"].map(csl_map)

    return df

if __name__ == "__main__":
    proc_dir = Path("data/processed")
    splits = ["train", "val", "test"]

    for i, split in enumerate(splits):
        in_path = proc_dir / f"{split}.parquet"
        df_split = pd.read_parquet(in_path)

        if i == 0:
            print(f"Processed columns ({len(df_split.columns)}): {df_split.columns.tolist()}")

        cols_lower = {c.lower().replace(" ", "_"): c for c in df_split.columns}
        origin_col = cols_lower.get("order_region", cols_lower.get("origin", df_split.columns[0]))
        dest_col = cols_lower.get("customer_state", cols_lower.get("destination", df_split.columns[1]))

        # Offset seed per partition so part sampling stays independent across chronological splits
        df_oem = build_oem_echelon_dataset(
            df_split, origin_col=origin_col, dest_col=dest_col, seed=42 + i
        )
        out_path = proc_dir / f"{split}_oem.parquet"
        df_oem.to_parquet(out_path, index=False)
        print(f"[✓] Saved {out_path} ({len(df_oem):,} rows) | Origin='{origin_col}', Dest='{dest_col}'")