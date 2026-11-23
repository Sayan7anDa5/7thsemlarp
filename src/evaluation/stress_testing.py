import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from src.models.temporal_gnn import TemporalAttentionPredictor
from src.evaluation.scm_metrics import compute_dynamic_safety_stock, evaluate_resilience_ttr_tts


def run_stress_test():
    print("[*] Loading 4-Echelon OEM Test Graph & Trained GATv2 Regression Checkpoint...")
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

    ship_mask = test_graph.is_shipment_edge
    ship_src = test_graph.edge_index[0, ship_mask].cpu().numpy()
    ship_dst = test_graph.edge_index[1, ship_mask].cpu().numpy()
    sched_days = test_graph.edge_attr[ship_mask, 1].cpu().numpy() * 7.0
    vor_pen = test_graph.vor_penalty[ship_mask].cpu().numpy()
    ved_num = test_graph.edge_attr[ship_mask, 2].cpu().numpy()

    with torch.no_grad():
        base_preds = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)[ship_mask].cpu().numpy()

    # =========================================================================
    # SCENARIO 1: REGIONAL PDC CAPACITY BOTTLENECK (IN-DISTRIBUTION SLA SHOCK)
    # =========================================================================
    busiest_pdc = int(np.bincount(ship_src).argmax())
    pdc_affected = ship_src == busiest_pdc
    num_affected = int(pdc_affected.sum())

    print("\n" + "=" * 64)
    print("SCENARIO 1: REGIONAL PDC CAPACITY BOTTLENECK (SLA-LOCKED)")
    print("=" * 64)
    print(f"[+] Critical Bottleneck PDC : Node {busiest_pdc}")
    print(f"[+] Exposed Customer Orders : {num_affected:,} shipments")

    # Keep all features strictly within training bounds [0.0, 1.0]:
    # Col 0 -> 1/3 (tight First-Class SLA queue saturation)
    # Col 2 -> 1.0 (Vital-part emergency expedite surge)
    shocked_attr = test_graph.edge_attr.clone()
    full_pdc_mask = test_graph.edge_index[0] == busiest_pdc
    shocked_attr[full_pdc_mask, 0] = 1.0 / 3.0
    shocked_attr[full_pdc_mask, 2] = 1.0

    with torch.no_grad():
        pdc_shock_preds = model(test_graph.x, test_graph.edge_index, shocked_attr)[ship_mask].cpu().numpy()

    delay_delta = pdc_shock_preds - base_preds
    direct_pos_delta = np.maximum(0.0, delay_delta[pdc_affected])
    spillover_delta = delay_delta[~pdc_affected]
    added_vor_cost = float((direct_pos_delta * vor_pen[pdc_affected]).sum())

    print("\n--- Disruption & Asymmetric Financial Impact ---")
    print(f"  Mean Positive Delay Spike (PDC {busiest_pdc}): +{direct_pos_delta.mean():.3f} days (Max: +{direct_pos_delta.max():.3f} days)")
    print(f"  Network Spillover (Other PDCs)    : {spillover_delta.mean():+.4f} days")
    print(f"  Incremental VOR Downtime Risk     : ${added_vor_cost:,.2f}")

    # Dynamic Safety Stock & ROP Recalculation for Vital Parts (AV/BV)
    vital_pdc = pdc_affected & (ved_num > 0.8)
    base_lt_vital = float((sched_days[vital_pdc] + np.maximum(0.0, base_preds[vital_pdc])).mean())
    shock_lt_vital = base_lt_vital + float(np.maximum(0.0, delay_delta[vital_pdc]).mean())
    ss_base = compute_dynamic_safety_stock(
        avg_daily_demand=12.0, std_daily_demand=3.5, pred_lead_time_days=base_lt_vital, std_lead_time_days=1.1, z_score=2.33
    )
    ss_shock = compute_dynamic_safety_stock(
        avg_daily_demand=12.0, std_daily_demand=3.5, pred_lead_time_days=shock_lt_vital, std_lead_time_days=1.8, z_score=2.33
    )
    print(
        f"  Vital Part Dynamic ROP Shift      : {ss_base['reorder_point_units']} -> "
        f"{ss_shock['reorder_point_units']} units (+{ss_shock['reorder_point_units'] - ss_base['reorder_point_units']:.1f} buffer units)"
    )

    # =========================================================================
    # SCENARIO 2: UPSTREAM OEM PLANT BOTTLENECK (+6D GSM SERIAL COUPLING)
    # =========================================================================
    print("\n" + "=" * 64)
    print("SCENARIO 2: UPSTREAM OEM PLANT & TIER-1 SUPPLY SHOCK (+6 DAYS)")
    print("=" * 64)

    # Combine +6.0d serial OEM assembly delay (GSM Node 3) with GNN-predicted transit lead time
    oem_serial_shock_days = 6.0
    total_recovery_lt = oem_serial_shock_days + sched_days + np.maximum(0.0, base_preds)

    unique_dealers = np.unique(ship_dst)
    breached_dealers = 0
    total_vor_gap_days = 0.0
    for d_node in unique_dealers:
        d_mask = ship_dst == d_node
        ttr_days = float(np.percentile(total_recovery_lt[d_mask], 90))
        res = evaluate_resilience_ttr_tts(pdc_inventory_units=90.0, avg_daily_demand=10.0, shock_recovery_days_ttr=ttr_days)
        if not res["is_resilient"]:
            breached_dealers += 1
            total_vor_gap_days += res["projected_vor_exposure_days"]

    print(f"  Mean End-to-End Recovery (TTR)    : {total_recovery_lt.mean():.2f} days (90th pct: {np.percentile(total_recovery_lt, 90):.2f} days)")
    print(f"  Dealerships Breaching TTR > TTS   : {breached_dealers} / {len(unique_dealers)} dealers (Forward Buffer TTS = 9.0 days)")
    print(f"  Cumulative Tail VOR Exposure      : {total_vor_gap_days:.1f} dealer-days")
    print("\n[✓] OEM Multi-Echelon Resilience Stress Test Complete.")


if __name__ == "__main__":
    run_stress_test()
