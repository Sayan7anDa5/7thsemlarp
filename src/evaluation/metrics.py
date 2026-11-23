import numpy as np
import pandas as pd


def evaluate_scm_economics(
    actual_delay_days: np.ndarray,
    pred_delay_days: np.ndarray,
    scheduled_lt_days: np.ndarray,
    vor_penalty_day: np.ndarray,
    holding_cost_day: np.ndarray,
    ved_flags: np.ndarray | None = None,
) -> dict:
    """
    Compares Static ERP Lead Time vs. Model-Predicted Dynamic Lead Time
    on Asymmetric Vehicle-Off-Road (VOR) Downtime Cost and Holding Cost.
    """
    actual_lt = scheduled_lt_days + actual_delay_days
    static_lt = scheduled_lt_days
    dynamic_lt = scheduled_lt_days + pred_delay_days

    static_err = actual_lt - static_lt
    dynamic_err = actual_lt - dynamic_lt

    def _cost_breakdown(err):
        vor = np.maximum(0.0, err) * vor_penalty_day
        hold = np.maximum(0.0, -err) * holding_cost_day
        return float((vor + hold).sum()), float(vor.sum()), float(hold.sum())

    static_tot, static_vor, _ = _cost_breakdown(static_err)
    dyn_tot, dyn_vor, _ = _cost_breakdown(dynamic_err)

    metrics = {
        "static_erp_total_cost_usd": round(static_tot, 2),
        "model_dynamic_total_cost_usd": round(dyn_tot, 2),
        "vor_cost_saved_usd": round(static_vor - dyn_vor, 2),
        "total_cost_reduction_pct": round(((static_tot - dyn_tot) / max(static_tot, 1e-6)) * 100.0, 2),
    }

    if ved_flags is not None:
        vital_mask = ved_flags == "V"
        if np.any(vital_mask):
            metrics["vital_parts_mae_days"] = round(
                float(np.abs(actual_lt[vital_mask] - dynamic_lt[vital_mask]).mean()), 3
            )

    return metrics


def compute_dynamic_safety_stock(
    avg_daily_demand: float,
    std_daily_demand: float,
    pred_lead_time_days: float,
    std_lead_time_days: float,
    z_score: float = 2.33,
) -> dict:
    """
    Computes Dynamic Safety Stock (SS) and Reorder Point (ROP) using GNN-predicted lead time.
    """
    ss = z_score * np.sqrt(
        (pred_lead_time_days * (std_daily_demand ** 2))
        + ((avg_daily_demand ** 2) * (std_lead_time_days ** 2))
    )
    rop = (avg_daily_demand * pred_lead_time_days) + ss
    return {"safety_stock_units": round(float(ss), 2), "reorder_point_units": round(float(rop), 2)}


def evaluate_resilience_ttr_tts(
    pdc_inventory_units: float,
    avg_daily_demand: float,
    shock_recovery_days_ttr: float,
) -> dict:
    """
    Evaluates Supply Chain Resilience: Critical vulnerability occurs whenever TTR > TTS.
    """
    tts = pdc_inventory_units / max(avg_daily_demand, 1e-6)
    stockout_gap = max(0.0, shock_recovery_days_ttr - tts)
    return {
        "time_to_survive_days": round(float(tts), 2),
        "time_to_recover_days": round(float(shock_recovery_days_ttr), 2),
        "is_resilient": bool(shock_recovery_days_ttr <= tts),
        "projected_vor_exposure_days": round(float(stockout_gap), 2),
    }