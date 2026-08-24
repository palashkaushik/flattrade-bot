"""Generate exact granular statistics for the 794.63 Calmar Ratio Undisputed Strategy."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.optimus_rejection_mechanics_lab_gpu import (
    df_sig, candles_3m, mechanics, t_sig_entries, t_sig_sl_dists, t_sig_tgt_dists,
    t_sig_custom_scores, t_sig_minutes, t_sig_dirs, t_sig_days, valid_fut, fut_h_m, fut_l_m, fut_c,
    LOT_SIZE, FEE_PER_TRADE, device, MAX_FUT, N_DAYS
)

# Mechanism 4: Two-Bar Structure Confirmation
mask_m4 = mechanics["4. Two-Bar Structure Confirmation (Break of Extreme)"]

t_mech = torch.tensor(mask_m4, device=device, dtype=torch.bool)
# Combined session 09:15-11:00 (555-660) + 13:30-15:00 (810-900)
session_mask = ((t_sig_minutes >= 555) & (t_sig_minutes <= 660)) | ((t_sig_minutes >= 810) & (t_sig_minutes <= 900))
# The 794.63 Calmar run used all valid two-bar confirmations in the dual session
active_mask = t_mech & session_mask

sl_dists = torch.maximum(t_sig_sl_dists * 0.3, torch.tensor(4.0, device=device)).unsqueeze(1)
tp_dists = torch.maximum(t_sig_tgt_dists * 1.5, torch.tensor(8.0, device=device)).unsqueeze(1)

dirs = t_sig_dirs.unsqueeze(1)
entries = t_sig_entries.unsqueeze(1)

is_long = (dirs == 1)
init_sl = torch.where(is_long, entries - sl_dists, entries + sl_dists)
init_tp = torch.where(is_long, entries + tp_dists, entries - tp_dists)

run_peaks_long = torch.cummax(torch.where(valid_fut, fut_h_m, entries), dim=1).values
gains_long = run_peaks_long - entries
trail_sl_long = run_peaks_long - 2.0
dyn_sl_long = torch.where(gains_long >= 6.0, torch.maximum(init_sl, trail_sl_long), init_sl)

run_peaks_short = torch.cummin(torch.where(valid_fut, fut_l_m, entries), dim=1).values
gains_short = entries - run_peaks_short
trail_sl_short = run_peaks_short + 2.0
dyn_sl_short = torch.where(gains_short >= 6.0, torch.minimum(init_sl, trail_sl_short), init_sl)

dyn_sl = torch.where(is_long, dyn_sl_long, dyn_sl_short)

hit_sl = torch.where(is_long, fut_l_m <= dyn_sl, fut_h_m >= dyn_sl)
hit_tp = torch.where(is_long, fut_h_m >= init_tp, fut_l_m <= init_tp)

BIG = 999999
sl_any = hit_sl.any(dim=1)
tp_any = hit_tp.any(dim=1)

sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

sl_exits = sl_any & (sl_first <= tp_first)
tp_exits = tp_any & (~sl_exits)

sl_idx_clamp = sl_first.clamp(max=MAX_FUT - 1).unsqueeze(1)
exit_sl_px = dyn_sl.gather(1, sl_idx_clamp).squeeze(1)
exit_tp_px = init_tp.squeeze(1)

last_valid_idx = (valid_fut.sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
exit_eod_px = fut_c.gather(1, last_valid_idx).squeeze(1)

exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, exit_eod_px))

pts_raw = torch.where(is_long.squeeze(1), exit_px - entries.squeeze(1), entries.squeeze(1) - exit_px)
pts = torch.where(active_mask, pts_raw, torch.zeros_like(pts_raw))
rs_net = torch.where(active_mask, pts * LOT_SIZE - FEE_PER_TRADE, torch.zeros_like(pts))

exit_reasons = np.where(sl_exits.cpu().numpy(), "SL", np.where(tp_exits.cpu().numpy(), "TP", "EOD"))

active_indices = np.where(active_mask.cpu().numpy())[0]

trade_records = []
for idx in active_indices:
    row = df_sig.iloc[idx]
    pnl_pts = float(pts[idx].cpu().numpy())
    net_val = float(rs_net[idx].cpu().numpy())
    sl_val = float(sl_dists[idx, 0].cpu().numpy())
    tp_val = float(tp_dists[idx, 0].cpu().numpy())
    t_str = str(row["time"])
    d_str = t_str[:10]
    trade_records.append({
        "time": t_str,
        "date": d_str,
        "year": d_str[:4],
        "month": d_str[:7],
        "direction": "LONG" if row["direction"] == 1 else "SHORT",
        "entry": float(row["entry"]),
        "exit": float(exit_px[idx].cpu().numpy()),
        "sl_pts": sl_val,
        "tp_pts": tp_val,
        "pnl_pts": pnl_pts,
        "net_rs": net_val,
        "reason": exit_reasons[idx],
        "level": str(row["level"]),
    })

df_t = pd.DataFrame(trade_records)

# Daily P&L for Daily Win Rate and Drawdown
day_pnls = {}
for d, grp in df_t.groupby("date"):
    day_pnls[d] = grp["net_rs"].sum()

green_days = sum(1 for v in day_pnls.values() if v > 0)
red_days = sum(1 for v in day_pnls.values() if v < 0)
active_days = len(day_pnls)
daily_win_rate = (green_days / active_days * 100) if active_days > 0 else 0

# Yearly Breakdown
yearly_stats = {}
for yr, grp in df_t.groupby("year"):
    wins = grp[grp["net_rs"] > 0]
    losses = grp[grp["net_rs"] <= 0]
    gross_w = wins["net_rs"].sum()
    gross_l = abs(losses["net_rs"].sum())
    pf = gross_w / gross_l if gross_l > 0 else 99.0
    yearly_stats[yr] = {
        "trades": len(grp),
        "days": grp["date"].nunique(),
        "trades_per_day": round(len(grp) / grp["date"].nunique(), 2),
        "win_rate": round(len(wins) / len(grp) * 100, 2),
        "net_points": round(grp["pnl_pts"].sum(), 2),
        "net_rs": round(grp["net_rs"].sum(), 2),
        "pf": round(pf, 3),
        "avg_win_pts": round(wins["pnl_pts"].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss_pts": round(losses["pnl_pts"].mean(), 2) if len(losses) > 0 else 0,
    }

# Monthly Stats
monthly_stats = {}
green_months = 0
for mo, grp in df_t.groupby("month"):
    pnl = grp["net_rs"].sum()
    if pnl > 0:
        green_months += 1
    monthly_stats[mo] = pnl

total_months = len(monthly_stats)
monthly_consistency = (green_months / total_months * 100) if total_months > 0 else 0

cum_eq = np.cumsum(list(day_pnls.values()))
peaks = np.maximum.accumulate(cum_eq)
dds = peaks - cum_eq
max_dd = float(np.max(dds))

summary_794 = {
    "title": "UNDISPUTED REJECTION CHAMPION (794.63 CALMAR RATIO)",
    "total_trades": len(df_t),
    "total_traded_days": active_days,
    "avg_trades_per_day": round(len(df_t) / active_days, 2),
    "daily_win_rate_pct": round(daily_win_rate, 2),
    "green_days": green_days,
    "red_days": red_days,
    "trade_win_rate_pct": round(len(df_t[df_t["net_rs"] > 0]) / len(df_t) * 100, 2),
    "total_points": round(df_t["pnl_pts"].sum(), 2),
    "total_realized_rs": round(df_t["net_rs"].sum(), 2),
    "profit_factor": round(df_t[df_t["net_rs"] > 0]["net_rs"].sum() / abs(df_t[df_t["net_rs"] <= 0]["net_rs"].sum()), 3),
    "max_drawdown_rs": round(max_dd, 2),
    "calmar_ratio": round(df_t["net_rs"].sum() / max_dd, 2),
    "avg_sl_pts": round(df_t["sl_pts"].mean(), 2),
    "min_sl_pts": round(df_t["sl_pts"].min(), 2),
    "max_sl_pts": round(df_t["sl_pts"].max(), 2),
    "avg_tp_pts": round(df_t["tp_pts"].mean(), 2),
    "min_tp_pts": round(df_t["tp_pts"].min(), 2),
    "max_tp_pts": round(df_t["tp_pts"].max(), 2),
    "avg_win_pts": round(df_t[df_t["net_rs"] > 0]["pnl_pts"].mean(), 2),
    "avg_loss_pts": round(df_t[df_t["net_rs"] <= 0]["pnl_pts"].mean(), 2),
    "avg_win_rs": round(df_t[df_t["net_rs"] > 0]["net_rs"].mean(), 2),
    "avg_loss_rs": round(df_t[df_t["net_rs"] <= 0]["net_rs"].mean(), 2),
    "payoff_ratio": round(df_t[df_t["net_rs"] > 0]["pnl_pts"].mean() / abs(df_t[df_t["net_rs"] <= 0]["pnl_pts"].mean()), 2),
    "monthly_consistency_pct": round(monthly_consistency, 1),
    "green_months": green_months,
    "total_months": total_months,
    "yearly": yearly_stats,
}

print(json.dumps(summary_794, indent=2))
Path(ROOT / "artifacts" / "f6_hybrid" / "undisputed_794_calmar_stats.json").write_text(json.dumps(summary_794, indent=2), encoding="utf-8")
