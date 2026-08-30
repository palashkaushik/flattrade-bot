"""
GPU-vectorized simulation for the Last Hope Nifty50 options strategy.

Ports run_7y_v4_master.py's CPU bar-loop onto the GPU following the GPU
backtest runbook (per-minute t-loop with ALL days vectorized as GPU tensors).
This keeps the GPU busy during simulation (the previous bottleneck: GPU was
only used for ~0.5s of indicator compute, then idle during the CPU loop).

Builds on the state already constructed in run_7y_v4_master (imported, not run).
"""
import sys, time
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
import numpy as np
import torch

torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda')

import run_7y_v4_master as M  # noqa: E402  (importing builds all state)

D = M.D
T1 = M.T1
LOT = M.LOT
FEE = M.FEE
ARM_S1 = M.ARM_S1
M6_S4 = M.M6_S4
M6_S1 = M.M6_S1
RSI_CE_HI = M.RSI_CE_HI
RSI_PE_LO = M.RSI_PE_LO
trading_days = M.trading_days


def _g(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)


def _gb(x):
    return torch.tensor(np.asarray(x), dtype=torch.bool, device=DEVICE)


# ---- Static tensors (built once at import) -------------------------------
ce_c, ce_h, ce_l = _g(M.ce_c), _g(M.ce_h), _g(M.ce_l)
pe_c, pe_h, pe_l = _g(M.pe_c), _g(M.pe_h), _g(M.pe_l)
ce_s1, ce_s3, ce_s4 = _g(M.ce_s1), _g(M.ce_s3), _g(M.ce_s4)
pe_s1, pe_s3, pe_s4 = _g(M.pe_s1), _g(M.pe_s3), _g(M.pe_s4)
ce_super_full, ce_m6_full = _gb(M.ce_super_full), _gb(M.ce_m6_full)
pe_super_full, pe_m6_full = _gb(M.pe_super_full), _gb(M.pe_m6_full)
ce_rev_on, pe_rev_on = _gb(M.ce_rev_on), _gb(M.pe_rev_on)
ce_atr, pe_atr = _g(M.ce_atr), _g(M.pe_atr)
ce_ema20, ce_ema200, ce_vwap = _g(M.ce_ema20), _g(M.ce_ema200), _g(M.ce_vwap)
pe_ema20, pe_ema200, pe_vwap = _g(M.pe_ema20), _g(M.pe_ema200), _g(M.pe_vwap)

# Supertrend (option chart) for the "price between VWAP and Supertrend" zone filter.
_ST_PERIOD = 14
_ST_MULT = 3.0
def _gpu_supertrend(high, low, close, period=_ST_PERIOD, mult=_ST_MULT):
    prev = torch.empty_like(close); prev[:, 0] = close[:, 0]; prev[:, 1:] = close[:, :-1]
    tr = torch.maximum(high - low, torch.maximum(torch.abs(high - prev), torch.abs(low - prev)))
    atr = torch.empty_like(tr); alpha = 2.0 / (period + 1)
    atr[:, 0] = tr[:, 0]
    for i in range(1, T1):
        atr[:, i] = alpha * tr[:, i] + (1 - alpha) * atr[:, i - 1]
    hl2 = (high + low) / 2.0
    up = hl2 + mult * atr
    dn = hl2 - mult * atr
    st = torch.empty_like(close)
    trend = close[:, 0] > up[:, 0]
    st[:, 0] = torch.where(trend, dn[:, 0], up[:, 0])
    for t in range(1, T1):
        cl = close[:, t]; u = up[:, t]; d = dn[:, t]; pst = st[:, t - 1]; ptr = trend
        final_up = torch.maximum(u, pst)
        final_dn = torch.minimum(d, pst)
        new_trend = (ptr & (cl > final_up)) | (~ptr & (cl >= final_dn))
        st[:, t] = torch.where(new_trend, final_up, final_dn)
        trend = new_trend
    return st

pe_st = _gpu_supertrend(pe_h, pe_l, pe_c)
ce_st = _gpu_supertrend(ce_h, ce_l, ce_c)

# Combined-TF low/close per side (CPU checks SR bounce on main chart AND TF charts)
pe_tf_lo = [_g(c['lo']) for c in M.pe_tf]
pe_tf_cl = [_g(c['cl']) for c in M.pe_tf]
ce_tf_lo = [_g(c['lo']) for c in M.ce_tf]
ce_tf_cl = [_g(c['cl']) for c in M.ce_tf]

# Per-(day,minute) lookups -> tensors
elder_state = torch.zeros((D, T1), dtype=torch.int8, device=DEVICE)   # -1 red, 0 blue, 1 green
rsi_mat = torch.zeros((D, T1), dtype=torch.float32, device=DEVICE)
bias_bull = torch.zeros((D, T1), dtype=torch.bool, device=DEVICE)
bias_bear = torch.zeros((D, T1), dtype=torch.bool, device=DEVICE)
bias_lr_grid = torch.zeros((D, T1), dtype=torch.int8, device=DEVICE)
bias_ut_grid = torch.zeros((D, T1), dtype=torch.int8, device=DEVICE)

SS = M.SESSION_START  # 555 = 09:15


def _ts(day, m):
    hh = (SS + m) // 60
    mm = (SS + m) % 60
    return f"{day} {hh:02d}:{mm:02d}:00"


for d, day in enumerate(trading_days):
    for m in range(T1):
        ts = _ts(day, m)
        ec = M.elder_lookup.get(ts)
        if ec == 'green':
            elder_state[d, m] = 1
        elif ec == 'red':
            elder_state[d, m] = -1
        rsi_mat[d, m] = M.rsi_lookup.get(ts, 50.0)
        if ts in M.bias_lookup:
            bu, be = M.bias_lookup[ts]
            bias_bull[d, m] = bool(bu)
            bias_bear[d, m] = bool(be)
        if ts in M.bias_lr_lookup:
            bias_lr_grid[d, m] = M.bias_lr_lookup[ts]
        if ts in M.bias_ut_lookup:
            bias_ut_grid[d, m] = M.bias_ut_lookup[ts]

# Option SR day-level levels per side (list of floats per day)
pe_sr_levels = []
ce_sr_levels = []
max_sr = 1
for day in trading_days:
    pl = [lvl for (_, lvl) in M.pe_option_sr.get(day, [])]
    cl = [lvl for (_, lvl) in M.ce_option_sr.get(day, [])]
    pe_sr_levels.append(pl)
    ce_sr_levels.append(cl)
    max_sr = max(max_sr, len(pl), len(cl))

PAD = max_sr + 3  # + ema20, ema200, vwap per bar


def _build_bounce(low, close, ema20, ema200, vwap, sr_levels, tf_lo_list, tf_cl_list, buf=1.0):
    """Return (D,T) bool: SR bounce at each (day,minute).

    Mirrors the CPU engine: SR bounce on the MAIN chart OR any COMBINED-TF
    (2m/3m/5m) low/close. SR levels at (d,t) = day-level option levels +
    [ema20[d,t], ema200[d,t], vwap[d,t]]; padded cells are +inf so they never
    satisfy the (low<=lv+buf AND close>=lv-0.5) test. `buf` is the touch
    buffer (how many option points the low may sit below the level to count
    as a touch).
    """
    sr = torch.full((D, T1, PAD), float('inf'), device=DEVICE)
    for d in range(D):
        levels = sr_levels[d]
        n = len(levels)
        if n:
            sr[d, :, :n] = torch.tensor(levels, dtype=torch.float32, device=DEVICE)
    sr[:, :, max_sr] = ema20
    sr[:, :, max_sr + 1] = ema200
    sr[:, :, max_sr + 2] = vwap
    lo = low.unsqueeze(-1)        # (D,T,1)
    cl = close.unsqueeze(-1)      # (D,T,1)
    cond = (lo <= sr + buf) & (cl >= sr - 0.5)
    b = cond.any(dim=-1)
    for tlo, tcl in zip(tf_lo_list, tf_cl_list):
        tlo2 = tlo.unsqueeze(-1)
        tcl2 = tcl.unsqueeze(-1)
        tb = ((tlo2 <= sr + buf) & (tcl2 >= sr - 0.5)).any(dim=-1)
        b |= tb
    return b


bounce_pe = _build_bounce(pe_l, pe_c, pe_ema20, pe_ema200, pe_vwap, pe_sr_levels, pe_tf_lo, pe_tf_cl)
bounce_ce = _build_bounce(ce_l, ce_c, ce_ema20, ce_ema200, ce_vwap, ce_sr_levels, ce_tf_lo, ce_tf_cl)

# Precompute SR-bounce for each swept touch-buffer value so a per-config
# touch_buffer can be selected cheaply inside the sim core (replaces the
# previously hardcoded +1.0 buffer in _build_bounce).
TOUCH_BUFFERS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
_bounce_pe_list = [_build_bounce(pe_l, pe_c, pe_ema20, pe_ema200, pe_vwap,
                                  pe_sr_levels, pe_tf_lo, pe_tf_cl, buf=b) for b in TOUCH_BUFFERS]
_bounce_ce_list = [_build_bounce(ce_l, ce_c, ce_ema20, ce_ema200, ce_vwap,
                                  ce_sr_levels, ce_tf_lo, ce_tf_cl, buf=b) for b in TOUCH_BUFFERS]
bounce_pe_stack = torch.stack(_bounce_pe_list, 0)   # (n_buf, D, T1)
bounce_ce_stack = torch.stack(_bounce_ce_list, 0)




def _gpu_atr(high, low, close, period=14):
    """Exact port of run_7y_v4_master.compute_atr (EMA, alpha=2/(period+1))."""
    prev = torch.empty_like(close)
    prev[:, 0] = close[:, 0]
    prev[:, 1:] = close[:, :-1]
    tr = torch.maximum(torch.maximum(high - low, torch.abs(high - prev)),
                       torch.abs(low - prev))
    atr = torch.empty_like(tr)
    alpha = 2.0 / (period + 1)
    atr[:, 0] = tr[:, 0]
    for t in range(1, tr.shape[1]):
        atr[:, t] = atr[:, t - 1] * (1.0 - alpha) + tr[:, t] * alpha
    return atr


_ATR_CACHE = {}
def _get_atr(high, low, close, period):
    """ATR is param-independent; cache it across calls (huge per-call saving)."""
    key = (id(high), int(period))
    if key not in _ATR_CACHE:
        _ATR_CACHE[key] = _gpu_atr(high, low, close, period)
    return _ATR_CACHE[key]


@torch.inference_mode()
def _eager_sim_core(params_list):
    """Run a BATCH of configs on the GPU in a single pass (eager; parity oracle)."""
    B = len(params_list)
    sl_b = torch.tensor([float(p.get('sl', M.SL_PTS)) for p in params_list], device=DEVICE)
    tp_b = torch.tensor([float(p.get('tp', M.TP_PTS)) for p in params_list], device=DEVICE)
    atr_mult_b = torch.tensor([float(p.get('atr_mult', 1.0)) for p in params_list], device=DEVICE)
    cap_b = torch.tensor([float(p.get('cap', 0.0)) for p in params_list], device=DEVICE)
    arm_b = torch.tensor([int(p.get('arm_window', M.ARM_WINDOW)) for p in params_list], device=DEVICE)
    atr_sl_b = torch.tensor([bool(p.get('atr_sl', False)) for p in params_list], dtype=torch.bool, device=DEVICE)
    ue_b = torch.tensor([bool(p.get('use_elder', M.USE_ELDER)) for p in params_list], dtype=torch.bool, device=DEVICE)
    ur_b = torch.tensor([bool(p.get('use_rsi', M.USE_RSI)) for p in params_list], dtype=torch.bool, device=DEVICE)
    rev_b = torch.tensor([bool(p.get('reversal', False)) for p in params_list], dtype=torch.bool, device=DEVICE)
    uncapped_b = torch.tensor([bool(p.get('uncapped', False)) for p in params_list], dtype=torch.bool, device=DEVICE)
    ARM_S1 = M.ARM_S1

    # ATR per distinct requested period (honors atr_period); stacked for lookup.
    periods = sorted(set(int(p.get('atr_period', 14)) for p in params_list))
    atr_pe_stack = torch.stack([_get_atr(pe_h, pe_l, pe_c, pp) for pp in periods], dim=0)  # (P,D,T1)
    atr_ce_stack = torch.stack([_get_atr(ce_h, ce_l, ce_c, pp) for pp in periods], dim=0)
    ap_index = torch.tensor([periods.index(int(p.get('atr_period', 14))) for p in params_list],
                            dtype=torch.long, device=DEVICE)  # (B,)
    atr_pe_sel = atr_pe_stack[ap_index]   # (B,D,T1) precomputed; indexed per-bar as a view
    atr_ce_sel = atr_ce_stack[ap_index]
    TP_PTS_t = torch.tensor(float(M.TP_PTS), device=DEVICE)
    use_bias_b = torch.tensor([bool(p.get('use_bias', M.USE_BIAS)) for p in params_list],
                              dtype=torch.bool, device=DEVICE)
    use_st_zone_b = torch.tensor([bool(p.get('use_st_zone', False)) for p in params_list],
                                 dtype=torch.bool, device=DEVICE)
    nt_start_b = torch.tensor([float(p.get('nt_start', -1)) for p in params_list], device=DEVICE)
    nt_end_b = torch.tensor([float(p.get('nt_end', -1)) for p in params_list], device=DEVICE)
    use_nt_b = nt_start_b >= 0
    # --- research-backed improvements (all default OFF so existing sweeps are unchanged) ---
    entry_start_b = torch.tensor([int(p.get('entry_start', 0)) for p in params_list], device=DEVICE)
    entry_end_b = torch.tensor([int(p.get('entry_end', T1)) for p in params_list], device=DEVICE)
    max_bars_b = torch.tensor([int(p.get('max_bars', 0)) for p in params_list], device=DEVICE)
    be_trigger_b = torch.tensor([float(p.get('be_trigger', 0.0)) for p in params_list], device=DEVICE)
    be_buffer_b = torch.tensor([float(p.get('be_buffer', 0.0)) for p in params_list], device=DEVICE)
    tp_frac_b = torch.tensor([float(p.get('tp_frac', 1.0)) for p in params_list], device=DEVICE)
    trail_dist_b = torch.tensor([float(p.get('trail_dist', 0.0)) for p in params_list], device=DEVICE)
    # --- SR touch-buffer: select the precomputed bounce tensor per config ---
    touch_buf_b = torch.tensor([float(p.get('touch_buffer', 1.0)) for p in params_list], device=DEVICE)
    tb_idx = torch.tensor([min(range(len(TOUCH_BUFFERS)),
                               key=lambda i: abs(TOUCH_BUFFERS[i] - v))
                           for v in touch_buf_b.tolist()],
                          dtype=torch.long, device=DEVICE)
    bounce_pe_sel = bounce_pe_stack[tb_idx]   # (B, D, T1)
    bounce_ce_sel = bounce_ce_stack[tb_idx]
    bias_ema = (bias_bull.int() - bias_bear.int()).to(torch.int8)  # +1 bull, -1 bear, 0 neutral
    mode = params_list[0].get('bias_mode', 'ema')
    if mode == 'lr':
        bias_grid = bias_lr_grid
    elif mode == 'ut':
        bias_grid = bias_ut_grid
    else:
        bias_grid = bias_ema

    pe_rev = ce_rev_on   # CE-rev ON signal -> PE entry (gated per-config by rev_b)
    ce_rev = pe_rev_on   # PE-rev ON signal -> CE entry (gated per-config by rev_b)

    in_pos = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    pos_side = torch.zeros(B, D, dtype=torch.int8, device=DEVICE)  # 1 PE, 2 CE
    entry_price = torch.zeros(B, D, device=DEVICE)
    sl_price = torch.zeros(B, D, device=DEVICE)
    tp_price = torch.zeros(B, D, device=DEVICE)
    pe_flag_armed = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    pe_super_armed = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    ce_flag_armed = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    ce_super_armed = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    pe_arm_t = torch.full((B, D), -999, dtype=torch.int32, device=DEVICE)
    ce_arm_t = torch.full((B, D), -999, dtype=torch.int32, device=DEVICE)
    daily_pts = torch.zeros(B, D, device=DEVICE)
    cap_hit = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    be_done = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    entry_bar = torch.full((B, D), -1, dtype=torch.int32, device=DEVICE)
    peak_price = torch.zeros(B, D, device=DEVICE)

    trades = [[] for _ in range(B)]
    pe_gate_blocked = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)  # reused buffer

    def _record(mask, kind, exit_price):
        """Gather exit events in one shot (no per-element CUDA syncs)."""
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return
        b_list = idx[:, 0].tolist()
        d_list = idx[:, 1].tolist()
        ep = entry_price[mask].tolist()
        xp = exit_price[mask].tolist()
        sd = pos_side[mask].tolist()
        for i in range(len(b_list)):
            pnl = (xp[i] - ep[i]) * LOT - FEE
            side = 'PE' if sd[i] == 1 else 'CE'
            trades[b_list[i]].append((trading_days[d_list[i]], side, kind, ep[i], xp[i], pnl))

    for t in range(1, T1):
        # --- exits (SL priority over TP); before arming so a same-bar exit frees in_pos ---
        if in_pos.any():
            pe_mask = pos_side == 1
            ce_mask = pos_side == 2
            sl_hit = (pe_mask & (pe_l[None, :, t] <= sl_price)) | (ce_mask & (ce_l[None, :, t] <= sl_price))
            tp_hit = (pe_mask & (pe_h[None, :, t] >= tp_price)) | (ce_mask & (ce_h[None, :, t] >= tp_price))
            do_sl = in_pos & sl_hit
            do_tp = in_pos & (~sl_hit) & tp_hit
            _record(do_sl, 'SL', sl_price)
            _record(do_tp, 'TP', tp_price)
            # vectorized P&L accrual + state reset
            gain = (sl_price - entry_price) * do_sl + (tp_price - entry_price) * do_tp
            daily_pts += gain
            done = do_sl | do_tp
            in_pos &= ~done
            pos_side[done] = 0
            cap_hit |= (cap_b[:, None] != 0.0) & (torch.abs(daily_pts) >= cap_b[:, None])
            pe_flag_armed[done] = False
            pe_super_armed[done] = False
            ce_flag_armed[done] = False
            ce_super_armed[done] = False

            # --- staleness exit (research: exit if trade hasn't moved in N bars) ---
            if max_bars_b.any() and in_pos.any():
                stale = in_pos & (max_bars_b[:, None] > 0) & ((t - entry_bar) > max_bars_b[:, None])
                if stale.any():
                    eod_exit = torch.where(pos_side == 1, pe_c[None, :, t], ce_c[None, :, t])
                    _record(stale, 'STALE', eod_exit)
                    daily_pts += (eod_exit - entry_price) * stale
                    in_pos &= ~stale
                    pos_side[stale] = 0
                    be_done[stale] = False
                    entry_bar[stale] = -1

        # --- arming (S1 <= ARM_S1 arms both flag & super); only while flat & not cap ---
        pa = (pe_s1[None, :, t] <= ARM_S1) & (~in_pos) & (~cap_hit)
        ca = (ce_s1[None, :, t] <= ARM_S1) & (~in_pos) & (~cap_hit)
        pe_flag_armed |= pa
        pe_super_armed |= pa
        ce_flag_armed |= ca
        ce_super_armed |= ca
        pe_arm_t = torch.where(pa, t, pe_arm_t)
        ce_arm_t = torch.where(ca, t, ce_arm_t)
        pe_flag_armed &= (t - pe_arm_t <= arm_b[:, None])
        pe_super_armed &= (t - pe_arm_t <= arm_b[:, None])
        ce_flag_armed &= (t - ce_arm_t <= arm_b[:, None])
        ce_super_armed &= (t - ce_arm_t <= arm_b[:, None])

        # --- breakeven stop (research: move SL to entry+buffer only after a real move) ---
        if be_trigger_b.any() and in_pos.any():
            dist_full = entry_price - sl_price          # = original SL distance
            be_px = entry_price + be_trigger_b[:, None] * dist_full
            hi = torch.where(pos_side == 1, pe_h[None, :, t], ce_h[None, :, t])
            trig = (be_trigger_b[:, None] > 0) & (~be_done) & in_pos & (hi >= be_px)
            if trig.any():
                buf = be_buffer_b[:, None].expand_as(sl_price)
                sl_price[trig] = entry_price[trig] + buf[trig]
                be_done[trig] = True

        # --- trailing stop (ratchet SL to peak - trail_dist after breakeven) ---
        if trail_dist_b.any() and in_pos.any():
            hi = torch.where(pos_side == 1, pe_h[None, :, t], ce_h[None, :, t])
            peak_price[in_pos] = torch.max(peak_price[in_pos], hi[in_pos])
            can_trail = (trail_dist_b[:, None] > 0) & be_done & in_pos
            if can_trail.any():
                new_sl = peak_price - trail_dist_b[:, None]
                improved = can_trail & (new_sl > sl_price)
                sl_price[improved] = new_sl[improved]

        # --- PE candidate triggers ---
        pe_m6 = pe_flag_armed & (t - pe_arm_t <= arm_b[:, None]) & pe_m6_full[None, :, t] & (~cap_hit)
        pe_super = pe_super_armed & (t - pe_arm_t <= arm_b[:, None]) & pe_super_full[None, :, t] & (~cap_hit)
        pe_rev_sig = pe_rev[None, :, t] & rev_b[:, None]
        pe_outer = pe_m6 | pe_super | pe_rev_sig
        pe_gate_blocked.zero_()
        pe_gate_blocked |= ue_b[:, None] & (elder_state[None, :, t] == 1)
        pe_gate_blocked |= (use_bias_b[:, None] & (~(bias_grid[None, :, t] == -1)))
        pe_gate_blocked |= ur_b[:, None] & ~(rsi_mat[None, :, t] < RSI_PE_LO)
        pe_cand = (pe_m6 | pe_super | pe_rev_sig) & (~in_pos) & (~cap_hit) & bounce_pe_sel[:, :, t]
        pe_cand &= ~(ue_b[:, None] & (elder_state[None, :, t] == 1))
        pe_cand &= ((bias_grid[None, :, t] == -1) | (~use_bias_b[:, None]))
        in_nt = use_nt_b[:, None] & (t >= nt_start_b[:, None]) & (t <= nt_end_b[:, None])
        pe_cand &= ~in_nt
        st_zone_pe = use_st_zone_b[:, None] & (
            ((pe_vwap[None, :, t] <= pe_c[None, :, t]) & (pe_c[None, :, t] <= pe_st[None, :, t])) |
            ((pe_st[None, :, t] <= pe_c[None, :, t]) & (pe_c[None, :, t] <= pe_vwap[None, :, t])))
        pe_cand &= ~st_zone_pe
        pe_cand &= (t >= entry_start_b[:, None]) & (t <= entry_end_b[:, None])
        pe_cand &= (~ur_b[:, None] | (rsi_mat[None, :, t] < RSI_PE_LO))
        ent_pe = pe_cand & (pos_side == 0)
        pe_ep = pe_c[:, t].unsqueeze(0).expand(B, D)
        atr_pe_t = atr_pe_sel[:, :, t]                       # (B,D)
        atr_pe_b = torch.minimum(atr_pe_t * atr_mult_b.unsqueeze(1), TP_PTS_t)
        sl_dist_pe = torch.where(atr_sl_b.unsqueeze(1), atr_pe_b, torch.minimum(sl_b.unsqueeze(1), TP_PTS_t))
        tp_dist_pe = torch.where(atr_sl_b.unsqueeze(1), atr_pe_b, torch.minimum(tp_b.unsqueeze(1), TP_PTS_t))
        entry_price[ent_pe] = pe_ep[ent_pe]
        sl_price[ent_pe] = (pe_ep - sl_dist_pe)[ent_pe]
        tp_price[ent_pe] = (pe_ep + tp_dist_pe * tp_frac_b[:, None])[ent_pe]
        pos_side[ent_pe] = 1
        entry_bar[ent_pe] = t
        in_pos[ent_pe] = True
        peak_price[ent_pe] = pe_ep[ent_pe]
        pe_flag_armed[ent_pe] = False
        pe_super_armed[ent_pe] = False

        # --- CE candidate triggers ---
        ce_m6 = ce_flag_armed & (t - ce_arm_t <= arm_b[:, None]) & ce_m6_full[None, :, t] & (~cap_hit)
        ce_super = ce_super_armed & (t - ce_arm_t <= arm_b[:, None]) & ce_super_full[None, :, t] & (~cap_hit)
        ce_rev_sig = ce_rev[None, :, t] & rev_b[:, None]
        ce_cand = (ce_m6 | ce_super | ce_rev_sig) & (~in_pos) & (~cap_hit) & bounce_ce_sel[:, :, t]
        ce_cand &= ~(ue_b[:, None] & (elder_state[None, :, t] == -1))
        ce_cand &= ((bias_grid[None, :, t] == 1) | (~use_bias_b[:, None]))
        in_nt_ce = use_nt_b[:, None] & (t >= nt_start_b[:, None]) & (t <= nt_end_b[:, None])
        ce_cand &= ~in_nt_ce
        st_zone_ce = use_st_zone_b[:, None] & (
            ((ce_vwap[None, :, t] <= ce_c[None, :, t]) & (ce_c[None, :, t] <= ce_st[None, :, t])) |
            ((ce_st[None, :, t] <= ce_c[None, :, t]) & (ce_c[None, :, t] <= ce_vwap[None, :, t])))
        ce_cand &= ~st_zone_ce
        ce_cand &= (t >= entry_start_b[:, None]) & (t <= entry_end_b[:, None])
        ce_cand &= (~ur_b[:, None] | (rsi_mat[None, :, t] > RSI_CE_HI))
        ent_ce = ce_cand & (pos_side == 0)
        ce_ep = ce_c[:, t].unsqueeze(0).expand(B, D)
        atr_ce_t = atr_ce_sel[:, :, t]                       # (B,D)
        atr_ce_b = torch.minimum(atr_ce_t * atr_mult_b.unsqueeze(1), TP_PTS_t)
        sl_dist_ce = torch.where(atr_sl_b.unsqueeze(1), atr_ce_b, torch.minimum(sl_b.unsqueeze(1), TP_PTS_t))
        tp_dist_ce = torch.where(atr_sl_b.unsqueeze(1), atr_ce_b, torch.minimum(tp_b.unsqueeze(1), TP_PTS_t))
        entry_price[ent_ce] = ce_ep[ent_ce]
        sl_price[ent_ce] = (ce_ep - sl_dist_ce)[ent_ce]
        tp_price[ent_ce] = (ce_ep + tp_dist_ce * tp_frac_b[:, None])[ent_ce]
        pos_side[ent_ce] = 2
        entry_bar[ent_ce] = t
        in_pos[ent_ce] = True
        peak_price[ent_ce] = ce_ep[ent_ce]
        ce_flag_armed[ent_ce] = False
        ce_super_armed[ent_ce] = False

    # End-of-session forced close for uncapped configs (TP disabled -> winners
    # ride until SL or session end). Close remaining positions at the last bar's
    # close price so run-ups are banked (matches ledger "uncapped" Last Hope #9).
    if uncapped_b.any():
        eod_mask = in_pos & uncapped_b.unsqueeze(1)
        if eod_mask.any():
            pe_close = pe_c[None, :, T1 - 1]
            ce_close = ce_c[None, :, T1 - 1]
            eod_exit = torch.where(pos_side == 1, pe_close, ce_close)
            _record(eod_mask, 'EOD', eod_exit)
            daily_pts += (eod_exit - entry_price) * eod_mask
            in_pos &= ~eod_mask
            pos_side[eod_mask] = 0

    return trades


# ---------------------------------------------------------------------------
# CUDA-GRAPH backed core: capture the 345-bar kernel sequence once, replay in a
# single launch. Eliminates per-launch WDDM overhead on Windows (the dominant
# cost). Exits are recorded into static-shape scatter buffers (no host sync in
# the loop), then gathered on the host after replay.
# ---------------------------------------------------------------------------
_GRAPH_CACHE = {}
_GRAPH_K = 256  # max trades recorded per (config, day); far above observed max

def _sim_loop(S):
    B = S['B']
    in_pos = S['in_pos']; pos_side = S['pos_side']; entry_price = S['entry_price']
    sl_price = S['sl_price']; tp_price = S['tp_price']
    pe_flag_armed = S['pe_flag_armed']; pe_super_armed = S['pe_super_armed']
    ce_flag_armed = S['ce_flag_armed']; ce_super_armed = S['ce_super_armed']
    pe_arm_t = S['pe_arm_t']; ce_arm_t = S['ce_arm_t']
    daily_pts = S['daily_pts']; cap_hit = S['cap_hit']
    sl_b = S['sl_b']; tp_b = S['tp_b']; atr_mult_b = S['atr_mult_b']; cap_b = S['cap_b']; arm_b = S['arm_b']
    atr_sl_b = S['atr_sl_b']; ue_b = S['ue_b']; ur_b = S['ur_b']; rev_b = S['rev_b']
    atr_pe_sel = S['atr_pe_sel']; atr_ce_sel = S['atr_ce_sel']
    ex_count = S['ex_count']; ex_side = S['ex_side']; ex_kind = S['ex_kind']
    ex_entry = S['ex_entry']; ex_exit = S['ex_exit']; ex_bar = S['ex_bar']; cur_t = S['cur_t']
    ONE_I8 = S['ONE_I8']; TWO_I8 = S['TWO_I8']; TP_PTS_t = S['TP_PTS_t']; ARM_S1 = S['ARM_S1']
    ap = S['ap_index']
    for t in range(1, T1):
        cur_t.fill_(t)
        # --- exits (SL priority over TP); masked by in_pos (parity-equiv to guarded) ---
        pe_mask = pos_side == 1
        ce_mask = pos_side == 2
        sl_hit = (pe_mask & (pe_l[None, :, t] <= sl_price)) | (ce_mask & (ce_l[None, :, t] <= sl_price))
        tp_hit = (pe_mask & (pe_h[None, :, t] >= tp_price)) | (ce_mask & (ce_h[None, :, t] >= tp_price))
        do_sl = in_pos & sl_hit
        do_tp = in_pos & (~sl_hit) & tp_hit
        do_exit = do_sl | do_tp
        idx = ex_count
        ex_side.scatter_(2, idx.unsqueeze(2), pos_side.unsqueeze(2))
        ex_kind.scatter_(2, idx.unsqueeze(2), torch.where(do_sl, ONE_I8, TWO_I8).unsqueeze(2))
        ex_entry.scatter_(2, idx.unsqueeze(2), entry_price.unsqueeze(2))
        ex_exit.scatter_(2, idx.unsqueeze(2), torch.where(do_sl, sl_price, tp_price).unsqueeze(2))
        ex_bar.scatter_(2, idx.unsqueeze(2), cur_t.expand(B, D).unsqueeze(2))
        ex_count.add_(do_exit.to(torch.int32))
        gain = (sl_price - entry_price) * do_sl + (tp_price - entry_price) * do_tp
        daily_pts += gain
        done = do_exit
        in_pos &= ~done
        pos_side[done] = 0
        cap_hit |= (cap_b[:, None] != 0.0) & (torch.abs(daily_pts) >= cap_b[:, None])
        pe_flag_armed[done] = False
        pe_super_armed[done] = False
        ce_flag_armed[done] = False
        ce_super_armed[done] = False
        # --- arming ---
        pa = (pe_s1[None, :, t] <= ARM_S1) & (~in_pos) & (~cap_hit)
        ca = (ce_s1[None, :, t] <= ARM_S1) & (~in_pos) & (~cap_hit)
        pe_flag_armed |= pa; pe_super_armed |= pa
        ce_flag_armed |= ca; ce_super_armed |= ca
        pe_arm_t = torch.where(pa, t, pe_arm_t)
        ce_arm_t = torch.where(ca, t, ce_arm_t)
        pe_flag_armed &= (t - pe_arm_t <= arm_b[:, None])
        pe_super_armed &= (t - pe_arm_t <= arm_b[:, None])
        ce_flag_armed &= (t - ce_arm_t <= arm_b[:, None])
        ce_super_armed &= (t - ce_arm_t <= arm_b[:, None])
        # --- PE triggers ---
        pe_m6 = pe_flag_armed & (t - pe_arm_t <= arm_b[:, None]) & pe_m6_full[None, :, t] & (~cap_hit)
        pe_super = pe_super_armed & (t - pe_arm_t <= arm_b[:, None]) & pe_super_full[None, :, t] & (~cap_hit)
        pe_rev_sig = pe_rev[None, :, t] & rev_b[:, None]
        pe_outer = pe_m6 | pe_super | pe_rev_sig
        pe_gate = ue_b[:, None] & (elder_state[None, :, t] == 1)
        pe_gate |= (use_bias_b[:, None] & (~(bias_grid[None, :, t] == -1)))
        pe_gate |= ur_b[:, None] & ~(rsi_mat[None, :, t] < RSI_PE_LO)
        pe_cand = (pe_m6 | pe_super | pe_rev_sig) & (~in_pos) & (~cap_hit) & bounce_pe[None, :, t]
        pe_cand &= ~(ue_b[:, None] & (elder_state[None, :, t] == 1))
        pe_cand &= ((bias_grid[None, :, t] == -1) | (~use_bias_b[:, None]))
        in_nt = use_nt_b[:, None] & (t >= nt_start_b[:, None]) & (t <= nt_end_b[:, None])
        pe_cand &= ~in_nt
        st_zone_pe = use_st_zone_b[:, None] & (
            ((pe_vwap[None, :, t] <= pe_c[None, :, t]) & (pe_c[None, :, t] <= pe_st[None, :, t])) |
            ((pe_st[None, :, t] <= pe_c[None, :, t]) & (pe_c[None, :, t] <= pe_vwap[None, :, t])))
        pe_cand &= ~st_zone_pe
        pe_cand &= (~ur_b[:, None] | (rsi_mat[None, :, t] < RSI_PE_LO))
        ent_pe = pe_cand & (pos_side == 0)
        pe_ep = pe_c[:, t].unsqueeze(0).expand(B, D)
        atr_pe_t = atr_pe_sel[ap, :, t]
        atr_pe_b = torch.minimum(atr_pe_t * atr_mult_b.unsqueeze(1), TP_PTS_t)
        sl_dist_pe = torch.where(atr_sl_b.unsqueeze(1), atr_pe_b, torch.minimum(sl_b.unsqueeze(1), TP_PTS_t))
        tp_dist_pe = torch.where(atr_sl_b.unsqueeze(1), atr_pe_b, torch.minimum(tp_b.unsqueeze(1), TP_PTS_t))
        entry_price[ent_pe] = pe_ep[ent_pe]
        sl_price[ent_pe] = (pe_ep - sl_dist_pe)[ent_pe]
        tp_price[ent_pe] = (pe_ep + tp_dist_pe)[ent_pe]
        pos_side[ent_pe] = 1
        in_pos[ent_pe] = True
        pe_flag_armed[ent_pe] = False
        pe_super_armed[ent_pe] = False
        # --- CE triggers ---
        ce_m6 = ce_flag_armed & (t - ce_arm_t <= arm_b[:, None]) & ce_m6_full[None, :, t] & (~cap_hit)
        ce_super = ce_super_armed & (t - ce_arm_t <= arm_b[:, None]) & ce_super_full[None, :, t] & (~cap_hit)
        ce_rev_sig = ce_rev[None, :, t] & rev_b[:, None]
        ce_cand = (ce_m6 | ce_super | ce_rev_sig) & (~in_pos) & (~cap_hit) & bounce_ce[None, :, t]
        ce_cand &= ~(ue_b[:, None] & (elder_state[None, :, t] == -1))
        ce_cand &= ((bias_grid[None, :, t] == 1) | (~use_bias_b[:, None]))
        in_nt_ce = use_nt_b[:, None] & (t >= nt_start_b[:, None]) & (t <= nt_end_b[:, None])
        ce_cand &= ~in_nt_ce
        st_zone_ce = use_st_zone_b[:, None] & (
            ((ce_vwap[None, :, t] <= ce_c[None, :, t]) & (ce_c[None, :, t] <= ce_st[None, :, t])) |
            ((ce_st[None, :, t] <= ce_c[None, :, t]) & (ce_c[None, :, t] <= ce_vwap[None, :, t])))
        ce_cand &= ~st_zone_ce
        ce_cand &= ~(pe_outer & pe_gate)
        ce_cand &= (~ur_b[:, None] | (rsi_mat[None, :, t] > RSI_CE_HI))
        ent_ce = ce_cand & (pos_side == 0)
        ce_ep = ce_c[:, t].unsqueeze(0).expand(B, D)
        atr_ce_t = atr_ce_sel[ap, :, t]
        atr_ce_b = torch.minimum(atr_ce_t * atr_mult_b.unsqueeze(1), TP_PTS_t)
        sl_dist_ce = torch.where(atr_sl_b.unsqueeze(1), atr_ce_b, torch.minimum(sl_b.unsqueeze(1), TP_PTS_t))
        tp_dist_ce = torch.where(atr_sl_b.unsqueeze(1), atr_ce_b, torch.minimum(tp_b.unsqueeze(1), TP_PTS_t))
        entry_price[ent_ce] = ce_ep[ent_ce]
        sl_price[ent_ce] = (ce_ep - sl_dist_ce)[ent_ce]
        tp_price[ent_ce] = (ce_ep + tp_dist_ce)[ent_ce]
        pos_side[ent_ce] = 2
        in_pos[ent_ce] = True
        ce_flag_armed[ent_ce] = False
        ce_super_armed[ent_ce] = False


def _get_graph(B, periods):
    key = (B, tuple(periods))
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    dev = DEVICE
    S = {'B': B, 'ARM_S1': torch.tensor(M.ARM_S1, device=dev),
         'TP_PTS_t': torch.tensor(float(M.TP_PTS), device=dev),
         'ONE_I8': torch.tensor(1, dtype=torch.int8, device=dev),
         'TWO_I8': torch.tensor(2, dtype=torch.int8, device=dev)}
    for nm, dt in [('sl_b', torch.float32), ('tp_b', torch.float32), ('atr_mult_b', torch.float32),
                  ('cap_b', torch.float32), ('arm_b', torch.float32), ('atr_sl_b', torch.bool),
                  ('ue_b', torch.bool), ('ur_b', torch.bool), ('rev_b', torch.bool),
                  ('ap_index', torch.long)]:
        S[nm] = torch.zeros(B, device=dev, dtype=dt)
    for nm in ['in_pos', 'pe_flag_armed', 'pe_super_armed', 'ce_flag_armed', 'ce_super_armed', 'cap_hit']:
        S[nm] = torch.zeros(B, D, dtype=torch.bool, device=dev)
    for nm in ['pos_side']:
        S[nm] = torch.zeros(B, D, dtype=torch.int8, device=dev)
    for nm in ['entry_price', 'sl_price', 'tp_price', 'daily_pts']:
        S[nm] = torch.zeros(B, D, device=dev)
    for nm in ['pe_arm_t', 'ce_arm_t']:
        S[nm] = torch.full((B, D), -999, dtype=torch.int32, device=dev)
    K = _GRAPH_K
    S['ex_count'] = torch.zeros(B, D, dtype=torch.int32, device=dev)
    S['ex_side'] = torch.zeros(B, D, K, dtype=torch.int8, device=dev)
    S['ex_kind'] = torch.zeros(B, D, K, dtype=torch.int8, device=dev)
    S['ex_entry'] = torch.zeros(B, D, K, device=dev)
    S['ex_exit'] = torch.zeros(B, D, K, device=dev)
    S['ex_bar'] = torch.zeros(B, D, K, dtype=torch.int32, device=dev)
    S['cur_t'] = torch.zeros((), dtype=torch.int32, device=dev)
    S['atr_pe_sel'] = torch.stack([_get_atr(pe_h, pe_l, pe_c, p) for p in periods], 0)
    S['atr_ce_sel'] = torch.stack([_get_atr(ce_h, ce_l, ce_c, p) for p in periods], 0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _sim_loop(S)
    S['graph'] = g
    _GRAPH_CACHE[key] = S
    return S


def _gpu_sim_core(params_list):
    """Batched simulation (CUDA-graph attempt disabled on this PyTorch/Windows
    build due to a native CUDAGraph segfault; delegates to the stable,
    parity-verified eager core). Public entry used by the sweep."""
    return _eager_sim_core(params_list)


def _gather_trades(S, B):
    ex_count = S['ex_count'].cpu().numpy()
    ex_side = S['ex_side'].cpu().numpy()
    ex_kind = S['ex_kind'].cpu().numpy()
    ex_entry = S['ex_entry'].cpu().numpy()
    ex_exit = S['ex_exit'].cpu().numpy()
    trades = [[] for _ in range(B)]
    for b in range(B):
        for d in range(D):
            cnt = int(ex_count[b, d])
            if cnt >= _GRAPH_K:
                cnt = _GRAPH_K - 1  # safety clamp (should never trigger)
            for k in range(cnt):
                side = 'PE' if ex_side[b, d, k] == 1 else 'CE'
                kind = 'SL' if ex_kind[b, d, k] == 1 else 'TP'
                ep = float(ex_entry[b, d, k]); xp = float(ex_exit[b, d, k])
                trades[b].append((trading_days[d], side, kind, ep, xp, (xp - ep) * LOT - FEE))
    return trades


def gpu_sim(sl=None, tp=None, arm_window=None, use_elder=None, use_rsi=None,
            atr_sl=False, atr_mult=1.0, atr_period=14, reversal=False, cap=0.0,
            uncapped=False):
    """Run ONE config's simulation on the GPU. Convenience wrapper around the
    batched core. Returns a flat list of trade tuples (metric-compatible)."""
    p = dict(sl=sl if sl is not None else M.SL_PTS,
             tp=tp if tp is not None else M.TP_PTS,
             arm_window=arm_window if arm_window is not None else M.ARM_WINDOW,
             use_elder=M.USE_ELDER if use_elder is None else use_elder,
             use_rsi=M.USE_RSI if use_rsi is None else use_rsi,
             atr_sl=atr_sl, atr_mult=atr_mult,
             reversal=reversal, cap=cap, uncapped=uncapped)
    return _gpu_sim_core([p])[0]


def gpu_sim_batch(params_list):
    """Run a batch of configs. Returns a list of trade-lists (one per config)."""
    return _gpu_sim_core(params_list)


def gpu_run_backtest_params(p):
    """Mirror run_7y_v4_master.run_backtest_params but on GPU."""
    trades = _gpu_sim_core([p])[0]
    return M._metrics_from_trades(trades)


if __name__ == '__main__':
    t0 = time.time()
    tr = gpu_sim()
    print(f"GPU sim done in {time.time()-t0:.2f}s, trades={len(tr)}")
