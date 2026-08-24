# 🚀 Live Market Trading & Launch Checklist (Tomorrow)

> **Strategy:** `COMBINED_SUPREME_STRATEGY` (3-Tier S/R + Two-Bar Confirmation + 15m Gate)  
> **Broker:** Flattrade (PiConnect REST API v2 | User ID: `FZ52739`)  
> **Instrument:** Nifty 50 Weekly Options (2nd ITM Strikes | 1 Lot = 65 Qty)  
> **Target Calmar:** `1,595+` | **Historical Win Rate:** `69.80%` | **Green Days:** `91.2%`

---

## 📋 Phase 1: Pre-Market Preparation (08:30 – 09:00 IST)

- [ ] **1. Verify Flattrade API Whitelisted IP**
  * Check **[wall.flattrade.in](https://wall.flattrade.in)** for `Quad Rotation Bot` app:
    * If running on **AWS Lightsail VPS:** Primary IP = **`65.2.50.227`**.
    * If running **Locally on PC:** Primary IP = Current Public IP from `https://api.ipify.org`.
- [ ] **2. Run Automated Broker Auth & Diagnostics**
  * Run test fire script:
    ```bash
    python flattrade_bot/test_live_broker_fire.py
    ```
  * Confirm output: `[PASS] Live Session Token Acquired` & `Status: 200 OK`.
- [ ] **3. Verify Account Margin & Risk Limits**
  * Confirm available cash is $\ge ₹15,000$ (Verified baseline: ₹22,389.96).
  * Max Daily Loss Guard: **`30.0 Points`** active (dynamically scales with lot size: ₹1,950 for 1 lot, ₹3,900 for 2 lots, etc.).
  * Max Consecutive Loss Cutoff: `8 trades` active.

---

## 📋 Phase 2: Live Market Engine Startup (09:05 – 09:15 IST)

- [ ] **4. Launch Live Trading Engine / Service**
  * **Option A (AWS Lightsail VPS):**
    ```bash
    sudo systemctl restart flattrade-bot
    sudo systemctl status flattrade-bot
    ```
  * **Option B (Local Terminal Dashboard):**
    ```bash
    python flattrade_bot/undisputed_main.py
    ```
- [ ] **5. Confirm Initialized 3-Tier S/R Matrix**
  * Verify that the bot has calculated all key levels:
    * **Tier 1+ Supreme:** Virgin CPR (Pivot, TC, BC)
    * **Tier 1 Core:** Camarilla H3/L3, Daily CPR (P/TC/BC), Daily VWAP, 5m EMA 20 & 200
    * **Tier 2 Momentum:** Opening 3m Range (High/Low formed at 09:18), 3m EMA 20, PDH/PDL
    * **Tier 3 Macro:** Fibonacci H3/L3, Camarilla H4/L4
- [ ] **6. Discord Remote Connection Check**
  * Open Discord on phone/PC and type:
    ```text
    /trading status
    /trading levels
    ```
  * Confirm bot responds with live spot price, level table, and active status.

---

## 📋 Phase 3: Live Session Execution Windows

- [ ] **7. Morning Prime Window (09:18 – 11:00 IST)**
  * Watch for first Two-Bar confirmed rejection signals ($\text{Score} \ge 50$).
  * Verify 2nd ITM strike selection ($\Delta \approx 0.60$) and Aggressive Limit entry ($\text{LTP} \pm 1.0\text{ pt}$).
  * Verify Step Trailing Stop activation ($+6.0\text{ pts}$ trigger, $2.0\text{ pt}$ trail).
- [ ] **8. Midday Standdown Window (11:00 – 13:30 IST)**
  * Confirm bot status changes to `STANDDOWN (11:00-13:30)` to block low-volume chop.
  * Bot will reject new entries during this window.
- [ ] **9. Afternoon Momentum Window (13:30 – 15:00 IST)**
  * Bot automatically resumes scanning for afternoon breakout/rejection setups.

---

## 📋 Phase 4: Market Close & EOD Reconciliation (15:00 – 15:35 IST)

- [ ] **10. Mandatory Auto-Squareoff (15:00 IST)**
  * Verify all open intraday option positions are squared off automatically.
- [ ] **11. Final Daily P&L Audit (15:30 IST)**
  * Review realized P&L card posted to Discord.
  * Cross-check trade executions with Flattrade OrderBook / TradeBook.

---

## 🆘 Quick Emergency Commands (Discord / Terminal)

| Action | Discord Slash Command | Terminal Command |
| :--- | :--- | :--- |
| **Check Live Status** | `/trading status` | `python -c "from flattrade_bot.undisputed_main import *; ..."` |
| **View Today's Levels** | `/trading levels` | Review terminal dashboard |
| **Emergency Pause** | `/trading pause` | `sudo systemctl stop flattrade-bot` |
| **Emergency Square-Off** | `/trading squareoff` | Broker Mobile App $\rightarrow$ Positions $\rightarrow$ Close All |
| **Resume Trading** | `/trading resume` | `sudo systemctl start flattrade-bot` |
