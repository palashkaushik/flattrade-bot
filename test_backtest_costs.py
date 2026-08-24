import unittest

from backtest_walkforward_fees import SLIPPAGE_PTS, apply_costs, trade_cost
from artifacts.f6_hybrid.causal_live_parity_research import (
    is_daily_loss_cap_reached,
    is_daily_profit_cap_reached,
    resolve_dynamic_exit_levels,
    resolve_tp_points,
    stats,
)


class BacktestCostModelTests(unittest.TestCase):
    def test_backtest_default_slippage_is_one_point_per_side(self):
        self.assertEqual(SLIPPAGE_PTS, 1.0)

        trade = {"pts": 10.0, "entry": 100.0, "exit": 110.0}
        apply_costs([trade], brokerage_per_order=0.0)

        self.assertEqual(trade["pts_net"], 8.0)

    def test_gst_applies_to_brokerage_exchange_and_sebi_only(self):
        self.assertEqual(
            trade_cost(100.0, 110.0, brokerage_per_order=20.0),
            57.52,
        )

    def test_cost_model_accepts_zero_slippage_override(self):
        trade = {"pts": 10.0, "entry": 100.0, "exit": 110.0}
        apply_costs([trade], brokerage_per_order=0.0, slippage_pts=0.0)

        self.assertEqual(trade["pts_net"], 10.0)

    def test_daily_profit_cap_uses_net_points_and_includes_the_boundary(self):
        self.assertFalse(is_daily_profit_cap_reached(30.0 * 65 - 0.01, 30.0))
        self.assertTrue(is_daily_profit_cap_reached(30.0 * 65, 30.0))
        self.assertFalse(is_daily_profit_cap_reached(30.0 * 65, None))

    def test_daily_loss_cap_uses_net_points_and_includes_the_boundary(self):
        self.assertFalse(is_daily_loss_cap_reached(-30.0 * 65 + 0.01, 30.0))
        self.assertTrue(is_daily_loss_cap_reached(-30.0 * 65, 30.0))
        self.assertFalse(is_daily_loss_cap_reached(-30.0 * 65, None))

    def test_stats_reports_average_trades_sl_and_tp(self):
        trades = [
            {"date": "2020-01-01", "rs_net": 65, "fee": 0, "sl_points": 10, "tp_points": 20},
            {"date": "2020-01-01", "rs_net": -65, "fee": 0, "sl_points": 20, "tp_points": 40},
            {"date": "2020-01-02", "rs_net": 65, "fee": 0, "sl_points": 30, "tp_points": 60},
        ]

        result = stats(trades, day_count=2)

        self.assertEqual(result["avg_trades_per_day"], 1.5)
        self.assertEqual(result["avg_sl_points"], 20.0)
        self.assertEqual(result["avg_tp_points"], 40.0)

    def test_fixed_tp_overrides_atr_tp_while_atr_sl_remains_available(self):
        self.assertEqual(resolve_tp_points(5.0, 6.0, 35.0, 15.0), 15.0)
        self.assertEqual(resolve_tp_points(5.0, 6.0, 35.0, None), 30.0)
        self.assertEqual(resolve_tp_points(None, 6.0, 35.0, None), 35.0)

    def test_dynamic_exit_policies_update_sl_and_tp_as_defined(self):
        params = {"atr_sl_mult": 2.0, "atr_tp_mult": 3.0}
        position = {"entry": 100.0, "sl": 90.0, "target": 120.0, "high_watermark": 100.0}

        both = resolve_dynamic_exit_levels(position, 5.0, params, "dynamic_both")
        ratchet = resolve_dynamic_exit_levels({**position, "sl": 94.0}, 6.0, params, "ratchet_sl_dynamic_tp")
        chandelier = resolve_dynamic_exit_levels({**position, "high_watermark": 110.0}, 5.0, params, "chandelier_sl_dynamic_tp")

        self.assertEqual(both, (90.0, 115.0))
        self.assertEqual(ratchet, (94.0, 118.0))
        self.assertEqual(chandelier, (100.0, 115.0))


if __name__ == "__main__":
    unittest.main()
