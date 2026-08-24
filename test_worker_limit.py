import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class WorkerLimitTests(unittest.TestCase):
    def test_backtest_sources_do_not_use_worker_caps_above_eight(self):
        sources = [
            *ROOT.glob("backtest*.py"),
            *ROOT.glob("analyze*.py"),
            *ROOT.glob("grid_*.py"),
            *ROOT.glob("test_*.py"),
            ROOT / "find_max_net_points.py",
            ROOT / "redo_last_two_backtests.py",
            ROOT / "deep_profit_audit.py",
            ROOT / "check_2m_sltp.py",
            ROOT / "compare_1m_vs_2m.py",
            ROOT / "compare_all_timeframes.py",
        ]
        patterns = (
            r"min\(cpu_count\(\),\s*12\)",
            r"max\(2,\s*int\(cpu_count\(\)\s*\*\s*0\.85\)\)",
        )
        violations = []
        for path in sorted(set(sources)):
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if any(re.search(pattern, text) for pattern in patterns):
                violations.append(path.name)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
