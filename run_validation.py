"""
Run full proof-of-life validation and print results.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from core.feature_engine import FeatureConfig
from backtesting.engine import BacktestEngine
from knowledge.strategy_library import get_strategy

data_dir = Path("data/backtests")
for p in ["XRP*merged*.parquet", "XRPUSDT*.parquet", "XRP*.parquet"]:
    cand = list(data_dir.glob(p))
    if cand:
        df = pd.read_parquet(cand[0])
        break
else:
    print("ERROR: No XRP data found")
    sys.exit(1)

print(f"Total rows: {len(df)}")
df = df.head(50000)
print(f"Using {len(df)} rows")

strategies_to_test = ["absorption", "delta_divergence", "stacked_imbalance", "value_area"]
all_trades = []
exit_counts = Counter()

for strat_name in strategies_to_test:
    strategy = get_strategy(strat_name)
    if strategy is None:
        continue
    engine = BacktestEngine(
        initial_capital=100000.0,
        feature_config=FeatureConfig(windows=[15, 30, 60, 300, 600, 900]),
    )
    metrics = engine.run(df, strategy)
    trades = engine.closed_trades
    all_trades.extend(trades)
    for t in trades:
        exit_counts[t.exit_reason] += 1
    sl_moved = sum(1 for t in trades if t.stop_loss and t.stop_loss > t.entry_price)
    print(
        f"[{strat_name}] Trades={metrics.total_trades}, "
        f"WinRate={metrics.win_rate*100:.1f}%, "
        f"AvgDur={metrics.avg_trade_duration_seconds:.0f}s, "
        f"SL_Moved={sl_moved}"
    )

print()
print("=" * 60)
print("PROOF OF LIFE — VALIDATION RESULTS")
print("=" * 60)

print(f"Exit Reason Distribution ({len(all_trades)} trades):")
for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
    pct = count / len(all_trades) * 100 if all_trades else 0
    print(f"  {reason:30s}: {count:4d} ({pct:.1f}%)")

noise_keys = [
    "buying_exhaustion", "delta_divergence_against", "selling_exhaustion",
    "sweep_against_long", "sweep_against_short", "book_pressure_collapse",
]
noise_exits = sum(exit_counts.get(k, 0) for k in noise_keys)
noise_status = "PASS" if noise_exits == 0 else "FAIL"
print(f"Noise exits: {noise_exits} (expected 0) [{noise_status}]")

if all_trades:
    winning = [t for t in all_trades if t.pnl > 0]
    win_rate = len(winning) / len(all_trades) * 100
    wr_status = "PASS" if win_rate > 30 else "FAIL"
    print(f"Win Rate: {win_rate:.1f}% ({len(winning)}/{len(all_trades)}) [{wr_status}] (need > 30%)")

    durations = [t.duration_seconds for t in all_trades]
    avg_min = np.mean(durations) / 60
    print(f"Avg Duration: {avg_min:.1f} min ({np.mean(durations):.0f}s)")

    sl_moved_count = sum(1 for t in all_trades if t.stop_loss and t.stop_loss > t.entry_price)
    print(f"Trades with SL moved above entry (breakeven/trailing): {sl_moved_count}")
    if sl_moved_count > 0:
        ex = next(t for t in all_trades if t.stop_loss and t.stop_loss > t.entry_price)
        print(f"  Example: entry={ex.entry_price:.4f}, exit={ex.exit_price:.4f}, "
              f"sl={ex.stop_loss:.4f}, pnl={ex.pnl:.2f}, reason={ex.exit_reason}")

    all_pass = (win_rate > 30 and noise_exits == 0 and len(all_trades) > 0)
    verdict = "PASS - All criteria met!" if all_pass else "FAIL - Some criteria not met"
    print(f"\nOVERALL: [{verdict}]")
else:
    print("FAIL - No trades generated")
print("=" * 60)
