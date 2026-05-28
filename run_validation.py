"""
PHASE 5: VALIDATION - R:R Fix, Breakeven Fix, Win Rate & Expectancy
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
    tp_trades = [t for t in trades if t.exit_reason == "take_profit"]
    avg_tp_pnl = np.mean([t.pnl for t in tp_trades]) if tp_trades else 0
    print(
        f"[{strat_name}] Trades={metrics.total_trades}, "
        f"WinRate={metrics.win_rate*100:.1f}%, "
        f"AvgDur={metrics.avg_trade_duration_seconds:.0f}s, "
        f"SL_Moved={sl_moved}, AvgTP_PnL={avg_tp_pnl:.2f}"
    )

print()
print("=" * 62)
print("PROOF OF LIFE - PHASE 5 VALIDATION")
print("=" * 62)

print(f"\nExit Reason Distribution ({len(all_trades)} trades):")
for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
    pct = count / len(all_trades) * 100 if all_trades else 0
    print(f"  {reason:30s}: {count:4d} ({pct:.1f}%)")

noise_keys = [
    "buying_exhaustion", "delta_divergence_against", "selling_exhaustion",
    "sweep_against_long", "sweep_against_short", "book_pressure_collapse",
]
noise_exits = sum(exit_counts.get(k, 0) for k in noise_keys)
print(f"Noise exits: {noise_exits} (expected 0) [{'PASS' if noise_exits == 0 else 'FAIL'}]")

if all_trades:
    # 1) Proof of R:R Fix: average TP PnL vs average SL PnL
    tp_trades = [t for t in all_trades if t.exit_reason == "take_profit"]
    sl_trades = [t for t in all_trades if t.exit_reason == "stop_loss"]
    avg_tp_pnl = np.mean([t.pnl for t in tp_trades]) if tp_trades else 0
    avg_sl_pnl = np.mean([t.pnl for t in sl_trades]) if sl_trades else 0

    print(f"\n>> PROOF 1: R:R FIX (Let Winners Run)")
    print(f"  Take Profit trades: {len(tp_trades)}, Avg PnL: {avg_tp_pnl:+.2f}")
    print(f"  Stop Loss trades:   {len(sl_trades)}, Avg PnL: {avg_sl_pnl:+.2f}")
    if len(tp_trades) > 1:
        tp_pnls = sorted([t.pnl for t in tp_trades])
        print(f"  TP PnL range: {tp_pnls[0]:+.2f} to {tp_pnls[-1]:+.2f}")
        big_runners = sum(1 for t in tp_trades if t.pnl > 100)
        print(f"  TP PnL > 100: {big_runners} trades (big runners)")
    if avg_tp_pnl > abs(avg_sl_pnl):
        print(f"  [PASS] Avg TP PnL ({avg_tp_pnl:+.2f}) exceeds Avg SL loss ({avg_sl_pnl:+.2f})")
    else:
        print(f"  [WARN] Avg TP PnL ({avg_tp_pnl:+.2f}) vs Avg SL loss ({avg_sl_pnl:+.2f})")

    # 2) Proof of Breakeven Fix
    be_trades = [t for t in all_trades if t.stop_loss and t.stop_loss > t.entry_price]
    be_net_pnl = sum(t.pnl for t in be_trades) if be_trades else 0
    print(f"\n>> PROOF 2: BREAKEVEN FIX (0.15% -> 0.25% activation)")
    print(f"  Trades with SL moved above entry: {len(be_trades)}")
    print(f"  Net PnL of those trades: {be_net_pnl:+.2f}")
    if len(be_trades) > 0:
        avg_be = np.mean([t.pnl for t in be_trades])
        print(f"  Avg PnL per such trade: {avg_be:+.2f}")
        for t in be_trades[:3]:
            print(f"    entry={t.entry_price:.4f}, exit={t.exit_price:.4f}, "
                  f"sl={t.stop_loss:.4f}, pnl={t.pnl:+.2f}, reason={t.exit_reason}")

    # 3) Win Rate & Expectancy
    winning = [t for t in all_trades if t.pnl > 0]
    losing = [t for t in all_trades if t.pnl <= 0]
    win_rate = len(winning) / len(all_trades) * 100
    loss_rate = 100 - win_rate

    avg_win = np.mean([t.pnl for t in winning]) if winning else 0
    avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 0
    expectancy = (win_rate / 100 * avg_win) - (loss_rate / 100 * avg_loss)
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    print(f"\n>> PROOF 3: WIN RATE & EXPECTANCY")
    wr_status = "PASS > 35%" if win_rate > 35 else "FAIL < 35%"
    print(f"  Win Rate:  {win_rate:.1f}% ({len(winning)}/{len(all_trades)}) [{wr_status}]")
    print(f"  Avg Win:   {avg_win:+.2f}")
    print(f"  Avg Loss:  {avg_loss:+.2f}")
    print(f"  Expectancy: {expectancy:+.2f} per trade")
    print(f"  R:R Ratio: {rr_ratio:.2f}:1")

    # Overall
    print(f"\n{'=' * 62}")
    checks = []
    checks.append(("Noise exits = 0", noise_exits == 0))
    checks.append(("Win rate > 35%", win_rate > 35))
    checks.append(("Trades generated", len(all_trades) > 0))
    all_pass = all(c[1] for c in checks)
    for label, ok in checks:
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {label}")
    print(f"\n  VERDICT: [{'PASS - All criteria met!' if all_pass else 'FAIL'}]")
    print(f"{'=' * 62}")
else:
    print("\n  [FAIL] No trades generated")
