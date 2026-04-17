#!/usr/bin/env python3
"""
Sanity check: Verify that optimized params actually change strategy behavior.
"""
from knowledge.strategy_library import create_absorption_strategy
from core.data_structures import OrderFlowState, OrderBook, Regime, PriceLevel
from datetime import datetime

# Create minimal mock state that would trigger absorption
def create_mock_state(absorption_strength: float, price: float = 1.32):
    ts = datetime.now()
    return OrderFlowState(
        timestamp=ts,
        order_book=OrderBook(
            timestamp=ts,
            bids=[PriceLevel(price=price - 0.001, size=1000)],
            asks=[PriceLevel(price=price + 0.001, size=1000)],
        ),
        regime=Regime.RANGING,
        features={
            "recent_absorption_strength": absorption_strength,
            "volume_acceleration": 2.0,
            "price_change_pct_60s": 0.0005,
            "abs_delta_60s": 1000,
            "depth_imbalance_10": 0.10,
            "price_vs_poc_pct": 0.002,
            "book_trade_agreement": 1.0,
            "spread_bps": 10.0,
            "bid_depth_10": 3000.0,
            "ask_depth_10": 3000.0,
            "price_change_pct_300s": 0.004,
            "atr_60s": 0.006,
            "delta_60s": 500,
            "net_pressure": 0.3,
        },
        recent_trades=[]
    )

strategy = create_absorption_strategy()

print("=== PARAM INJECTION TEST ===")

# Test 1: Default params (threshold = 0.30 hardcoded in condition)
state_high = create_mock_state(absorption_strength=0.80)  # High strength
signal_default = strategy.evaluate(state_high)
print(f"1. Default params, strength=0.80: signal={signal_default is not None}")

# Test 2: Inject HIGH threshold (0.75) - should REJECT the same state
signal_high_thresh = strategy.evaluate(
    state_high, 
    optimized_params={"abs__entry_str_min": 0.75}
)
print(f"2. High threshold (0.75), strength=0.80: signal={signal_high_thresh is not None}")
print(f"   → Condition threshold after injection: {strategy.entry_conditions[0].threshold}")

# Test 3: Inject LOW threshold (0.10) - should ACCEPT more easily
state_low = create_mock_state(absorption_strength=0.25)  # Lower strength
signal_low_thresh = strategy.evaluate(
    state_low,
    optimized_params={"abs__entry_str_min": 0.10}
)
print(f"3. Low threshold (0.10), strength=0.25: signal={signal_low_thresh is not None}")
print(f"   → Condition threshold after injection: {strategy.entry_conditions[0].threshold}")

# EXPECTED OUTPUT:
# 1. True  (0.80 > 0.30 default)
# 2. True  (0.80 > 0.75 injected) ← If this is False, injection WORKS
# 3. True  (0.25 > 0.10 injected) ← If this is False, injection WORKS
# 
# If ALL THREE are True → injection is BROKEN (threshold not updating)
# If #2 is False while #1 is True → injection is WORKING ✓
