# Optimization Crisis Fix Plan

## 🚨 Critical Issues Identified

### Issue #1: All Results Match Default Values
**Symptom**: Every optimized parameter equals its default value.

**Root Causes** (ranked by probability):

1. **Backtest Failures Silently Returning -inf**
   - If `backtest_fn(params)` raises exceptions for most parameter combinations
   - Optuna only has valid scores for the warm-start defaults
   - **Evidence**: 10+ day runtime suggests many failed trials

2. **Warm-Start Trial Never Surpassed**
   - Objective function may be noisy or flat
   - Other parameter combinations fail, leaving warm-start as "best"

3. **Parameter Space Issues**
   - Some ranges too wide (e.g., `abs__entry_delta_min: 0–50000`) cause numerical instability
   - `abs__entry_poc_range` uses symmetric bounds but optimizer suggests single value

4. **Missing Common Parameters in Output**
   - Where are: `min_conditions_satisfied`, `base_position_pct`, regime multipliers?
   - If these weren't optimized, risk management wasn't tuned

---

## 🔧 Immediate Fixes Required

### Fix #1: Add Comprehensive Backtest Error Logging
**File**: `optimization/optuna_optimizer.py`

**Problem**: Exceptions are caught but not logged with parameter details.

**Solution**:
```python
except Exception as e:
    logger.error(f"[objective] Trial {trial.number} backtest FAILED:")
    logger.error(f"  Parameters: {params}")
    logger.error(f"  Exception: {type(e).__name__}: {e}")
    import traceback
    logger.error(f"  Traceback: {traceback.format_exc()}")
    return float("-inf")
```

### Fix #2: Narrow Parameter Ranges to Realistic Bounds
**File**: `optimization/optuna_optimizer.py` & `wider_optimizer.py`

**Problem**: Extremely wide ranges cause:
- Numerical instability
- Excessive trial failures
- 10+ day optimization times

**Solution**: Set realistic, tight bounds based on XRP/USDT microstructure:

| Parameter | Old Range | New Range | Rationale |
|-----------|-----------|-----------|-----------|
| `abs__entry_chg60_max` | 0.0–0.08 | **0.0005–0.005** | Absorption needs tight stability |
| `abs__entry_delta_min` | 0–50000 | **0–5000** | 50k is unrealistic for XRP |
| `abs__entry_imbal_min` | 0.0–0.70 | **0.02–0.40** | 70% imbalance is extreme |
| `abs__entry_poc_range` | 0.0–0.08 | **0.001–0.015** | POC proximity must be tight |
| `abs__filter_bid_min` | 500–100000 | **1000–20000** | Prevent overfitting to low depth |
| `abs__filter_ask_min` | 500–100000 | **1000–20000** | Prevent overfitting to low depth |
| `abs__filter_chg300_range` | 0.003–0.15 | **0.003–0.02** | Tight regime filter |
| `min_conditions_satisfied` | 1–6 | **2–5** | Prevent degenerate strategies |
| `base_position_pct` | 0.01–0.50 | **0.02–0.20** | Risk control |

### Fix #3: Add Trial Success Rate Monitoring
**File**: `optimization/optuna_optimizer.py`

**Problem**: No visibility into how many trials are failing.

**Solution**:
```python
def optimize(self, ..., success_threshold: float = 0.3):
    successful_trials = 0
    failed_trials = 0
    
    # In objective function:
    if metrics.get("_penalized", False) or score == float("-inf"):
        failed_trials += 1
    else:
        successful_trials += 1
    
    # Log every 50 trials:
    if trial.number % 50 == 0:
        success_rate = successful_trials / (successful_trials + failed_trials)
        logger.warning(f"Trial {trial.number}: Success rate = {success_rate:.2%}")
        if success_rate < success_threshold:
            logger.error(f"CRITICAL: Success rate below {success_threshold:.0%}!")
            logger.error("Parameter ranges may be too wide or backtest has bugs.")
```

### Fix #4: Fix POC Range Symmetric Handling
**File**: `wider_optimizer.py` line 2190

**Problem**: `abs__entry_poc_range` min is 0.001, but could suggest values that break between logic.

**Solution**: Ensure min > 0 and add validation:
```python
"abs__entry_poc_range": {"min": 0.001, "max": 0.015, "step": 0.001, "default": 0.006},
# In _apply_params:
if param_name.endswith("_range") and cond.operator == "between":
    abs_val = abs(value)
    if abs_val < 0.0001:  # Prevent degenerate range
        logger.warning(f"POC range too small: {abs_val}, clamping to 0.001")
        abs_val = 0.001
    cond.threshold = -abs_val
    cond.threshold_high = abs_val
```

### Fix #5: Add Timeout Per Trial
**File**: `optimization/optuna_optimizer.py`

**Problem**: Single slow trial can hang optimization for hours.

**Solution**:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Backtest exceeded time limit")

def backtest_with_timeout(params, timeout=60):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = backtest_fn(params)
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        logger.error(f"Trial timeout with params: {params}")
        return {"_penalized": True}
```

### Fix #6: Validate Warm-Start Params Before Enqueue
**File**: `optimization/optuna_optimizer.py` line 449-456

**Problem**: Warm-start might have invalid parameter combinations.

**Solution**:
```python
if self.warm_start_params:
    # Test warm-start first
    try:
        test_metrics = backtest_fn(self.warm_start_params)
        if test_metrics.get("_penalized", False):
            logger.warning("Warm-start params penalized! Adjusting...")
            # Adjust to minimum viable params
            self.warm_start_params["min_conditions_satisfied"] = 2
            self.warm_start_params["min_score_threshold"] = 1.5
    except Exception as e:
        logger.error(f"Warm-start backtest failed: {e}")
        # Fallback to safe defaults
    
    valid_warm_start = {k: v for k, v in self.warm_start_params.items() 
                       if k in self.param_ranges}
    if valid_warm_start:
        study.enqueue_trial(valid_warm_start)
```

---

## 📊 Realistic Parameter Ranges for ALL Strategies

### Common Parameters (All Strategies)
```python
common = {
    "trailing_stop_activation_pct": {"min": 0.002, "max": 0.03,  "step": 0.001, "default": 0.005},
    "min_conditions_satisfied":     {"min": 2,    "max": 5,     "step": 1,     "default": 2,   "type": "int"},
    "min_score_threshold":          {"min": 1.0,  "max": 6.0,   "step": 0.25,  "default": 2.5},
    "base_position_pct":            {"min": 0.02, "max": 0.20,  "step": 0.01,  "default": 0.10},
}
```

### Absorption Strategy
```python
"absorption": {
    "abs__entry_str_min":       {"min": 0.2,  "max": 0.8,  "step": 0.05,  "default": 0.45},
    "abs__entry_vol_min":       {"min": 0.5,  "max": 3.0,  "step": 0.1,   "default": 1.0},
    "abs__entry_chg60_max":     {"min": 0.0005, "max": 0.005, "step": 0.0005, "default": 0.002},
    "abs__entry_delta_min":     {"min": 0,    "max": 5000, "step": 100,   "default": 0,    "type": "int"},
    "abs__entry_imbal_min":     {"min": 0.02, "max": 0.40, "step": 0.01,  "default": 0.08},
    "abs__entry_poc_range":     {"min": 0.001, "max": 0.015, "step": 0.001, "default": 0.006},
    "abs__filter_spread_max":   {"min": 5.0,  "max": 50.0, "step": 1.0,   "default": 15.0},
    "abs__filter_bid_min":      {"min": 1000.0, "max": 20000.0, "step": 500.0, "default": 2000.0},
    "abs__filter_ask_min":      {"min": 1000.0, "max": 20000.0, "step": 500.0, "default": 2000.0},
    "abs__filter_chg300_range": {"min": 0.003, "max": 0.02, "step": 0.001, "default": 0.006},
}
```

### Delta Divergence Strategy
```python
"delta_divergence": {
    "div__entry_delta_cum_min":  {"min": 500,  "max": 5000, "step": 100,   "default": 2000},
    "div__entry_price_chg_max":  {"min": 0.001, "max": 0.01, "step": 0.001, "default": 0.003},
    "div__filter_vol_ratio_min": {"min": 1.5,  "max": 5.0,  "step": 0.1,   "default": 2.5},
    # ... etc
}
```

---

## ⚡ Quick Win: Reduced Trial Count Strategy

Instead of 200+ trials over 10 days:

1. **Use 50 trials with tight ranges** → ~1-2 days
2. **Validate top 5 params on OOS data**
3. **If Sharpe > 1.0, run 100 more trials with refined ranges**
4. **Total time: 3-4 days instead of 10+**

```python
# Phase 1: Exploration (tight ranges, 50 trials)
result1 = optimizer.optimize(backtest_fn, n_trials=50, objective_type="robust")

# Analyze top 5
top_params = analyze_top_trials(result1.all_trials, n=5)

# Phase 2: Exploitation (narrow around best, 100 trials)
refined_ranges = narrow_ranges_around(top_params[0], width=0.2)
optimizer.param_ranges = refined_ranges
result2 = optimizer.optimize(backtest_fn, n_trials=100, objective_type="sharpe")
```

---

## 🎯 Recommended Action Plan

### Immediate (Today):
1. ✅ Apply Fix #1 (error logging)
2. ✅ Apply Fix #2 (realistic ranges)
3. ✅ Apply Fix #3 (success rate monitoring)
4. Run 20-trial test to verify backtests work

### Short-term (2-3 days):
5. Run 50-trial optimization with new ranges
6. Analyze success rate and top params
7. If success rate > 50%, proceed to 100 trials

### Medium-term (1 week):
8. Walk-forward validation on top params
9. Apply same range tightening to other strategies
10. Document optimal ranges per strategy/asset

---

## 📝 Key Insight

**The optimizer is NOT broken — the parameter ranges are.** 

Your 10-day optimization with all-default results is a classic sign of:
- Backtests failing for most parameter combinations
- Optimizer giving up and returning warm-start defaults
- No visibility into failure modes (fixed by logging)

**Tight, realistic ranges + error logging = 10x faster optimization with actual results.**
