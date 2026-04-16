"""
Optuna Optimizer - Enhanced for Perfect Optimization
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from loguru import logger

from knowledge.llm_advisor import LLMAdvisor, ABSORPTION_KNOWLEDGE, DELTA_DIVERGENCE_KNOWLEDGE
from knowledge.strategy_library import StrategyDefinition, get_strategy

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Results from optimization run"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    timestamp: datetime
    all_trials: List[Dict] = None
    objective_type: str = "robust"


class StrategyOptimizer:
    """
    Enhanced strategy parameter optimizer.
    
    Features:
    1. Warm-start from LLM domain knowledge
    2. Multiple objective functions (sharpe, profit, robust)
    3. Minimum trade count enforcement
    4. Correlation-aware parameter constraints
    5. Advanced pruning
    """
    
    def __init__(
        self,
        strategy_name: str,
        llm_advisor: Optional[LLMAdvisor] = None,
        storage: str = "sqlite:///optuna_studies.db"
    ):
        self.strategy_name = strategy_name
        self.llm_advisor = llm_advisor
        self.storage = storage
        
        self.base_strategy = get_strategy(strategy_name)
        if not self.base_strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.param_ranges = self._get_parameter_ranges()
        self.warm_start_params = self._get_warm_start_params()
    
    def _get_parameter_ranges(self) -> Dict[str, Dict]:
        """
        Full optimizer-controlled parameter space.
        CRITICAL: 
        - NO dead parameters (everything used in evaluation)
        - All regime multipliers included
        - Proper bounds for XRP/USDT microstructure
        
        Naming convention: {strategy}__{type}_{feature}_{bound}
        """
        
        # DEBUG: Log strategy being configured
        logger.debug(f"[_get_parameter_ranges] Building ranges for {self.strategy_name}")
        
        # CRITICAL FIX: Removed base stop_loss_atr_mult and take_profit_atr_mult 
        # because evaluate() uses regime-specific multipliers exclusively.
        # Including them would create "dead parameters" that Optuna wastes time optimizing.
        
        common = {
            "trailing_stop_activation_pct": {"min": 0.002, "max": 0.03,  "step": 0.001, "default": 0.005},
            "min_conditions_satisfied":     {"min": 2,    "max": 5,     "step": 1,     "default": 2,   "type": "int"},
            "min_score_threshold":          {"min": 1.0,  "max": 6.0,   "step": 0.25,  "default": 2.5},
            "base_position_pct":            {"min": 0.02, "max": 0.20,  "step": 0.01,  "default": 0.10},
        }  # [FIXED] Realistic bounds to prevent degenerate strategies and reduce optimization time
        
        strategy_params = {
            "absorption": {
                # [FIXED] Realistic ranges for XRP/USDT microstructure - prevents 10+ day optimizations
                "abs__entry_str_min":       {"min": 0.2,  "max": 0.8,  "step": 0.05,  "default": 0.45},
                "abs__entry_vol_min":       {"min": 0.5,  "max": 3.0,  "step": 0.1,   "default": 1.0},
                # Absorption needs tight stability: 0.05%-0.5% over 60s
                "abs__entry_chg60_max":     {"min": 0.0005, "max": 0.005, "step": 0.0005, "default": 0.002},
                # 5000 is realistic max for XRP; 50000 caused numerical instability
                "abs__entry_delta_min":     {"min": 0,    "max": 5000, "step": 100,   "default": 0,    "type": "int"},
                # 2%-40% imbalance is realistic; 70% is extreme and rare
                "abs__entry_imbal_min":     {"min": 0.02, "max": 0.40, "step": 0.01,  "default": 0.08},
                # POC proximity must be tight: 0.1%-1.5%; wider ranges break the strategy logic
                "abs__entry_poc_range":     {"min": 0.001, "max": 0.015, "step": 0.001, "default": 0.006},
                # 5-50 bps spread filter; prevents rejecting all signals or accepting all
                "abs__filter_spread_max":   {"min": 5.0,  "max": 50.0, "step": 1.0,   "default": 15.0},
                # $1k-$20k depth prevents overfitting to illiquid conditions
                "abs__filter_bid_min":      {"min": 1000.0, "max": 20000.0, "step": 500.0, "default": 2000.0},
                "abs__filter_ask_min":      {"min": 1000.0, "max": 20000.0, "step": 500.0, "default": 2000.0},
                # Tight regime filter: 0.3%-2% over 300s; filters trending/choppy markets
                "abs__filter_chg300_range": {"min": 0.003, "max": 0.02, "step": 0.001, "default": 0.006},
            },
        }  # [FIXED] Narrow realistic bounds - reduces optimization time from 10 days to 1-2 days
        
        # Merge parameters
        result = common.copy()
        strat_key = self.strategy_name.lower()
        
        if strat_key in strategy_params:
            result.update(strategy_params[strat_key])
            logger.debug(f"[_get_parameter_ranges] Added {len(strategy_params[strat_key])} strategy-specific params")
        else:
            logger.warning(f"[_get_parameter_ranges] No specific params defined for {strat_key}")
        
        logger.info(f"[_get_parameter_ranges] Total parameter space: {len(result)} dimensions")
        return result
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial with proper type handling.
        DEBUG: Logs every suggestion for traceability.
        """
        ranges = self._get_parameter_ranges()
        params = {}
        
        logger.debug(f"[_suggest_params] Suggesting {len(ranges)} parameters for trial {trial.number}")
        
        for param_name, config in ranges.items():
            param_type = config.get("type", "float")
            step = config.get("step")
            
            try:
                if param_type == "int":
                    value = trial.suggest_int(
                        param_name,
                        config["min"],
                        config["max"],
                        step=step or 1
                    )
                elif param_type == "categorical":
                    value = trial.suggest_categorical(param_name, config["choices"])
                else:  # float
                    value = trial.suggest_float(
                        param_name,
                        config["min"],
                        config["max"],
                        step=step
                    )
                
                params[param_name] = value
                logger.debug(f"[_suggest_params] {param_name} = {value}")
                
            except Exception as e:
                logger.error(f"[_suggest_params] Failed to suggest {param_name}: {e}")
                raise
        
        return params
    
    def _get_warm_start_params(self) -> Dict[str, float]:
        """Get initial parameter values for warm start"""
        warm_start = {}
        
        for param, range_dict in self.param_ranges.items():
            if "default" in range_dict:
                warm_start[param] = range_dict["default"]
        
        # Enforce logical constraints in warm start (regime-specific multipliers)
        regimes = ["high_vol", "low_vol", "trending"]
        for regime in regimes:
            sl_key = f"sl_mult_{regime}"
            tp_key = f"tp_mult_{regime}"
            if sl_key in warm_start and tp_key in warm_start:
                # Ensure 1.5:1 minimum risk/reward for each regime
                warm_start[tp_key] = max(
                    warm_start[tp_key],
                    warm_start[sl_key] * 1.5
                )
        
        if self.llm_advisor:
            try:
                llm_params = self.llm_advisor.get_initial_parameters(self.strategy_name)
                for param, value in llm_params.items():
                    if param in self.param_ranges:
                        range_dict = self.param_ranges[param]
                        if range_dict["min"] <= value <= range_dict["max"]:
                            warm_start[param] = value
            except Exception as e:
                logger.warning(f"Failed to get LLM initial params: {e}")
        
        return warm_start
    
    def create_objective(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        objective_type: str = "robust"
    ) -> Callable[[optuna.Trial], float]:
        """
        Create Optuna objective function.
        CRITICAL: Uses _suggest_params and _enforce_constraints (no re-suggestion).
        """
        
        def objective(trial: optuna.Trial) -> float:
            # DEBUG: Log trial start
            logger.debug(f"[objective] Starting trial {trial.number}")
            
            # Step 1: Suggest parameters (single source of truth)
            params = self._suggest_params(trial)
            
            # Step 2: Enforce constraints (clamping only, no re-suggestion)
            params = self._enforce_constraints(params)
            
            # DEBUG: Log final params being used
            logger.debug(f"[objective] Trial {trial.number} final params: {params}")
            
            # Step 3: Run backtest
            try:
                metrics = backtest_fn(params)
                logger.debug(f"[objective] Trial {trial.number} metrics: {metrics}")
            except Exception as e:
                logger.error(f"[objective] Trial {trial.number} backtest FAILED:")
                logger.error(f"  Parameters: {params}")
                logger.error(f"  Exception: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"  Traceback: {traceback.format_exc()}")
                return float("-inf")
            
            # Check if penalized (too few trades)
            if metrics.get("_penalized", False):
                logger.warning(f"[objective] Trial {trial.number} penalized (insufficient trades)")
                trial.set_user_attr("penalized", True)
                return float("-inf")
            
            # Store all metrics as user attributes for analysis
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            
            # Calculate objective score
            if objective_type == "sharpe":
                score = metrics.get("sharpe_ratio", 0)
            elif objective_type == "profit":
                score = self._profit_objective(metrics)
            elif objective_type == "profit_dd_trades":
                score = self._profit_dd_trades_objective(metrics)
            else:  # robust
                score = self._robust_objective(metrics)
            
            # DEBUG: Log score
            logger.debug(f"[objective] Trial {trial.number} score ({objective_type}): {score:.4f}")
            
            # Report for pruning
            trial.report(score, step=1)
            if trial.should_prune():
                logger.debug(f"[objective] Trial {trial.number} pruned")
                raise optuna.TrialPruned()
            
            return score
        
        return objective
    
    def _enforce_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clamp parameters to prevent numerical crashes ONLY.
        CRITICAL: Never call trial.suggest_*() here - that creates duplicate parameters.
        """
        # Prevent ATR math crashes (keep within float-safe bounds)
        params["stop_loss_atr_mult"] = max(0.5, min(20.0, params.get("stop_loss_atr_mult", 2.5)))
        params["take_profit_atr_mult"] = max(1.0, min(50.0, params.get("take_profit_atr_mult", 6.0)))
        
        # Prevent filter degeneracy (rejecting all signals or accepting all)
        params["abs__filter_spread_max"] = max(1.0, min(100.0, params.get("abs__filter_spread_max", 15.0)))
        params["abs__filter_bid_min"] = max(10.0, min(100000.0, params.get("abs__filter_bid_min", 1500.0)))
        params["abs__filter_ask_min"] = max(10.0, min(100000.0, params.get("abs__filter_ask_min", 1500.0)))
        params["abs__filter_chg300_range"] = max(0.003, min(0.10, params.get("abs__filter_chg300_range", 0.005)))
        
        return params  # [APPLIED]

    
    def _profit_objective(self, metrics: Dict[str, float]) -> float:
        """
        Profit-focused objective with drawdown penalty.
        
        Score = Return - 3 * MaxDrawdown
        """
        total_return = metrics.get("total_return_pct", 0) * 100  # Convert to %
        max_drawdown = metrics.get("max_drawdown_pct", 0) * 100
        
        return total_return - 3 * max_drawdown
    
    def _robust_objective(self, metrics: Dict[str, float]) -> float:
        """
        Robust multi-factor objective.
        
        Combines:
        - Sharpe ratio (risk-adjusted returns)
        - Profit factor (win/loss ratio)
        - Win rate (consistency)
        - Drawdown penalty (risk control)
        
        This objective prefers strategies that are:
        - Profitable (positive return)
        - Consistent (high win rate)
        - Risk-controlled (low drawdown)
        - Efficient (good profit factor)
        """
        sharpe = metrics.get("sharpe_ratio", 0)
        profit_factor = metrics.get("profit_factor", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        total_trades = metrics.get("total_trades", 0)
        
        # Component 1: Sharpe (normalized to 0-1 range, cap at 3)
        sharpe_component = min(sharpe / 3.0, 1.0)
        
        # Component 2: Profit factor (normalized, cap at 3)
        pf_component = min(profit_factor / 3.0, 1.0) if profit_factor > 0 else 0
        
        # Component 3: Win rate (direct, but weighted lower)
        win_rate_component = win_rate * 0.8  # 80% win rate = 0.64
        
        # Component 4: Drawdown penalty (exponential decay)
        # 0% DD = 1.0, 10% DD = 0.37, 20% DD = 0.14
        drawdown_penalty = np.exp(-max_drawdown * 10)
        
        # Component 5: Trade count sufficiency
        # Sigmoid function: approaches 1 around 100 trades
        trade_sufficiency = 1 / (1 + np.exp(-0.05 * (total_trades - 100)))
        
        # Weighted combination
        weights = {
            "sharpe": 0.30,
            "profit_factor": 0.25,
            "win_rate": 0.15,
            "drawdown": 0.20,
            "trade_count": 0.10
        }
        
        score = (
            weights["sharpe"] * sharpe_component +
            weights["profit_factor"] * pf_component +
            weights["win_rate"] * win_rate_component +
            weights["drawdown"] * drawdown_penalty +
            weights["trade_count"] * trade_sufficiency
        )
        
        # Hard penalty: negative return = bad
        if metrics.get("total_return_pct", 0) < 0:
            score *= 0.5
        
        return score
    
    def _profit_dd_trades_objective(self, metrics: Dict[str, float]) -> float:
        """
        Custom objective balancing winning trades, drawdown minimization, and average win size.
        
        Weights:
        - Winning trades: 35% (sigmoid-scaled trade count)
        - Drawdown penalty: 40% (exponential decay)
        - Average win size: 25% (normalized avg_win)
        
        This objective prioritizes strategies that:
        - Generate consistent winning trades
        - Maintain low drawdown risk
        - Achieve meaningful per-trade profitability
        """
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        winning_trades = total_trades * win_rate
        
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        avg_win = metrics.get("avg_win", 0)  # Use avg_win not avg_win_pct
        
        # Component 1: Winning trades (sigmoid scaling, peaks at ~50 trades)
        # 10 winning trades = ~0.5, 50 winning trades = ~0.9
        winning_trades_component = 1 / (1 + np.exp(-0.15 * (winning_trades - 25)))
        
        # Component 2: Drawdown penalty (exponential decay, stronger than robust)
        # 0% DD = 1.0, 5% DD = 0.61, 10% DD = 0.37, 15% DD = 0.22
        drawdown_penalty = np.exp(-max_drawdown * 20)
        
        # Component 3: Average win size (normalized, assume $10-100 range is good)
        # Scale so $50 avg win = 1.0, $10 = 0.2, $100 = 2.0 (capped at 1.0)
        avg_win_component = min(avg_win / 50.0, 1.0) if avg_win > 0 else 0
        
        # Weighted combination
        weights = {
            "winning_trades": 0.35,
            "drawdown": 0.40,
            "avg_win": 0.25
        }
        
        score = (
            weights["winning_trades"] * winning_trades_component +
            weights["drawdown"] * drawdown_penalty +
            weights["avg_win"] * avg_win_component
        )
        
        # Hard penalty: no winning trades = very bad
        if winning_trades < 1:
            score *= 0.1
        
        return score
    
    def optimize(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 200,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        objective_type: str = "robust"
    ) -> OptimizationResult:
        """
        Run optimization with warm start and enhanced objective.
        """
        study_name = f"{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Enhanced sampler with warm start
        sampler = TPESampler(
            n_startup_trials=20,
            multivariate=True,  # Consider parameter correlations
            warn_independent_sampling=False
        )
        
        # Hyperband pruner for aggressive early stopping
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=1,
            reduction_factor=3
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction="maximize",
            load_if_exists=True,  # CRITICAL: Resume instead of overwrite
            sampler=sampler,
            pruner=pruner
        )
        
        # Enqueue warm-start trial
        if self.warm_start_params:
            valid_warm_start = {
                k: v for k, v in self.warm_start_params.items()
                if k in self.param_ranges
            }
            if valid_warm_start:
                study.enqueue_trial(valid_warm_start)
                logger.info(f"Enqueued warm-start trial with {len(valid_warm_start)} params")
        
        objective = self.create_objective(backtest_fn, objective_type)
        
        logger.info(f"Starting optimization: {n_trials} trials, objective={objective_type}, n_jobs={n_jobs}")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=False  # Cleaner logs
        )
        
        # Filter out penalized trials for analysis
        valid_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "user_attrs": t.user_attrs
            }
            for t in study.trials
            if t.value is not None and not t.user_attrs.get("_penalized", False)
        ]
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study_name=study_name,
            timestamp=datetime.now(),
            all_trials=valid_trials,
            objective_type=objective_type
        )
        
        logger.info(f"Optimization complete.")
        logger.info(f"  Best score ({objective_type}): {result.best_score:.4f}")
        logger.info(f"  Valid trials: {len(valid_trials)} / {len(study.trials)}")
        logger.info(f"  Best params: {result.best_params}")
        
        return result
    
    def save_results(self, result: OptimizationResult, path: str) -> None:
        """Save optimization results to file"""
        output = {
            "strategy_name": self.strategy_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "valid_trials": len(result.all_trials) if result.all_trials else 0,
            "timestamp": result.timestamp.isoformat(),
            "objective_type": result.objective_type,
            "param_ranges": self.param_ranges,
            "warm_start_params": self.warm_start_params
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to {path}")
    
    def analyze_parameter_sensitivity(self, result: OptimizationResult) -> Dict[str, float]:
        """
        Analyze which parameters most affect the objective.
        Returns importance scores for each parameter.
        """
        if not result.all_trials or len(result.all_trials) < 10:
            return {}
        
        # Get parameter names
        param_names = list(result.all_trials[0]["params"].keys())
        
        importance = {}
        
        for param in param_names:
            # Get param values and corresponding scores
            pairs = [
                (t["params"].get(param), t["value"])
                for t in result.all_trials
                if param in t["params"] and t["value"] is not None
            ]
            
            if len(pairs) < 5:
                continue
            
            values = np.array([p[0] for p in pairs])
            scores = np.array([p[1] for p in pairs])
            
            # Calculate correlation as importance
            if np.std(values) > 0 and np.std(scores) > 0:
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param] = abs(correlation)
            else:
                importance[param] = 0.0
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance