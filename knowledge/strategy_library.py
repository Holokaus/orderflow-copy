"""
Strategy Library
Complete definitions of order flow trading strategies.
Each strategy encapsulates domain knowledge in executable form.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import numpy as np
from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Regime
)

class StrategyCategory(Enum):
    ABSORPTION = auto()
    MOMENTUM = auto()
    REVERSAL = auto()
    BREAKOUT = auto()
    MEAN_REVERSION = auto()


@dataclass
class StrategyCondition:
    """Single condition for strategy entry/exit with optimizer parameter mapping"""
    feature: str
    operator: str  # ">", "<", ">=", "<=", "==", "between"
    threshold: float
    threshold_high: Optional[float] = None  # For "between" operator
    weight: float = 1.0
    required: bool = False
    param_key: Optional[str] = None  # CRITICAL: Links to Optuna parameter name
    
    def validate(self) -> bool:
        """Validate condition configuration before runtime"""
        if self.operator == "between":
            if self.threshold_high is None:
                raise ValueError(f"[{self.feature}] 'between' operator requires threshold_high")
            if self.threshold >= self.threshold_high:
                raise ValueError(f"[{self.feature}] threshold ({self.threshold}) must be < threshold_high ({self.threshold_high})")
        # DEBUG: Log validation success
        logger.debug(f"[StrategyCondition] Validated: {self.feature} (operator={self.operator})")
        return True
    
    def evaluate(self, features: Dict[str, float]) -> tuple:
        """Returns (satisfied: bool, score: float)"""
        if self.feature not in features:
            logger.debug(f"[StrategyCondition] Feature {self.feature} not found in features")
            return (False, 0.0)
        
        value = features[self.feature]
        
        if self.operator == ">":
            satisfied = value > self.threshold
        elif self.operator == "<":
            satisfied = value < self.threshold
        elif self.operator == ">=":
            satisfied = value >= self.threshold
        elif self.operator == "<=":
            satisfied = value <= self.threshold
        elif self.operator == "==":
            satisfied = abs(value - self.threshold) < 1e-9
        elif self.operator == "between":
            satisfied = self.threshold <= value <= self.threshold_high
        else:
            satisfied = False
        
        # Calculate score (how far beyond threshold)
        if satisfied and self.operator in [">", ">="]:
            score = min((value - self.threshold) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied and self.operator in ["<", "<="]:
            score = min((self.threshold - value) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied:
            score = 1.0
        else:
            score = 0.0
        
        logger.debug(f"[StrategyCondition] {self.feature}: value={value:.4f}, threshold={self.threshold}, satisfied={satisfied}")
        return (satisfied, score * self.weight)


@dataclass
class StrategyDefinition:
    """Complete strategy definition with optimizer-controlled parameters"""
    name: str
    category: StrategyCategory
    description: str
    
    entry_conditions: List[StrategyCondition] = field(default_factory=list)
    min_conditions_satisfied: int = 3
    min_score_threshold: float = 2.0
    
    # Risk parameters (base values - may be overridden by optimizer)
    stop_loss_atr_mult: float = 2.5
    take_profit_atr_mult: float = 6.0
    max_holding_seconds: int = 86400  # [APPLIED] 24h = unlimited hold time, optimizer controls exits via SL/TP only
    trailing_stop_activation_pct: float = 0.005
    
    # CRITICAL: Regime-specific multipliers (optimizer-controlled)
    # These allow different risk parameters for different market conditions
    sl_mult_high_vol: float = 3.5
    sl_mult_low_vol: float = 1.8
    sl_mult_trending: float = 2.5
    tp_mult_high_vol: float = 7.0
    tp_mult_low_vol: float = 2.5
    tp_mult_trending: float = 5.0
    
    filters: List[StrategyCondition] = field(default_factory=list)
    allowed_regimes: List[Regime] = field(default_factory=lambda: list(Regime))
    
    # ML ensemble (optional, set externally)
    ml_ensemble: Optional[object] = None
    ml_min_confidence: float = 0.7

    # ML ensemble lazy loading state (not a dataclass field, set in __post_init__)
    _ml_available: bool = False
    _ml_loaded: bool = False

    def __post_init__(self) -> None:
        """Post-initialization to set non-dataclass defaults"""
        self._ml_available = False
        self._ml_loaded = False

    def _load_ml_ensemble(self) -> None:
        """Lazy-load ML ensemble module only when accessed"""
        if self._ml_loaded:
            return
        self._ml_loaded = True
        try:
            from prediction.ml_ensemble import MLEnsemble
            self._ml_available = True
            logger.debug("[StrategyDefinition] ML ensemble module loaded")
        except ImportError:
            self._ml_available = False
            logger.debug("[StrategyDefinition] ML ensemble module not available (optional)")

    # Position sizing
    base_position_pct: float = 0.1
    max_position_pct: float = 0.25
    scale_with_score: bool = True
    
    def evaluate(self, state: OrderFlowState) -> Optional[Signal]:
        """
        Evaluate strategy conditions and return signal if triggered.
        CRITICAL: Uses regime-specific multipliers controlled by optimizer.
        """
        features = state.features
        
        # DEBUG: Log regime check
        logger.debug(f"[{self.name}] Evaluating - Regime: {state.regime}, Features: {len(features)}")
        
        # Check for LOW_LIQUIDITY regime (hard reject)
        if state.regime == Regime.LOW_LIQUIDITY:
            logger.debug(f"[{self.name}] REJECTED: LOW_LIQUIDITY regime")
            return None
        
        # Validate all conditions before evaluation
        try:
            for cond in self.entry_conditions + self.filters:
                cond.validate()
        except ValueError as e:
            logger.error(f"[{self.name}] Validation error: {e}")
            return None
        
        # Check filters first (if satisfied, REJECT the trade)
        for i, filter_cond in enumerate(self.filters):
            satisfied, _ = filter_cond.evaluate(features)
            if satisfied:
                logger.debug(f"[{self.name}] FILTER REJECTED by condition {i}: {filter_cond.feature}")
                return None
        
        # Check allowed regimes
        if state.regime not in self.allowed_regimes:
            logger.debug(f"[{self.name}] REJECTED: Regime {state.regime} not in allowed list")
            return None
        
        # Evaluate entry conditions
        satisfied_count = 0
        total_score = 0.0
        required_satisfied = True
        reasons = []
        failed_conditions = []
        
        for condition in self.entry_conditions:
            satisfied, score = condition.evaluate(features)
            
            if satisfied:
                satisfied_count += 1
                total_score += score
                reasons.append(f"{condition.feature} {condition.operator} {condition.threshold:.4f}")
                logger.debug(f"[{self.name}] Condition SATISFIED: {condition.feature}")
            else:
                failed_conditions.append(f"{condition.feature} ({condition.operator} {condition.threshold})")
                if condition.required:
                    required_satisfied = False
        
        # DEBUG: Log evaluation summary
        logger.debug(f"[{self.name}] Stats: satisfied={satisfied_count}/{len(self.entry_conditions)}, "
                    f"score={total_score:.2f}, required_ok={required_satisfied}")
        
        # Check entry criteria
        if not required_satisfied:
            logger.debug(f"[{self.name}] REJECTED: Required condition failed")
            return None
        
        if satisfied_count < self.min_conditions_satisfied:
            logger.debug(f"[{self.name}] REJECTED: Only {satisfied_count} conditions, need {self.min_conditions_satisfied}")
            return None
        
        if total_score < self.min_score_threshold:
            logger.debug(f"[{self.name}] REJECTED: Score {total_score:.2f} < threshold {self.min_score_threshold}")
            return None
        
        # Determine direction
        direction = self._determine_direction(state, total_score)
        if direction == SignalType.NEUTRAL:
            logger.debug(f"[{self.name}] REJECTED: No clear direction (NEUTRAL)")
            return None
        
        # CRITICAL: Select regime-specific multipliers
        regime = state.regime
        if regime == Regime.HIGH_VOLATILITY:
            sl_mult, tp_mult = self.sl_mult_high_vol, self.tp_mult_high_vol
            logger.debug(f"[{self.name}] Using HIGH_VOL multipliers: SL={sl_mult}, TP={tp_mult}")
        elif regime in (Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION):
            sl_mult, tp_mult = self.sl_mult_low_vol, self.tp_mult_low_vol
            logger.debug(f"[{self.name}] Using LOW_VOL multipliers: SL={sl_mult}, TP={tp_mult}")
        else:  # TRENDING_UP, TRENDING_DOWN, BREAKOUT, etc.
            sl_mult, tp_mult = self.sl_mult_trending, self.tp_mult_trending
            logger.debug(f"[{self.name}] Using TRENDING multipliers: SL={sl_mult}, TP={tp_mult}")
        
        # Calculate stops using selected multipliers
        atr = self._estimate_atr(state)
        mid_price = state.order_book.mid_price
        
        stop_dist = atr * sl_mult
        tp_dist = atr * tp_mult
        
        # Enforce minimum TP distance (0.37% of price)
        min_tp_dist = mid_price * 0.0037
        if tp_dist < min_tp_dist:
            logger.debug(f"[{self.name}] TP distance adjusted from {tp_dist:.2f} to {min_tp_dist:.2f} (min 0.37%)")
            tp_dist = min_tp_dist
        
        if direction in (SignalType.BUY, SignalType.STRONG_BUY):
            stop_loss = mid_price - stop_dist
            take_profit = mid_price + tp_dist
        else:
            stop_loss = mid_price + stop_dist
            take_profit = mid_price - tp_dist
        
        # Calculate position size
        position_pct = self.base_position_pct
        if self.scale_with_score:
            position_pct = min(self.base_position_pct * min(total_score / self.min_score_threshold, 2.0), 
                             self.max_position_pct)
        
        confidence = min(total_score / (self.min_score_threshold * 2), 1.0)

        # ML ensemble check (optional, lazy-loaded)
        self._load_ml_ensemble()
        if self.ml_ensemble is not None and self._ml_available:
            try:
                # Extract features from state for ML prediction
                ml_features = self._extract_ml_features(state)
                prediction = self.ml_ensemble.predict(
                    ml_features,
                    confidence_threshold=self.ml_min_confidence
                )

                # Check ML confidence threshold
                if prediction.confidence < self.ml_min_confidence:
                    logger.debug(
                        f"[{self.name}] ML REJECTED: confidence {prediction.confidence:.2f} "
                        f"< {self.ml_min_confidence:.2f}"
                    )
                    return None

                # Check ML direction agrees with rule-based signal
                ml_direction = "BUY" if direction in (SignalType.BUY, SignalType.STRONG_BUY) else "SELL"
                if prediction.decision != ml_direction:
                    logger.debug(
                        f"[{self.name}] ML REJECTED: ML says {prediction.decision}, "
                        f"rule says {ml_direction}"
                    )
                    return None

                logger.debug(
                    f"[{self.name}] ML APPROVED: {prediction.decision} @ "
                    f"confidence={prediction.confidence:.2f}"
                )
            except Exception as e:
                logger.warning(f"[{self.name}] ML ensemble error (proceeding without ML): {e}")

        # DEBUG: Log signal generation
        logger.info(f"[{self.name}] SIGNAL: {direction.name} @ {mid_price:.4f} | "
                   f"SL={stop_loss:.4f} TP={take_profit:.4f} | "
                   f"Regime={regime.name} | Score={total_score:.2f}")
        
        return Signal(
            timestamp=state.timestamp,
            signal_type=direction,
            confidence=confidence,
            entry_price=mid_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_pct,
            primary_reason=reasons[0] if reasons else self.name,
            supporting_factors=reasons[1:5],
            risk_reward_ratio=abs(tp_dist) / abs(stop_dist)
        )
    
    def _determine_direction(self, state: OrderFlowState, score: float) -> SignalType:
        """Determine signal direction from state"""
        features = state.features
        
        # Use delta and imbalance to determine direction
        delta = features.get("delta_60s", 0)
        imbalance = features.get("depth_imbalance_10", 0)
        pressure = features.get("net_pressure", 0)
        
        bullish_score = sum([
            delta > 0,
            imbalance > 0.1,  # Preserve the existing noise filter threshold
            pressure > 0
        ])
        
        if bullish_score == 3:
            return SignalType.STRONG_BUY if score > self.min_score_threshold * 1.5 else SignalType.BUY
        elif bullish_score == 2:
            return SignalType.BUY
        elif bullish_score == 1:
            return SignalType.NEUTRAL # Ambiguous — 1 of 3 indicators bullish, no clear edge
        elif bullish_score == 0:
            # Use score parameter for downside signals too (was unused for 0/3 case)
            return SignalType.STRONG_SELL if score > self.min_score_threshold * 1.5 else SignalType.SELL
        
    
    def _estimate_atr(self, state: OrderFlowState, default: float = 100.0) -> float:
        """Estimate ATR for stop/TP placement.
        
        Priority order:
          1. Use atr_60s from FeatureEngine if available (most accurate)
          2. Use price_range_60s as a proxy if atr_60s missing
          3. Fall back to 0.5% of mid price (last resort)
        
        Always enforce a minimum of 0.5% of price so stops are never
        tighter than the spread.
        """
        mid_price = state.order_book.mid_price
        if mid_price <= 0:
            return default

        features = state.features

        # Priority 1: Real ATR from feature engine  # [FIXED]
        if features.get("atr_60s", 0) > 0:
            return max(features["atr_60s"], mid_price * 0.005)

        # Priority 2: Price range as ATR proxy  # [FIXED]
        if features.get("price_range_60s", 0) > 0:
            return max(features["price_range_60s"], mid_price * 0.005)

        # Priority 3: Fallback — 0.5% of price
        return mid_price * 0.005

    def _extract_ml_features(self, state: OrderFlowState) -> dict:
        """
        Extract features from OrderFlowState for ML ensemble prediction.
        Converts FeatureEngine output into the format expected by ML models.

        Returns a dict with numeric feature values.
        """
        features = state.features
        if not features:
            return {}

        ml_features = {}

        # Price-derived features
        for key in ['mid_price', 'spread_bps', 'price_change_pct_60s',
                     'price_change_pct_300s', 'price_range_60s', 'price_range_300s',
                     'price_vs_vwap_pct', 'price_vs_poc_pct']:
            if key in features:
                ml_features[key] = features[key]

        # Volume features
        for key in ['volume_acceleration', 'volume_30s', 'volume_60s',
                     'volume_300s', 'trade_count_60s', 'trade_intensity_60s']:
            if key in features:
                ml_features[key] = features[key]

        # Delta / order flow
        for key in ['delta_60s', 'delta_300s', 'abs_delta_60s', 'delta_pct_60s',
                     'delta_pct_300s', 'cvd_60s', 'cvd_300s']:
            if key in features:
                ml_features[key] = features[key]

        # Depth / book features
        for key in ['depth_imbalance_10', 'depth_imbalance_20',
                     'bid_depth_10', 'ask_depth_10', 'net_pressure',
                     'slope_asymmetry', 'book_trade_agreement']:
            if key in features:
                ml_features[key] = features[key]

        # Technical features
        for key in ['atr_60s', 'atr_300s', 'exhaustion_score',
                     'buying_exhaustion', 'selling_exhaustion',
                     'in_value_area', 'va_breakout_potential',
                     'footprint_imbalance_count', 'pressure_confirmed']:
            if key in features:
                ml_features[key] = features[key]

        logger.debug(f"[_extract_ml_features] Extracted {len(ml_features)} features")
        return ml_features


# ==================== PRE-DEFINED STRATEGIES ====================

def create_absorption_strategy() -> StrategyDefinition:
    """
    Absorption Strategy with full optimizer control.
    CRITICAL: All thresholds mapped via param_key for optimization.
    """
    return StrategyDefinition(
        name="Absorption",
        category=StrategyCategory.ABSORPTION,
        description="Trade after detecting absorption of aggressive orders",
        
        entry_conditions=[
            # Core absorption detection
            StrategyCondition(
                feature="recent_absorption_strength",
                operator=">=",
                threshold=0.30,
                weight=2.0,
                required=True,
                param_key="abs__entry_str_min"  # Optimizer controls this threshold
            ),
            StrategyCondition(
                feature="volume_acceleration",
                operator=">",
                threshold=1.0,
                weight=1.5,
                param_key="abs__entry_vol_min"
            ),
            StrategyCondition(
                feature="price_change_pct_60s",
                operator="<",
                threshold=0.001,
                weight=1.0,
                param_key="abs__entry_chg60_max"
            ),
            StrategyCondition(
                feature="abs_delta_60s",
                operator=">",
                threshold=0,
                weight=1.5,
                param_key="abs__entry_delta_min"
            ),
            StrategyCondition(
                feature="depth_imbalance_10",
                operator=">",
                threshold=0.05,
                weight=1.0,
                param_key="abs__entry_imbal_min"
            ),
            # POC proximity - symmetric range (will use abs(value) for both bounds)
            StrategyCondition(
                feature="price_vs_poc_pct",
                operator="between",
                threshold=-0.005,  # Will be overridden by optimizer via param_key
                threshold_high=0.005,
                weight=0.5,
                param_key="abs__entry_poc_range"
            ),
            StrategyCondition(
                feature="book_trade_agreement",
                operator="==",
                threshold=1.0,
                weight=1.5,       # High weight — strongly preferred but not required
                required=False,   # NOT required: early absorption entries lag trade flow
                                  # Making it required would filter out the best entries
                param_key="abs__entry_agreement"   # Exposed to optimizer  # [FIXED]
            ),
        ],
        
        filters=[
            # Spread filter: REJECT if spread > threshold
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=15.0,
                param_key="abs__filter_spread_max"
            ),
            # Depth filters: REJECT if depth < threshold (illiquidity protection)
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=1500.0,
                param_key="abs__filter_bid_min"
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=1500.0,
                param_key="abs__filter_ask_min"
            ),
            # Price change filter: REJECT if outside symmetric range (dead/choppy market protection)
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.005,
                threshold_high=0.005,
                param_key="abs__filter_chg300_range"
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        
        # Base risk parameters (optimizer can override via param_keys below)
        stop_loss_atr_mult=2.5,
        take_profit_atr_mult=6.0,
        trailing_stop_activation_pct=0.005,
        
        # CRITICAL: Regime-specific multipliers exposed to optimizer
        sl_mult_high_vol=3.5,
        sl_mult_low_vol=1.8,
        sl_mult_trending=2.5,
        tp_mult_high_vol=7.0,
        tp_mult_low_vol=2.5,
        tp_mult_trending=5.0,
        
        allowed_regimes=[
            Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION, 
            Regime.TRENDING_UP, Regime.TRENDING_DOWN
        ]
    )


def create_delta_divergence_strategy() -> StrategyDefinition:
    """
    Delta Divergence Strategy
    
    Enters when price and delta disagree, indicating potential reversal.
    """
    return StrategyDefinition(
        name="Delta Divergence",
        category=StrategyCategory.REVERSAL,
        description="Trade reversals when price diverges from cumulative delta",
        
        entry_conditions=[
            # Divergence detected
            StrategyCondition(
                feature="delta_divergence_60s",
                operator="==",
                threshold=1.0,
                weight=2.5,
                required=True
            ),
            # CVD divergence confirms
            StrategyCondition(
                feature="cvd_price_divergence",
                operator="==",
                threshold=1.0,
                weight=2.0
            ),
            # Exhaustion signals
            StrategyCondition(
                feature="exhaustion_score",
                operator=">",
                threshold=0.3,
                weight=1.5
            ),
            # Volume declining
            StrategyCondition(
                feature="volume_acceleration",
                operator="<",
                threshold=0.8,
                weight=1.0
            ),
            # At value area extreme
            StrategyCondition(
                feature="in_value_area",
                operator="==",
                threshold=0,  # Outside value area
                weight=1.0
            ),
            # FIX 5: Strengthen delta threshold - ensure divergence has volume behind it
            StrategyCondition(
                feature="abs_delta_60s",
                operator=">",
                threshold=8.0,
                weight=1.5,
                required=True  # Significant order flow imbalance required
            ),
        ],
        
        filters=[
            # Don't fight strong momentum
            StrategyCondition(
                feature="delta_pct_300s",
                operator=">",
                threshold=0.7
            ),
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.RANGING]
    )


def create_liquidity_sweep_strategy() -> StrategyDefinition:
    """
    Liquidity Sweep Strategy
    
    Enters after a stop hunt / liquidity sweep reverses.
    """
    return StrategyDefinition(
        name="Liquidity Sweep",
        category=StrategyCategory.REVERSAL,
        description="Fade liquidity sweeps after reversal confirmation",
        
        entry_conditions=[
            # Sweep detected
            StrategyCondition(
                feature="recent_sweep_detected",
                operator="==",
                threshold=1.0,
                weight=2.5,
                required=True
            ),
            # Strong reversal
            StrategyCondition(
                feature="recent_sweep_reversal_strength",
                operator=">",
                threshold=0.35,
                weight=2.0
            ),
            # Volume spike during sweep
            StrategyCondition(
                feature="volume_acceleration",
                operator=">",
                threshold=1.5,
                weight=1.5
            ),
            # Price back inside range
            StrategyCondition(
                feature="in_value_area",
                operator="==",
                threshold=1.0,
                weight=1.0
            ),
            # Book shifted in reversal direction
            StrategyCondition(
                feature="book_trade_agreement",
                operator="==",
                threshold=1.0,
                weight=2.5,      # High weight but NOT required
                required=False   # Let it influence the score, not veto the trade
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        # Note: LOW_LIQUIDITY regime handled by FIX 3 in evaluate()
        allowed_regimes=[Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION]
    )


def create_stacked_imbalance_strategy() -> StrategyDefinition:
    """
    Stacked Imbalance Strategy
    
    Enters in direction of stacked footprint imbalances.
    """
    return StrategyDefinition(
        name="Stacked Imbalance",
        category=StrategyCategory.MOMENTUM,
        description="Trade momentum when multiple price levels show same-side imbalance",
        
        entry_conditions=[
            # Stacked imbalance detected
            StrategyCondition(
                feature="footprint_imbalance_count",
                operator=">=",
                threshold=2,
                weight=2.0,
                required=True
            ),
            # Delta confirms direction
            StrategyCondition(
                feature="delta_pct_60s",
                operator=">",
                threshold=0.2,
                weight=1.5
            ),
            # Book supports
            StrategyCondition(
                feature="depth_imbalance_10",
                operator=">",
                threshold=0.15,
                weight=1.5
            ),
            # Pressure confirms
            StrategyCondition(
                feature="pressure_confirmed",
                operator="==",
                threshold=1.0,
                weight=1.0
            ),
            # Not overextended
            StrategyCondition(
                feature="price_vs_vwap_pct",
                operator="between",
                threshold=-0.003,
                threshold_high=0.003,
                weight=0.5
            ),
        ],
        
        filters=[
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.BREAKOUT, Regime.ACCUMULATION, Regime.DISTRIBUTION] #Add Regime.ACCUMULATION, Regime.DISTRIBUTION
    )


def create_value_area_strategy() -> StrategyDefinition:
    """
    Value Area Strategy
    
    Mean reversion trades at value area boundaries.
    """
    return StrategyDefinition(
        name="Value Area Mean Reversion",
        category=StrategyCategory.MEAN_REVERSION,
        description="Fade moves to value area boundaries",
        
        entry_conditions=[
            # At value area boundary
            StrategyCondition(
                feature="va_breakout_potential",
                operator="==",
                threshold=0,  # NOT breaking out = mean reversion setup
                weight=1.5
            ),
            # Price at VAH or VAL
            StrategyCondition(
                feature="price_vs_vah_pct",
                operator="between",
                threshold=-0.002,
                threshold_high=0.002,
                weight=2.0
            ),
            # Delta showing rejection
            StrategyCondition(
                feature="delta_divergence_60s",
                operator="==",
                threshold=1.0,
                weight=1.5
            ),
            # Volume declining
            StrategyCondition(
                feature="volume_acceleration",
                operator="<",
                threshold=1.0,
                weight=1.0
            ),
            # Book shifting
            StrategyCondition(
                feature="slope_asymmetry",
                operator=">",
                threshold=0,
                weight=0.5
            ),
        ],
        
        filters=[
            # Don't fade strong momentum
            StrategyCondition(
                feature="trade_intensity_60s",
                operator=">",
                threshold=80  # Too active
            ),
            # FIX 4: Market filters - reject bad market environments
            StrategyCondition(
                feature="spread_bps",
                operator=">",
                threshold=8.0  # Reject wide spreads
            ),
            StrategyCondition(
                feature="bid_depth_10",
                operator="<",
                threshold=3.0  # Reject thin bids
            ),
            StrategyCondition(
                feature="ask_depth_10",
                operator="<",
                threshold=3.0  # Reject thin asks
            ),
            StrategyCondition(
                feature="price_change_pct_300s",
                operator="between",
                threshold=-0.008,
                threshold_high=0.008  # Reject dead/choppy markets
            ),
        ],
        
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        # FIX 2: 3:1 risk/reward ratio
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=6.0,
        allowed_regimes=[Regime.RANGING]
    )


# Strategy registry
STRATEGY_LIBRARY = {
    "absorption": create_absorption_strategy,
    "delta_divergence": create_delta_divergence_strategy,
    "liquidity_sweep": create_liquidity_sweep_strategy,
    "stacked_imbalance": create_stacked_imbalance_strategy,
    "value_area": create_value_area_strategy,
}


def get_all_strategies() -> Dict[str, StrategyDefinition]:
    """Get all strategy definitions"""
    return {name: factory() for name, factory in STRATEGY_LIBRARY.items()}


def get_strategy(name: str) -> Optional[StrategyDefinition]:
    """Get a specific strategy by name"""
    if name in STRATEGY_LIBRARY:
        return STRATEGY_LIBRARY[name]()
    return None