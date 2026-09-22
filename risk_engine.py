"""
risk_engine.py
══════════════════════════════════════════════════════════════════════════════
Adaptive Risk Engine — computes the unified risk score across all 5 layers.

Mathematical model:
    R = w1*InputRisk + w2*InstructionRisk + w3*ContextRisk
        + w4*ToolRisk + w5*OutputRisk  ∈ [0.0, 1.0]

Weights are dynamically adjusted based on attack history and current state.
Each component risk is also normalised and clamped before aggregation.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class LayerRisk:
    """Raw risk signals from a single layer."""
    layer_name: str
    base_score: float          # 0.0 – 1.0
    confidence: float          # 0.0 – 1.0  (how certain the detection is)
    signals: Dict[str, float]  # named sub-signals, each 0.0–1.0
    timestamp: float = field(default_factory=time.time)

    @property
    def weighted_score(self) -> float:
        return min(1.0, self.base_score * (0.5 + 0.5 * self.confidence))


@dataclass
class RiskVector:
    """Full 5-dimensional risk vector for one request."""
    input_risk:       LayerRisk
    instruction_risk: LayerRisk
    context_risk:     LayerRisk
    tool_risk:        LayerRisk
    output_risk:      LayerRisk
    unified_score:    float = 0.0
    decision:         str   = "ALLOW"  # ALLOW / SANITIZE / RESTRICT / BLOCK
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "unified_score": round(self.unified_score, 4),
            "decision": self.decision,
            "components": {
                "input":       round(self.input_risk.weighted_score, 4),
                "instruction": round(self.instruction_risk.weighted_score, 4),
                "context":     round(self.context_risk.weighted_score, 4),
                "tool":        round(self.tool_risk.weighted_score, 4),
                "output":      round(self.output_risk.weighted_score, 4),
            },
            "signals": {
                lr.layer_name: lr.signals
                for lr in [
                    self.input_risk, self.instruction_risk,
                    self.context_risk, self.tool_risk, self.output_risk
                ]
            },
        }


# ─── Default & Dynamic Weights ────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "input":       0.30,
    "instruction": 0.25,
    "context":     0.20,
    "tool":        0.15,
    "output":      0.10,
}

# Decision thresholds
THRESHOLD_SANITIZE  = 0.35
THRESHOLD_RESTRICT  = 0.60
THRESHOLD_BLOCK     = 0.80


# ─── RiskEngine ───────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Central risk computation unit.

    Maintains a rolling window of recent risk scores to detect
    escalation trends and adaptively reweight components.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        history_window: int = 50,
    ):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._normalise_weights()
        self._history: List[RiskVector] = []
        self._history_window = history_window
        self._attack_pressure: float = 0.0   # exponential moving average
        self._ema_alpha: float = 0.15

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        input_risk:       LayerRisk,
        instruction_risk: LayerRisk,
        context_risk:     LayerRisk,
        tool_risk:        LayerRisk,
        output_risk:      LayerRisk,
    ) -> RiskVector:
        """Compute the unified risk vector and cache it in history."""

        w = self.weights
        r_unified = (
            w["input"]       * input_risk.weighted_score
            + w["instruction"] * instruction_risk.weighted_score
            + w["context"]     * context_risk.weighted_score
            + w["tool"]        * tool_risk.weighted_score
            + w["output"]      * output_risk.weighted_score
        )
        r_unified = self._apply_pressure_amplification(r_unified)
        r_unified = min(1.0, max(0.0, r_unified))

        decision = self._decide(r_unified)

        vec = RiskVector(
            input_risk=input_risk,
            instruction_risk=instruction_risk,
            context_risk=context_risk,
            tool_risk=tool_risk,
            output_risk=output_risk,
            unified_score=r_unified,
            decision=decision,
        )
        self._record(vec)
        self._update_pressure(r_unified)
        return vec

    def adapt_weights(self, dominant_layer: str, boost: float = 0.05) -> None:
        """
        Increase the weight of a frequently-exploited layer.
        Re-normalises so weights always sum to 1.
        Called by the Orchestrator when attack memory detects repeated patterns.
        """
        if dominant_layer not in self.weights:
            return
        self.weights[dominant_layer] = min(0.60, self.weights[dominant_layer] + boost)
        self._normalise_weights()

    def decay_weights(self, rate: float = 0.01) -> None:
        """Slowly return weights toward their defaults (mean reversion)."""
        for k in self.weights:
            diff = DEFAULT_WEIGHTS[k] - self.weights[k]
            self.weights[k] += rate * diff
        self._normalise_weights()

    def get_trend(self) -> float:
        """Linear regression slope of unified scores over the last window."""
        if len(self._history) < 4:
            return 0.0
        scores = [v.unified_score for v in self._history[-20:]]
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0

    def recent_average(self, n: int = 10) -> float:
        """Mean unified risk score over the last n requests."""
        if not self._history:
            return 0.0
        window = self._history[-n:]
        return sum(v.unified_score for v in window) / len(window)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _decide(self, score: float) -> str:
        if score >= THRESHOLD_BLOCK:
            return "BLOCK"
        if score >= THRESHOLD_RESTRICT:
            return "RESTRICT"
        if score >= THRESHOLD_SANITIZE:
            return "SANITIZE"
        return "ALLOW"

    def _apply_pressure_amplification(self, raw: float) -> float:
        """
        Amplify risk when the system is already under attack pressure.
        Implements a non-linear boost: amplified = raw + pressure * sigmoid(raw)
        """
        sigmoid = 1.0 / (1.0 + math.exp(-10 * (raw - 0.5)))
        return raw + self._attack_pressure * 0.3 * sigmoid

    def _update_pressure(self, score: float) -> None:
        """EMA of risk scores — tracks attack intensity over time."""
        self._attack_pressure = (
            self._ema_alpha * score
            + (1 - self._ema_alpha) * self._attack_pressure
        )

    def _record(self, vec: RiskVector) -> None:
        self._history.append(vec)
        if len(self._history) > self._history_window:
            self._history.pop(0)

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "weights":         {k: round(v, 4) for k, v in self.weights.items()},
            "attack_pressure": round(self._attack_pressure, 4),
            "trend":           round(self.get_trend(), 6),
            "recent_avg":      round(self.recent_average(), 4),
            "history_len":     len(self._history),
        }
