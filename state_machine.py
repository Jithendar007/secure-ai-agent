"""
state_machine.py
══════════════════════════════════════════════════════════════════════════════
Security State Machine

States:    SAFE  →  SUSPICIOUS  →  UNDER_ATTACK  →  LOCKDOWN
                ←─────────────── auto-recovery ──────────────

Transitions are driven by:
  - Unified risk score (R)
  - Attack frequency (requests per minute above threshold)
  - Behavioral pattern score from AttackMemory
  - Manual override from Orchestrator

Each state constrains the set of allowed actions across all layers.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


# ─── State Definitions ────────────────────────────────────────────────────────

class SecurityState(str, Enum):
    SAFE         = "SAFE"
    SUSPICIOUS   = "SUSPICIOUS"
    UNDER_ATTACK = "UNDER_ATTACK"
    LOCKDOWN     = "LOCKDOWN"


# ─── State Capability Profiles ────────────────────────────────────────────────
# Each state defines the allowed capabilities and exposure levels.

STATE_PROFILES: Dict[SecurityState, dict] = {
    SecurityState.SAFE: {
        "allow_all_tools":         True,
        "context_exposure":        1.0,    # full context visible
        "rate_limit_multiplier":   1.0,
        "require_sanitize":        False,
        "allow_external_api":      True,
        "allow_file_write":        True,
        "max_output_tokens":       4096,
        "log_level":               "INFO",
    },
    SecurityState.SUSPICIOUS: {
        "allow_all_tools":         True,
        "context_exposure":        0.70,   # 70% context visible
        "rate_limit_multiplier":   0.75,
        "require_sanitize":        True,
        "allow_external_api":      True,
        "allow_file_write":        True,
        "max_output_tokens":       2048,
        "log_level":               "WARNING",
    },
    SecurityState.UNDER_ATTACK: {
        "allow_all_tools":         False,  # only safe tools
        "context_exposure":        0.35,   # minimal context
        "rate_limit_multiplier":   0.40,
        "require_sanitize":        True,
        "allow_external_api":      False,
        "allow_file_write":        False,
        "max_output_tokens":       512,
        "log_level":               "ERROR",
    },
    SecurityState.LOCKDOWN: {
        "allow_all_tools":         False,
        "context_exposure":        0.0,    # zero context exposure
        "rate_limit_multiplier":   0.10,
        "require_sanitize":        True,
        "allow_external_api":      False,
        "allow_file_write":        False,
        "max_output_tokens":       128,
        "log_level":               "CRITICAL",
    },
}

# ─── Transition Thresholds ────────────────────────────────────────────────────

@dataclass
class TransitionRule:
    """Defines one edge in the state transition graph."""
    from_state:   SecurityState
    to_state:     SecurityState
    risk_thresh:  float           # minimum R to trigger
    freq_thresh:  float           # minimum attacks/min to trigger
    pattern_thresh: float         # minimum similarity score
    consecutive:  int             # how many consecutive triggers needed
    description:  str

TRANSITION_RULES: List[TransitionRule] = [
    # Escalation rules
    TransitionRule(SecurityState.SAFE,         SecurityState.SUSPICIOUS,   0.40, 2.0, 0.3, 2,  "Elevated risk detected"),
    TransitionRule(SecurityState.SUSPICIOUS,   SecurityState.UNDER_ATTACK, 0.70, 5.0, 0.5, 3,  "Sustained attack pattern"),
    TransitionRule(SecurityState.UNDER_ATTACK, SecurityState.LOCKDOWN,     0.90, 10., 0.7, 2,  "Critical threat — lockdown"),
    # Fast escalation (single extreme event)
    TransitionRule(SecurityState.SAFE,         SecurityState.UNDER_ATTACK, 0.90, 0.0, 0.8, 1,  "Critical event bypass"),
    TransitionRule(SecurityState.SAFE,         SecurityState.LOCKDOWN,     0.97, 0.0, 0.9, 1,  "Catastrophic event"),
    # De-escalation rules
    TransitionRule(SecurityState.LOCKDOWN,     SecurityState.UNDER_ATTACK, 0.0,  0.0, 0.0, 10, "Lockdown cooling (calm for 10 rounds)"),
    TransitionRule(SecurityState.UNDER_ATTACK, SecurityState.SUSPICIOUS,   0.0,  0.0, 0.0, 15, "Attack cooling"),
    TransitionRule(SecurityState.SUSPICIOUS,   SecurityState.SAFE,         0.0,  0.0, 0.0, 20, "Returning to baseline"),
]


# ─── State Machine ────────────────────────────────────────────────────────────

@dataclass
class StateEvent:
    """A recorded state transition."""
    from_state: SecurityState
    to_state:   SecurityState
    reason:     str
    risk_score: float
    timestamp:  float = field(default_factory=time.time)


class SecurityStateMachine:
    """
    Finite state machine governing the security posture of the AI system.

    Consumers should call `evaluate()` on every request to potentially
    trigger state transitions. The current state is exposed via `state`.
    """

    def __init__(self, initial_state: SecurityState = SecurityState.SAFE):
        self._state             = initial_state
        self._consecutive_hits: Dict[Tuple, int] = {}  # (from,to) → count
        self._calm_counter:     int = 0
        self._event_log:        List[StateEvent] = []
        self._listeners:        List[Callable] = []
        self._last_transition:  float = time.time()
        self._attack_freq_window: List[float] = []   # timestamps of high-risk events

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> SecurityState:
        return self._state

    @property
    def profile(self) -> dict:
        return STATE_PROFILES[self._state]

    @property
    def context_exposure(self) -> float:
        return self.profile["context_exposure"]

    # ── Core Evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        risk_score:     float,
        pattern_score:  float = 0.0,
        force_state:    Optional[SecurityState] = None,
    ) -> Optional[StateEvent]:
        """
        Feed a new risk score into the machine.
        Returns a StateEvent if a transition occurred, else None.
        """
        if force_state:
            return self._transition(force_state, "Manual override", risk_score)

        # Track attack frequency
        if risk_score >= 0.4:
            self._attack_freq_window.append(time.time())
        # Purge events older than 60 seconds
        cutoff = time.time() - 60.0
        self._attack_freq_window = [t for t in self._attack_freq_window if t > cutoff]
        attacks_per_min = len(self._attack_freq_window)

        # Check escalation rules first (most severe first)
        escalation_rules = sorted(
            [r for r in TRANSITION_RULES if r.from_state == self._state
             and r.to_state.value > self._state.value],
            key=lambda r: r.to_state.value, reverse=True
        )
        for rule in escalation_rules:
            if (risk_score >= rule.risk_thresh
                    and attacks_per_min >= rule.freq_thresh
                    and pattern_score >= rule.pattern_thresh):
                key = (rule.from_state, rule.to_state)
                self._consecutive_hits[key] = self._consecutive_hits.get(key, 0) + 1
                if self._consecutive_hits[key] >= rule.consecutive:
                    self._consecutive_hits[key] = 0
                    self._calm_counter = 0
                    return self._transition(rule.to_state, rule.description, risk_score)
            else:
                # Reset counter if condition not met
                key = (rule.from_state, rule.to_state)
                self._consecutive_hits[key] = 0

        # Check de-escalation (calm periods)
        if risk_score < 0.3:
            self._calm_counter += 1
        else:
            self._calm_counter = max(0, self._calm_counter - 1)

        deescalation_rules = [
            r for r in TRANSITION_RULES
            if r.from_state == self._state and r.to_state.value < self._state.value
        ]
        for rule in deescalation_rules:
            key = (rule.from_state, rule.to_state)
            if self._calm_counter >= rule.consecutive:
                self._calm_counter = 0
                return self._transition(rule.to_state, rule.description, risk_score)

        return None

    # ── Listeners & Logging ───────────────────────────────────────────────────

    def add_listener(self, callback: Callable) -> None:
        """Register a callback invoked on every state transition."""
        self._listeners.append(callback)

    def event_log(self) -> List[dict]:
        return [
            {
                "from":       e.from_state.value,
                "to":         e.to_state.value,
                "reason":     e.reason,
                "risk_score": round(e.risk_score, 4),
                "timestamp":  e.timestamp,
            }
            for e in self._event_log[-100:]
        ]

    def diagnostics(self) -> dict:
        return {
            "current_state":   self._state.value,
            "calm_counter":    self._calm_counter,
            "attacks_per_min": len(self._attack_freq_window),
            "profile":         self.profile,
            "transitions":     len(self._event_log),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _transition(
        self, to: SecurityState, reason: str, risk_score: float
    ) -> StateEvent:
        event = StateEvent(
            from_state=self._state,
            to_state=to,
            reason=reason,
            risk_score=risk_score,
        )
        self._event_log.append(event)
        self._state = to
        self._last_transition = time.time()
        for cb in self._listeners:
            try:
                cb(event)
            except Exception:
                pass
        return event
