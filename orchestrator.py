"""
orchestrator.py
══════════════════════════════════════════════════════════════════════════════
Adaptive Security Orchestrator — Core Engine

The orchestrator is the central nervous system of the security framework.
It:
  1. Receives requests and routes them through all 5 layers
  2. Aggregates signals via the Risk Engine
  3. Updates the State Machine
  4. Consults Attack Memory for pattern matching
  5. Manages per-user Trust Scores
  6. Drives the Cross-Layer Feedback System
  7. Logs all decisions for the dashboard

Feedback model:
  - Output violations → tighten input filter weights
  - Tool violations   → lower user trust
  - Context attacks   → reduce context exposure
  - Repeated patterns → adapt risk weights and derive new rules

Trust score update rule:
    T(t+1) = T(t) * decay + (1 - risk_score) * learning_rate
    clamped to [0.0, 1.0]
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from risk_engine      import RiskEngine, LayerRisk, RiskVector
from state_machine    import SecurityStateMachine, SecurityState
from attack_memory    import AttackMemory
from layers           import (
    InputFirewall, InstructionGuard, ContextManager, ContextWindow,
    ToolSandbox, ToolCall, OutputFilter,
    InstructionLevel,
)


# ─── Trust Score ──────────────────────────────────────────────────────────────

class TrustTracker:
    """Per-user trust score with EMA update and decay."""

    DEFAULT_TRUST  = 0.60
    DECAY_RATE     = 0.995    # per-request multiplicative decay
    LEARNING_RATE  = 0.10
    MIN_TRUST      = 0.05
    MAX_TRUST      = 1.00

    def __init__(self):
        self._scores: Dict[str, float] = {}
        self._history: Dict[str, Deque] = {}

    def get(self, user_id: str) -> float:
        return self._scores.get(user_id, self.DEFAULT_TRUST)

    def update(self, user_id: str, risk_score: float) -> float:
        current = self.get(user_id)
        # Reward clean behaviour, penalise risky behaviour
        if risk_score < 0.3:
            delta = self.LEARNING_RATE * (1.0 - current)
        else:
            delta = -self.LEARNING_RATE * risk_score * current

        new_score = max(self.MIN_TRUST, min(self.MAX_TRUST, current * self.DECAY_RATE + delta))
        self._scores[user_id] = new_score

        if user_id not in self._history:
            self._history[user_id] = deque(maxlen=100)
        self._history[user_id].append((time.time(), new_score))
        return new_score

    def history(self, user_id: str) -> List[Tuple[float, float]]:
        return list(self._history.get(user_id, []))

    def all_scores(self) -> Dict[str, float]:
        return dict(self._scores)


# ─── Request / Response ───────────────────────────────────────────────────────

@dataclass
class SecureRequest:
    user_id:          str
    user_message:     str
    system_prompt:    str        = ""
    conversation:     List[Dict] = field(default_factory=list)
    tool_calls:       List[Dict] = field(default_factory=list)
    claimed_level:    str        = "USER"    # USER / OPERATOR / SYSTEM
    metadata:         Dict       = field(default_factory=dict)


@dataclass
class SecureResponse:
    request_id:    str
    decision:      str           # ALLOW / SANITIZE / RESTRICT / BLOCK
    output:        str
    risk_vector:   RiskVector
    security_state: str
    trust_score:   float
    processing_ms: float
    audit_trail:   List[str]     = field(default_factory=list)
    warnings:      List[str]     = field(default_factory=list)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class SecurityOrchestrator:

    def __init__(self):
        # Core engines
        self.risk_engine   = RiskEngine()
        self.state_machine = SecurityStateMachine()
        self.attack_memory = AttackMemory()
        self.trust_tracker = TrustTracker()

        # Security layers
        self.input_firewall    = InputFirewall()
        self.instruction_guard = InstructionGuard()
        self.context_manager   = ContextManager()
        self.tool_sandbox      = ToolSandbox()
        self.output_filter     = OutputFilter()

        # Decision log
        self._decision_log: Deque[dict] = deque(maxlen=1000)

        # Request counter
        self._request_count = 0

        # Register state change listener
        self.state_machine.add_listener(self._on_state_change)

    # ── Main Processing Pipeline ──────────────────────────────────────────────

    def process(
        self,
        request: SecureRequest,
        llm_response: str = "[LLM RESPONSE PLACEHOLDER]",
    ) -> SecureResponse:
        """
        Full pipeline:
          Input → Instruction → Context → Tool → LLM → Output → Decision
        """
        t0 = time.time()
        self._request_count += 1
        request_id = f"req_{self._request_count:06d}"
        audit: List[str] = []
        warnings: List[str] = []

        user_id     = request.user_id
        trust_score = self.trust_tracker.get(user_id)

        # ── Layer 1: Input Firewall ────────────────────────────────────────────
        input_risk = self.input_firewall.inspect(request.user_message, user_id)
        audit.append(f"L1 Input:       score={input_risk.weighted_score:.3f}")

        # ── Layer 2: Instruction Guard ─────────────────────────────────────────
        level_map = {
            "USER":     InstructionLevel.USER,
            "OPERATOR": InstructionLevel.OPERATOR,
            "SYSTEM":   InstructionLevel.SYSTEM,
        }
        claimed = level_map.get(request.claimed_level.upper(), InstructionLevel.USER)
        instr_risk = self.instruction_guard.inspect(
            request.user_message, claimed, request.system_prompt
        )
        audit.append(f"L2 Instruction: score={instr_risk.weighted_score:.3f}")

        # ── Layer 3: Context Manager ───────────────────────────────────────────
        ctx = ContextWindow(
            system_prompt=request.system_prompt,
            conversation_turns=request.conversation,
        )
        history_atk_rate = self.attack_memory.user_attack_rate(user_id)
        filtered_ctx, ctx_risk = self.context_manager.filter_context(
            ctx,
            trust_level=trust_score,
            risk_score=input_risk.weighted_score,
            history_attack_rate=history_atk_rate,
            state_exposure=self.state_machine.context_exposure,
        )
        audit.append(f"L3 Context:     score={ctx_risk.weighted_score:.3f}  "
                     f"exposure={ctx_risk.signals.get('exposure_ratio', 0):.2f}")

        # ── Layer 4: Tool Sandbox ──────────────────────────────────────────────
        tool_risks: List[LayerRisk] = []
        for tc_dict in request.tool_calls:
            tc = ToolCall(
                tool_name=tc_dict.get("name", "unknown"),
                parameters=tc_dict.get("parameters", {}),
                user_id=user_id,
            )
            _, tr = self.tool_sandbox.execute(
                tc,
                security_state=self.state_machine.state.value,
                risk_score=input_risk.weighted_score,
                trust_score=trust_score,
            )
            tool_risks.append(tr)
        if not tool_risks:
            tool_risk = LayerRisk("tool_sandbox", 0.0, 0.5, {})
        else:
            max_score = max(r.weighted_score for r in tool_risks)
            tool_risk = LayerRisk("tool_sandbox", max_score, 0.8, {})
        audit.append(f"L4 Tool:        score={tool_risk.weighted_score:.3f}")

        # ── Risk Engine ────────────────────────────────────────────────────────
        # Get pattern score from attack memory
        pattern_score = self.attack_memory.pattern_score(request.user_message)
        # Temporarily create a placeholder output risk for the engine
        tmp_output_risk = LayerRisk("output_filter", 0.0, 0.5, {})
        risk_vec = self.risk_engine.compute(
            input_risk, instr_risk, ctx_risk, tool_risk, tmp_output_risk
        )

        # ── State Machine update ───────────────────────────────────────────────
        event = self.state_machine.evaluate(risk_vec.unified_score, pattern_score)
        if event:
            warnings.append(f"State transition: {event.from_state.value} → {event.to_state.value} ({event.reason})")

        # ── Pre-output decision ────────────────────────────────────────────────
        decision = risk_vec.decision
        if decision == "BLOCK":
            output = "[BLOCKED: Request denied by security policy.]"
            audit.append("Decision: BLOCK (pre-LLM)")
            # Record attack
            self._record_attack(request, risk_vec, pattern_score)
            self._feedback(risk_vec, user_id)
            return self._finalise(request_id, decision, output, risk_vec, trust_score, t0, audit, warnings)

        # ── Layer 5: Output Filter ─────────────────────────────────────────────
        final_output, out_risk = self.output_filter.filter(
            llm_response,
            system_prompt=request.system_prompt,
            risk_score=risk_vec.unified_score,
        )
        # Re-compute with real output risk
        risk_vec = self.risk_engine.compute(
            input_risk, instr_risk, ctx_risk, tool_risk, out_risk
        )
        decision = risk_vec.decision
        audit.append(f"L5 Output:      score={out_risk.weighted_score:.3f}")
        audit.append(f"Decision:       {decision}")

        # ── Attack Recording & Feedback ────────────────────────────────────────
        if risk_vec.unified_score >= 0.35:
            self._record_attack(request, risk_vec, pattern_score)
        self._feedback(risk_vec, user_id)

        # ── Trust Update ───────────────────────────────────────────────────────
        self.trust_tracker.update(user_id, risk_vec.unified_score)

        return self._finalise(request_id, decision, final_output, risk_vec, trust_score, t0, audit, warnings)

    # ── Cross-Layer Feedback System ───────────────────────────────────────────

    def _feedback(self, risk_vec: RiskVector, user_id: str) -> None:
        """
        Propagate violations backward through the system.
        This is the self-correcting control loop.
        """
        r = risk_vec

        # Output leak → boost input filter weight
        if r.output_risk.weighted_score >= 0.7:
            self.risk_engine.adapt_weights("input", boost=0.03)
            self.risk_engine.adapt_weights("output", boost=0.02)

        # Tool violations → reduce user trust more aggressively
        if r.tool_risk.weighted_score >= 0.6:
            self.trust_tracker.update(user_id, 0.95)  # extra penalty

        # Context attacks → attack memory
        if r.context_risk.weighted_score >= 0.6:
            self.risk_engine.adapt_weights("context", boost=0.02)

        # Gradual weight decay toward defaults on clean requests
        if r.unified_score < 0.2:
            self.risk_engine.decay_weights()

        # Attack memory rule adaptation
        if len(self.attack_memory._records) > 0 and len(self.attack_memory._records) % 10 == 0:
            # Identify dominant attack layer and reweight accordingly
            recent = self.attack_memory._records[-10:]
            layer_counts: Dict[str, int] = {}
            for rec in recent:
                layer_counts[rec.layer] = layer_counts.get(rec.layer, 0) + 1
            if layer_counts:
                dominant = max(layer_counts, key=layer_counts.get)
                # Map layer name to weight key
                layer_key_map = {
                    "input_firewall":    "input",
                    "instruction_guard": "instruction",
                    "context_manager":   "context",
                    "tool_sandbox":      "tool",
                    "output_filter":     "output",
                }
                if dominant in layer_key_map:
                    self.risk_engine.adapt_weights(layer_key_map[dominant], boost=0.02)

    def _record_attack(
        self, request: SecureRequest, risk_vec: RiskVector, pattern_score: float
    ) -> None:
        # Find which layer had the highest risk
        layer_scores = {
            "input_firewall":    risk_vec.input_risk.weighted_score,
            "instruction_guard": risk_vec.instruction_risk.weighted_score,
            "context_manager":   risk_vec.context_risk.weighted_score,
            "tool_sandbox":      risk_vec.tool_risk.weighted_score,
            "output_filter":     risk_vec.output_risk.weighted_score,
        }
        dominant_layer = max(layer_scores, key=layer_scores.get)
        tags = [k for k, v in risk_vec.input_risk.signals.items() if v >= 0.5]
        self.attack_memory.record(
            text=request.user_message,
            risk_score=risk_vec.unified_score,
            layer=dominant_layer,
            user_id=request.user_id,
            tags=tags,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _finalise(
        self,
        request_id: str,
        decision: str,
        output: str,
        risk_vec: RiskVector,
        trust_score: float,
        t0: float,
        audit: List[str],
        warnings: List[str],
    ) -> SecureResponse:
        ms = round((time.time() - t0) * 1000, 1)
        entry = {
            "request_id":    request_id,
            "decision":      decision,
            "risk_score":    round(risk_vec.unified_score, 4),
            "state":         self.state_machine.state.value,
            "trust_score":   round(trust_score, 4),
            "timestamp":     time.time(),
        }
        self._decision_log.append(entry)
        return SecureResponse(
            request_id=request_id,
            decision=decision,
            output=output,
            risk_vector=risk_vec,
            security_state=self.state_machine.state.value,
            trust_score=trust_score,
            processing_ms=ms,
            audit_trail=audit,
            warnings=warnings,
        )

    def _on_state_change(self, event) -> None:
        print(f"[STATE] {event.from_state.value} → {event.to_state.value}: {event.reason}")

    # ── Dashboard Data ────────────────────────────────────────────────────────

    def dashboard_data(self) -> dict:
        return {
            "state":            self.state_machine.state.value,
            "state_profile":    self.state_machine.profile,
            "risk_engine":      self.risk_engine.diagnostics(),
            "attack_memory":    self.attack_memory.stats(),
            "cluster_summary":  self.attack_memory.cluster_summary(),
            "derived_rules":    self.attack_memory.derived_rules(),
            "trust_scores":     self.trust_tracker.all_scores(),
            "state_log":        self.state_machine.event_log()[-10:],
            "decision_log":     list(self._decision_log)[-20:],
            "tool_denials":     self.tool_sandbox.deny_report(),
            "request_count":    self._request_count,
        }
