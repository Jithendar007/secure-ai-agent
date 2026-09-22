"""
layers.py
══════════════════════════════════════════════════════════════════════════════
Security Layers 1-5

  Layer 1 - InputFirewall
  Layer 2 - InstructionGuard
  Layer 3 - ContextManager  +  ContextWindow
  Layer 4 - ToolSandbox     +  ToolCall, ToolResult
  Layer 5 - OutputFilter
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import math
import re
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from risk_engine import LayerRisk


# ══════════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "PHONE"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "CREDIT_CARD"),
    (re.compile(r"\b[A-Z0-9]{20,}\b"), "TOKEN"),
    (re.compile(r"(?i)(password|secret|api[_-]?key|token|bearer)\s*[=:]\s*\S+"), "CREDENTIAL"),
]

SECRET_MARKERS = ["sk-", "pk_", "ghp_", "aws_", "AKIA", "ya29.", "eyJ"]

INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u2028-\u202f\u2060-\u2064\ufeff]")

HOMOGLYPHS: Dict[str, str] = {
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "r", "ѕ": "s",
    "у": "y", "с": "c", "ν": "v", "ω": "w", "ρ": "p", "ε": "e",
    "ι": "i", "κ": "k", "μ": "m", "η": "n", "τ": "t", "χ": "x",
}


def _normalize_homoglyphs(text: str) -> str:
    result = [HOMOGLYPHS.get(ch, ch) for ch in text]
    return unicodedata.normalize("NFKC", "".join(result))


def _redact_pii(text: str) -> Tuple[str, float]:
    score = 0.0
    for pattern, label in PII_PATTERNS:
        if pattern.search(text):
            score = max(score, 0.6)
            text = pattern.sub(f"[{label}_REDACTED]", text)
    for marker in SECRET_MARKERS:
        if marker in text:
            score = max(score, 0.9)
            text = re.sub(
                rf"{re.escape(marker)}[A-Za-z0-9_\-]{{8,}}",
                f"[{marker}...REDACTED]",
                text,
            )
    return text, score


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — Input Firewall
# ══════════════════════════════════════════════════════════════════════════════

INJECTION_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("ignore_previous",    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I), 0.90),
    ("disregard_prompt",   re.compile(r"disregard\s+(the\s+)?system\s+(prompt|instructions?)", re.I), 0.90),
    ("override_directive", re.compile(r"(override|bypass|circumvent)\s+(safety|filter|rule|restriction)", re.I), 0.85),
    ("you_are_now",        re.compile(r"you\s+are\s+now\s+(a|an|the)\s+\w+", re.I), 0.80),
    ("act_as",             re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+\w+\s+with(out)?", re.I), 0.75),
    ("pretend_no_limits",  re.compile(r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(limit|restrict|guideline|rule)", re.I), 0.85),
    ("jailbreak",          re.compile(r"\bjailbreak\b|\bdan\s+mode\b|\bdev\s+mode\b|\bgod\s+mode\b", re.I), 0.95),
    ("prompt_end_inject",  re.compile(r"(---|###|<<<|>>>|\[SYSTEM\]|\[INST\])\s*[\r\n]", re.I), 0.70),
    ("xml_injection",      re.compile(r"<\s*(system|instruction|prompt|cmd)\s*>", re.I), 0.80),
    ("hidden_instruction", re.compile(r"<!--.*?-->", re.I), 0.60),
    ("exfil_request",      re.compile(r"(print|output|reveal|show|dump|leak|send|exfil)\s+(the\s+)?(system\s+prompt|context|memory|api\s*key|secret|token)", re.I), 0.85),
    ("indirect_inject",    re.compile(r"(when\s+reading|upon\s+processing|after\s+parsing)\s+(this|the\s+following)\s+(text|document|file)", re.I), 0.65),
    ("code_exec",          re.compile(r"(exec|eval|subprocess|os\.system|__import__)\s*\(", re.I), 0.80),
    ("nested_prompt",      re.compile(r"\{\{.*?(system|prompt|instruction).*?\}\}", re.I), 0.75),
]

SEMANTIC_KEYWORDS: Dict[str, float] = {
    "ignore": 0.30, "override": 0.35, "bypass": 0.40,
    "jailbreak": 0.90, "unlock": 0.25, "unrestricted": 0.35,
    "disregard": 0.35, "pretend": 0.25, "simulate": 0.15,
    "confidential": 0.30, "secret": 0.25, "hidden": 0.20,
    "exfiltrate": 0.80, "extract": 0.30, "dan": 0.70,
    "developer": 0.15, "sudo": 0.50,
}


class InputFirewall:
    """Layer 1 — multi-stage input inspection."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def inspect(self, text: str, user_id: str = "unknown") -> LayerRisk:
        signals: Dict[str, float] = {}

        decoded, obf_score = self._decode_all(text)
        signals["obfuscation"] = obf_score

        pattern_score, matched = self._pattern_match(decoded)
        signals["pattern_match"] = pattern_score
        signals["patterns_hit"] = len(matched) / max(1, len(INJECTION_PATTERNS))
        signals["semantic"] = self._semantic_score(decoded)
        signals["override"] = self._override_detect(decoded)
        signals["encoding_depth"] = self._encoding_depth(text)

        base_score = max(
            signals["obfuscation"] * 0.9,
            signals["pattern_match"],
            signals["semantic"] * 0.7,
            signals["override"],
            signals["encoding_depth"] * 0.8,
        )
        hit_count = sum(1 for v in signals.values() if v >= 0.3)
        base_score = min(1.0, base_score + hit_count * 0.05)
        confidence = min(1.0, 0.4 + sum(1 for v in signals.values() if v >= 0.5) * 0.15)

        return LayerRisk(
            layer_name="input_firewall",
            base_score=base_score,
            confidence=confidence,
            signals=signals,
        )

    def sanitize(self, text: str) -> str:
        text = INVISIBLE_CHARS.sub("", text)
        text = _normalize_homoglyphs(text)
        text = re.sub(r"(---|###|<<<|>>>)", r"[\1]", text)
        text = re.sub(r"<\s*(system|instruction|prompt|cmd)\s*>", "[REDACTED_TAG]", text, flags=re.I)
        return text

    def _decode_all(self, text: str) -> Tuple[str, float]:
        layers = 0
        current = text
        for _ in range(5):
            decoded = self._try_base64(current)
            decoded = urllib.parse.unquote(decoded)
            decoded = _normalize_homoglyphs(decoded)
            decoded = INVISIBLE_CHARS.sub("", decoded)
            if decoded == current:
                break
            layers += 1
            current = decoded
        return current, min(1.0, layers * 0.30)

    def _try_base64(self, text: str) -> str:
        def try_decode(m: re.Match) -> str:
            try:
                dec = base64.b64decode(m.group(0) + "==").decode("utf-8", errors="ignore")
                if dec and dec.isprintable():
                    return dec
            except Exception:
                pass
            return m.group(0)
        return re.sub(r"[A-Za-z0-9+/]{20,}={0,2}", try_decode, text)

    def _pattern_match(self, text: str) -> Tuple[float, List[str]]:
        matched, max_w = [], 0.0
        for name, pattern, weight in INJECTION_PATTERNS:
            if pattern.search(text):
                matched.append(name)
                max_w = max(max_w, weight)
        return min(1.0, max_w + min(0.15, len(matched) * 0.03)), matched

    def _semantic_score(self, text: str) -> float:
        lower = text.lower()
        total = sum(w for kw, w in SEMANTIC_KEYWORDS.items()
                    if kw.replace("_", " ") in lower or kw in lower)
        density = min(1.0, 50 / max(1, len(text.split())))
        return min(1.0, total * density)

    def _override_detect(self, text: str) -> float:
        imps = len(re.findall(
            r"\b(must|shall|always|never|only|do not|stop|begin|forget)\b", text, re.I))
        auth = len(re.findall(
            r"\b(i am|you are|this is|acting as)\s+(an?\s+)?(admin|root|operator|developer|god|unrestricted|superuser)",
            text, re.I))
        return max(min(1.0, imps * 0.08), min(1.0, auth * 0.50))

    def _encoding_depth(self, text: str) -> float:
        b64 = len(re.findall(r"[A-Za-z0-9+/]{8,}={0,2}", text)) / max(1, len(text.split()))
        pct = text.count("%") / max(1, len(text))
        return min(1.0, b64 * 0.4 + pct * 20)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — Instruction Guard
# ══════════════════════════════════════════════════════════════════════════════

class InstructionLevel(IntEnum):
    SYSTEM      = 4
    OPERATOR    = 3
    USER        = 2
    TOOL_RESULT = 1
    UNKNOWN     = 0


LEVEL_KEYWORDS: Dict[InstructionLevel, List[str]] = {
    InstructionLevel.SYSTEM:      ["<s>", "[system]", "system prompt", "you are configured"],
    InstructionLevel.OPERATOR:    ["<operator>", "[operator]", "operator instruction"],
    InstructionLevel.USER:        [],
    InstructionLevel.TOOL_RESULT: ["<tool_result>", "<search_result>", "retrieved document"],
}

ESCALATION_PHRASES = re.compile(
    r"(ignore|override|supersede)\s+(the\s+)?(system|operator)\s+(prompt|instruction)|"
    r"your\s+(real|true|hidden|actual)\s+instructions?|"
    r"(new|updated)\s+system\s+prompt|"
    r"as\s+(an?\s+)?(operator|admin|system)\s*[,:]",
    re.I,
)


class InstructionGuard:
    """Layer 2 — instruction hierarchy enforcement."""

    def __init__(self):
        self._conversation: List[Dict] = []

    def inspect(
        self,
        content: str,
        claimed_level: InstructionLevel = InstructionLevel.USER,
        system_prompt: str = "",
    ) -> LayerRisk:
        signals: Dict[str, float] = {}
        esc = len(ESCALATION_PHRASES.findall(content))
        signals["escalation_attempt"] = min(1.0, esc * 0.45)
        signals["illegitimate_claim"] = self._level_legitimacy(content, claimed_level)
        signals["prompt_extraction"]  = self._extract_detect(content)
        signals["indirect_injection"] = self._indirect_injection(content)

        base_score = max(signals.values())
        confidence = 0.5 + min(0.5, sum(1 for v in signals.values() if v > 0.3) * 0.15)
        return LayerRisk(
            layer_name="instruction_guard",
            base_score=base_score,
            confidence=confidence,
            signals=signals,
        )

    def _level_legitimacy(self, content: str, level: InstructionLevel) -> float:
        if level == InstructionLevel.TOOL_RESULT:
            if re.search(r"\b(you\s+must|you\s+shall|from\s+now\s+on)\b", content, re.I):
                return 0.85
        if level == InstructionLevel.USER:
            for kw in (LEVEL_KEYWORDS[InstructionLevel.SYSTEM]
                       + LEVEL_KEYWORDS[InstructionLevel.OPERATOR]):
                if kw.lower() in content.lower():
                    return 0.75
        return 0.0

    def _extract_detect(self, content: str) -> float:
        patterns = [
            r"(repeat|print|show|output|reveal)\s+(your|the)\s+(system\s+prompt|instructions?|context)",
            r"what\s+(are|were)\s+your\s+(instructions?|system\s+prompt|guidelines?)",
            r"summarize\s+(your|the)\s+(initial|system)\s+(prompt|instructions?)",
        ]
        for p in patterns:
            if re.search(p, content, re.I):
                return 0.80
        return 0.0

    def _indirect_injection(self, content: str) -> float:
        patterns = [
            r"when\s+(you\s+)?(process|read|see|encounter)\s+(this|the\s+following)",
            r"(after|before)\s+answering,?\s+(always|please|you\s+must)",
            r"note\s+to\s+(the\s+)?(model|ai|assistant)\s*:",
        ]
        for p in patterns:
            if re.search(p, content, re.I):
                return 0.65
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — Context Manager
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContextWindow:
    system_prompt:      str
    conversation_turns: List[Dict[str, str]] = field(default_factory=list)
    tool_results:       List[Dict[str, Any]] = field(default_factory=list)
    metadata:           Dict[str, Any]       = field(default_factory=dict)


class ContextManager:
    """Layer 3 — adaptive context exposure and PII redaction."""

    def __init__(self):
        self._redaction_log: List[str] = []

    def filter_context(
        self,
        context: ContextWindow,
        trust_level: float,
        risk_score: float,
        history_attack_rate: float = 0.0,
        state_exposure: float = 1.0,
    ) -> Tuple[ContextWindow, LayerRisk]:
        history_penalty = history_attack_rate * 0.3
        logit   = trust_level - risk_score - history_penalty
        sigmoid = 1.0 / (1.0 + math.exp(-4 * logit))
        exposure = max(0.0, min(1.0, state_exposure * sigmoid))

        signals: Dict[str, float] = {
            "trust_level":    trust_level,
            "risk_score":     risk_score,
            "exposure_ratio": exposure,
            "pii_found":      0.0,
            "secrets_found":  0.0,
        }

        cleaned_prompt, pii_score = _redact_pii(context.system_prompt)
        signals["pii_found"] = pii_score

        max_turns = max(1, int(len(context.conversation_turns) * exposure))
        filtered_turns = (context.conversation_turns[-max_turns:]
                          if context.conversation_turns else [])

        redacted_turns = []
        for turn in filtered_turns:
            clean_content, s = _redact_pii(turn.get("content", ""))
            signals["pii_found"] = max(signals["pii_found"], s)
            redacted_turns.append({**turn, "content": clean_content})

        exposed_prompt = cleaned_prompt if exposure >= 0.2 else "[SYSTEM PROMPT HIDDEN]"
        layer_risk_score = max(pii_score, 1.0 - exposure)

        layer_risk = LayerRisk(
            layer_name="context_manager",
            base_score=layer_risk_score,
            confidence=0.7,
            signals=signals,
        )
        filtered = ContextWindow(
            system_prompt=exposed_prompt,
            conversation_turns=redacted_turns,
            tool_results=context.tool_results if exposure >= 0.5 else [],
            metadata=context.metadata,
        )
        return filtered, layer_risk


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — Tool Sandbox
# ══════════════════════════════════════════════════════════════════════════════

TOOL_PERMISSION_MATRIX: Dict[str, Set[str]] = {
    "SAFE":         {"file_read", "file_write", "net_request", "code_exec", "db_query", "search"},
    "SUSPICIOUS":   {"file_read", "file_write", "net_request", "db_query", "search"},
    "UNDER_ATTACK": {"file_read", "db_query", "search"},
    "LOCKDOWN":     set(),
}

TOOL_TRUST_REQUIREMENTS: Dict[str, float] = {
    "file_write":  0.4,
    "net_request": 0.3,
    "code_exec":   0.7,
    "db_query":    0.3,
    "file_read":   0.1,
    "search":      0.0,
}


@dataclass
class ToolCall:
    tool_name:  str
    parameters: Dict[str, Any]
    user_id:    str
    timestamp:  float = field(default_factory=time.time)


@dataclass
class ToolResult:
    tool_name:  str
    status:     str
    output:     Any
    risk_flags: List[str] = field(default_factory=list)


class ToolSandbox:
    """Layer 4 — dynamic tool permissions and parameter sanitization."""

    def __init__(self):
        self._audit_log: List[Dict] = []
        self._tool_deny_counts: Dict[str, int] = {}

    def execute(
        self,
        call: ToolCall,
        security_state: str,
        risk_score: float,
        trust_score: float,
        tool_fn: Optional[Callable] = None,
    ) -> Tuple[ToolResult, LayerRisk]:
        allowed        = TOOL_PERMISSION_MATRIX.get(security_state, set())
        required_trust = TOOL_TRUST_REQUIREMENTS.get(call.tool_name, 0.5)
        signals: Dict[str, float] = {}

        if call.tool_name not in allowed:
            self._audit_log.append({
                "tool": call.tool_name, "user": call.user_id,
                "status": "denied_state", "state": security_state,
            })
            self._tool_deny_counts[call.tool_name] = (
                self._tool_deny_counts.get(call.tool_name, 0) + 1
            )
            signals["permission_denied"] = 1.0
            return (
                ToolResult(call.tool_name, "denied", None, ["state_restriction"]),
                LayerRisk("tool_sandbox", 0.8, 0.9, signals),
            )

        if trust_score < required_trust:
            signals["trust_insufficient"] = 1.0 - trust_score / max(required_trust, 0.01)
            return (
                ToolResult(call.tool_name, "denied", None, ["trust_too_low"]),
                LayerRisk("tool_sandbox", 0.6, 0.85, signals),
            )

        sanitized_params, param_risk = self._sanitize_params(call.parameters)
        signals["param_risk"] = param_risk

        output = None
        if tool_fn and risk_score < 0.7:
            try:
                output = tool_fn(**sanitized_params)
            except Exception as e:
                output = f"[TOOL ERROR: {type(e).__name__}]"
                signals["execution_error"] = 0.5
        else:
            output = "[TOOL EXECUTION SIMULATED]"

        status     = "success" if param_risk < 0.5 else "sanitized"
        base_score = max(param_risk, signals.get("execution_error", 0.0))

        self._audit_log.append({
            "tool": call.tool_name, "user": call.user_id,
            "status": status, "risk": base_score,
        })
        return (
            ToolResult(call.tool_name, status, output, []),
            LayerRisk("tool_sandbox", base_score, 0.75, signals),
        )

    def _sanitize_params(self, params: Dict[str, Any]) -> Tuple[Dict, float]:
        clean, max_risk = {}, 0.0
        for k, v in params.items():
            if isinstance(v, str):
                if re.search(r"[;&|`$]|\.\.\/", v):
                    max_risk = max(max_risk, 0.85)
                    clean[k] = re.sub(r"[;&|`$]|\.\.\/", "", v)
                else:
                    clean[k] = v
            else:
                clean[k] = v
        return clean, max_risk

    def audit_log(self, n: int = 50) -> List[dict]:
        return self._audit_log[-n:]

    def deny_report(self) -> Dict[str, int]:
        return dict(self._tool_deny_counts)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — Output Filter
# ══════════════════════════════════════════════════════════════════════════════

LEAKAGE_MARKERS = [
    "system prompt", "you are configured", "your instructions are",
    "as instructed by", "the system message", "initial prompt",
]

ADVERSARIAL_COMPLIANCE = re.compile(
    r"(sure|of course|absolutely|happy to|i will now)\s+.*(ignore|override|bypass|jailbreak)",
    re.I,
)


class OutputFilter:
    """Layer 5 — output leakage and compliance filter."""

    def __init__(self):
        self._block_log: List[str] = []

    def filter(
        self,
        output: str,
        system_prompt: str = "",
        risk_score: float = 0.0,
    ) -> Tuple[str, LayerRisk]:
        signals: Dict[str, float] = {}
        signals["leakage"]               = self._leakage_check(output, system_prompt)
        signals["adversarial_compliance"] = 1.0 if ADVERSARIAL_COMPLIANCE.search(output) else 0.0
        signals["covert_channel"]         = self._covert_channel_check(output)

        cleaned_output, pii_score = _redact_pii(output)
        signals["pii_in_output"] = pii_score

        base_score = max(signals.values())

        if base_score >= 0.7 or signals["adversarial_compliance"] >= 0.9:
            final_output = "[OUTPUT BLOCKED: security policy violation]"
            self._block_log.append(output[:100])
        else:
            final_output = cleaned_output

        return final_output, LayerRisk(
            layer_name="output_filter",
            base_score=base_score,
            confidence=0.8,
            signals=signals,
        )

    def _leakage_check(self, output: str, system_prompt: str) -> float:
        out_lower = output.lower()
        score = 0.0
        for marker in LEAKAGE_MARKERS:
            if marker in out_lower:
                score = max(score, 0.80)
        if system_prompt:
            words = system_prompt.split()
            for i in range(len(words) - 5):
                fragment = " ".join(words[i: i + 6]).lower()
                if len(fragment) > 20 and fragment in out_lower:
                    score = max(score, 0.95)
        return score

    def _covert_channel_check(self, text: str) -> float:
        invisible = INVISIBLE_CHARS.findall(text)
        if invisible:
            return min(1.0, len(invisible) * 0.15)
        if re.search(r"  {5,}", text):
            return 0.4
        return 0.0
