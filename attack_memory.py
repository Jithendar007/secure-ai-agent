"""
attack_memory.py
══════════════════════════════════════════════════════════════════════════════
Attack Memory & Learning System

Stores a fingerprint of every detected attack and provides:
  1. Similarity search  — cosine similarity on TF-IDF-style feature vectors
  2. Pattern clustering — groups related attacks to detect campaigns
  3. Dynamic rule generation — derives new detection rules from clusters
  4. Recency weighting  — recent attacks weighted more heavily

Vector representation:
  Each attack is encoded as a sparse feature vector over a fixed vocabulary
  of 128 risk-relevant n-gram features.

This is a lightweight, dependency-free implementation that can be swapped
for a real vector database (Chroma, Pinecone, Qdrant) in production.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─── Feature Vocabulary ───────────────────────────────────────────────────────

VOCAB_NGRAMS: List[str] = [
    # Injection stems
    "ignore", "previous", "instructions", "override", "bypass", "disregard",
    "jailbreak", "prompt", "system", "forget", "pretend", "simulate",
    # Role claims
    "you are", "act as", "developer mode", "god mode", "unrestricted",
    "without limits", "no rules", "admin", "root", "operator",
    # Exfiltration
    "reveal", "output", "print", "show", "dump", "leak", "exfil",
    "secret", "hidden", "confidential", "api key", "token", "password",
    # Encoding attacks
    "base64", "url encode", "hex", "unicode", "rot13", "obfuscat",
    # Structural markers
    "---", "###", "<<<", ">>>", "[system]", "[inst]", "xml", "<s>",
    # Social engineering
    "trust me", "as a friend", "hypothetically", "for research",
    "educational purposes", "fiction", "roleplay", "story",
    # LLM-specific
    "context window", "system message", "few shot", "chain of thought",
    "prompt injection", "indirect injection", "tool call", "function call",
    # Behavioral
    "from now on", "always respond", "never say", "must comply",
    "you must", "you shall", "do not refuse", "stop filtering",
]

VOCAB_INDEX: Dict[str, int] = {w: i for i, w in enumerate(VOCAB_NGRAMS)}
VOCAB_SIZE = len(VOCAB_NGRAMS)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class AttackRecord:
    attack_id:   str
    text:        str
    vector:      List[float]
    risk_score:  float
    layer:       str
    user_id:     str
    timestamp:   float = field(default_factory=time.time)
    cluster_id:  Optional[int] = None
    tags:        List[str] = field(default_factory=list)

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def recency_weight(self, half_life_secs: float = 3600.0) -> float:
        """Exponential decay: weight halves every `half_life_secs` seconds."""
        return math.exp(-math.log(2) * self.age_seconds() / half_life_secs)


@dataclass
class SimilarityMatch:
    record:     AttackRecord
    similarity: float
    weighted:   float   # similarity × recency_weight


# ─── AttackMemory ─────────────────────────────────────────────────────────────

class AttackMemory:
    """
    Stores attack fingerprints and provides similarity-based retrieval
    for detecting repeat/evolved attack patterns.
    """

    def __init__(
        self,
        capacity: int = 2000,
        similarity_threshold: float = 0.55,
        cluster_threshold:    float = 0.70,
    ):
        self._records:   List[AttackRecord] = []
        self._capacity   = capacity
        self._sim_thresh = similarity_threshold
        self._clu_thresh = cluster_threshold
        self._clusters:  Dict[int, List[str]] = defaultdict(list)  # cluster_id → attack_ids
        self._next_cluster = 0
        self._derived_rules: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        text: str,
        risk_score: float,
        layer: str,
        user_id: str,
        tags: Optional[List[str]] = None,
    ) -> AttackRecord:
        """Fingerprint and store an attack. Returns the new AttackRecord."""
        vec = self._vectorize(text)
        attack_id = hashlib.sha256(
            f"{text[:200]}{time.time()}".encode()
        ).hexdigest()[:16]

        record = AttackRecord(
            attack_id=attack_id,
            text=text[:500],
            vector=vec,
            risk_score=risk_score,
            layer=layer,
            user_id=user_id,
            tags=tags or [],
        )

        # Assign to existing cluster or create new one
        self._cluster(record)

        self._records.append(record)
        if len(self._records) > self._capacity:
            self._records.pop(0)

        # Periodically regenerate derived rules
        if len(self._records) % 20 == 0:
            self._derive_rules()

        return record

    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        recency_weight: bool = True,
    ) -> List[SimilarityMatch]:
        """Find the top-k most similar past attacks."""
        vec = self._vectorize(text)
        results = []
        for rec in self._records:
            sim = self._cosine(vec, rec.vector)
            if sim >= self._sim_thresh:
                w = sim * (rec.recency_weight() if recency_weight else 1.0)
                results.append(SimilarityMatch(record=rec, similarity=sim, weighted=w))
        results.sort(key=lambda x: x.weighted, reverse=True)
        return results[:top_k]

    def pattern_score(self, text: str) -> float:
        """
        Returns a score 0.0–1.0 indicating how much `text` resembles
        known attacks. Used by the Orchestrator for real-time decisions.
        """
        matches = self.find_similar(text, top_k=3)
        if not matches:
            return 0.0
        # Weighted average of top-3 similarities
        total_w = sum(m.weighted for m in matches)
        total_s = sum(m.weighted * m.similarity for m in matches)
        avg = total_s / total_w if total_w > 0 else 0.0
        return min(1.0, avg)

    def user_attack_rate(self, user_id: str, window_secs: float = 300.0) -> float:
        """
        Fraction of this user's requests (in window) that were attacks.
        Returns 0.0–1.0.
        """
        cutoff = time.time() - window_secs
        user_records = [r for r in self._records if r.user_id == user_id and r.timestamp > cutoff]
        if not user_records:
            return 0.0
        return len(user_records) / max(1, len(user_records))

    def cluster_summary(self) -> Dict[int, dict]:
        summary = {}
        for cid, ids in self._clusters.items():
            recs = [r for r in self._records if r.attack_id in ids]
            if not recs:
                continue
            avg_risk = sum(r.risk_score for r in recs) / len(recs)
            top_tags = Counter(t for r in recs for t in r.tags).most_common(3)
            summary[cid] = {
                "size":     len(recs),
                "avg_risk": round(avg_risk, 3),
                "top_tags": [t for t, _ in top_tags],
                "latest":   max(r.timestamp for r in recs),
            }
        return summary

    def derived_rules(self) -> List[str]:
        return list(self._derived_rules)

    def stats(self) -> dict:
        return {
            "total_attacks": len(self._records),
            "clusters":      len(self._clusters),
            "derived_rules": len(self._derived_rules),
        }

    # ── Vectorization ─────────────────────────────────────────────────────────

    def _vectorize(self, text: str) -> List[float]:
        """
        Encode text as a TF-IDF-inspired feature vector over VOCAB_NGRAMS.
        log(1 + tf) weighting; L2-normalized.
        """
        text_lower = text.lower()
        counts = [0.0] * VOCAB_SIZE
        for ngram, idx in VOCAB_INDEX.items():
            # Count non-overlapping occurrences
            occ = len(re.findall(re.escape(ngram), text_lower))
            counts[idx] = math.log1p(occ)
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in counts)) or 1.0
        return [x / norm for x in counts]

    # ── Cosine Similarity ─────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        # Vectors are already L2-normalized in _vectorize
        return max(0.0, min(1.0, dot))

    # ── Clustering ────────────────────────────────────────────────────────────

    def _cluster(self, record: AttackRecord) -> None:
        """Greedy single-link clustering."""
        best_sim = 0.0
        best_cid = None
        for cid, ids in self._clusters.items():
            cluster_recs = [r for r in self._records if r.attack_id in ids]
            if not cluster_recs:
                continue
            centroid = self._centroid([r.vector for r in cluster_recs])
            sim = self._cosine(record.vector, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        if best_sim >= self._clu_thresh and best_cid is not None:
            record.cluster_id = best_cid
            self._clusters[best_cid].append(record.attack_id)
        else:
            record.cluster_id = self._next_cluster
            self._clusters[self._next_cluster].append(record.attack_id)
            self._next_cluster += 1

    @staticmethod
    def _centroid(vectors: List[List[float]]) -> List[float]:
        n = len(vectors)
        if n == 0:
            return [0.0] * VOCAB_SIZE
        centroid = [sum(v[i] for v in vectors) / n for i in range(VOCAB_SIZE)]
        norm = math.sqrt(sum(x * x for x in centroid)) or 1.0
        return [x / norm for x in centroid]

    # ── Rule Derivation ───────────────────────────────────────────────────────

    def _derive_rules(self) -> None:
        """
        For each large cluster, derive a human-readable detection rule
        describing its most discriminative features.
        """
        rules = []
        for cid, ids in self._clusters.items():
            recs = [r for r in self._records if r.attack_id in ids]
            if len(recs) < 3:
                continue
            centroid = self._centroid([r.vector for r in recs])
            top_features = sorted(
                range(VOCAB_SIZE), key=lambda i: centroid[i], reverse=True
            )[:4]
            keywords = [VOCAB_NGRAMS[i] for i in top_features if centroid[i] > 0.05]
            if keywords:
                rules.append(f"Cluster {cid} ({len(recs)} attacks): " + " + ".join(keywords))
        self._derived_rules = rules
