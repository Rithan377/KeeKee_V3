"""
router.py — Task Router
=========================
Upgraded from KeeKee v2's route_task().
Improvements:
  - GenerationProfile dataclass (typed, not raw dict)
  - Normalized keyword scoring (score / len(keywords)) prevents long lists dominating
  - Per-task sampling strategy instances (not just params)
  - Extensible: add tasks by subclassing or updating TASK_REGISTRY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sampling import SamplingMethod, Greedy, TopKSampling, TopPSampling, TemperatureSampling


# ─────────────────────────────────────────────
# GenerationProfile  (typed, replaces raw dict)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class GenerationProfile:
    """
    Parameters for generation, selected by the TaskRouter.

    Attributes:
        task_name:      human-readable task identifier
        description:    one-line description
        sampling:       SamplingMethod instance (controls how tokens are picked)
        max_new_tokens: max tokens to generate for this task
    """
    task_name: str
    description: str
    sampling: SamplingMethod
    max_new_tokens: int


# ─────────────────────────────────────────────
# Task Registry  (upgraded TASK_PROFILES)
# ─────────────────────────────────────────────
# Each entry:
#   keywords     — trigger words (scored by normalized overlap)
#   sampling     — SamplingMethod instance
#   max_tokens   — generation budget
#   description  — human label
@dataclass
class _TaskEntry:
    keywords: list[str]
    sampling: SamplingMethod
    max_new_tokens: int
    description: str

    def score(self, prompt_lower: str) -> float:
        """
        Normalized keyword overlap score.
        Divides by keyword count so tasks with many keywords don't dominate.
        (Improvement over v2's raw count)
        """
        if not self.keywords:
            return 0.0
        hits = sum(1 for kw in self.keywords if kw in prompt_lower)
        return hits / len(self.keywords)


TASK_REGISTRY: dict[str, _TaskEntry] = {
    "supply_chain": _TaskEntry(
        keywords=[
            "route", "shipment", "inventory", "warehouse", "port", "logistics",
            "delivery", "freight", "sku", "stock", "dispatch", "vendor", "supplier",
            "order", "transport", "cargo", "lead time", "reorder",
        ],
        sampling=TopKSampling(k=20, temperature=0.4),   # low temp → factual
        max_new_tokens=128,
        description="Supply Chain & Logistics — deterministic, fact-focused",
    ),
    "creative": _TaskEntry(
        keywords=[
            "story", "poem", "write", "imagine", "creative", "fiction",
            "narrative", "character", "plot", "fantasy", "adventure",
        ],
        sampling=TopPSampling(p=0.92, temperature=0.95),  # high diversity
        max_new_tokens=256,
        description="Creative writing — high diversity, expressive output",
    ),
    "qa": _TaskEntry(
        keywords=[
            "what", "why", "how", "explain", "define", "describe",
            "who", "when", "where", "which", "difference between",
        ],
        sampling=TopKSampling(k=40, temperature=0.5),
        max_new_tokens=100,
        description="Question Answering — balanced precision + fluency",
    ),
    "code": _TaskEntry(
        keywords=[
            "code", "function", "implement", "python", "algorithm", "class",
            "debug", "error", "syntax", "loop", "variable", "script",
        ],
        sampling=TopKSampling(k=10, temperature=0.2),   # very deterministic
        max_new_tokens=200,
        description="Code generation — low temperature, precise output",
    ),
    "summarise": _TaskEntry(
        keywords=[
            "summarise", "summarize", "summary", "tldr", "brief", "condense",
            "shorten", "key points", "main idea",
        ],
        sampling=TopKSampling(k=30, temperature=0.45),
        max_new_tokens=80,
        description="Summarisation — concise, faithful to source",
    ),
}

_DEFAULT_PROFILE = GenerationProfile(
    task_name="default",
    description="Default generation",
    sampling=TopKSampling(k=50, temperature=0.7),
    max_new_tokens=128,
)


# ─────────────────────────────────────────────
# TaskRouter
# ─────────────────────────────────────────────
class TaskRouter:
    """
    Classifies an input prompt into a task type and returns
    a GenerationProfile with the appropriate sampling parameters.

    Upgraded from v2:
      - Normalized scoring (avoids bias toward large keyword lists)
      - Returns typed GenerationProfile (not raw dict)
      - Pluggable TASK_REGISTRY
      - min_score threshold: only route if confident enough

    Usage:
        router = TaskRouter()
        profile = router.route("How do I implement a binary search?")
        # profile.task_name == "code"
        # profile.sampling == TopKSampling(k=10, temperature=0.2)
    """

    def __init__(
        self,
        registry: dict[str, _TaskEntry] = TASK_REGISTRY,
        default_profile: GenerationProfile = _DEFAULT_PROFILE,
        min_score: float = 0.05,   # minimum normalized score to not fall back to default
    ):
        self.registry = registry
        self.default_profile = default_profile
        self.min_score = min_score

    def route(self, prompt: str) -> GenerationProfile:
        """
        Classify `prompt` and return the best GenerationProfile.

        Args:
            prompt: raw input text (not tokenized)

        Returns:
            GenerationProfile
        """
        prompt_lower = prompt.lower()
        scores = {
            task: entry.score(prompt_lower)
            for task, entry in self.registry.items()
        }

        best_task = max(scores, key=scores.get)
        best_score = scores[best_task]

        if best_score < self.min_score:
            return self.default_profile

        entry = self.registry[best_task]
        return GenerationProfile(
            task_name=best_task,
            description=entry.description,
            sampling=entry.sampling,
            max_new_tokens=entry.max_new_tokens,
        )

    def describe(self, prompt: str) -> None:
        """Print routing decision for debugging."""
        profile = self.route(prompt)
        print(f"[Router] Task: {profile.task_name}")
        print(f"         {profile.description}")
        print(f"         sampling={profile.sampling.__class__.__name__}, "
              f"max_tokens={profile.max_new_tokens}")
