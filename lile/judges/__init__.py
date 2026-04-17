"""Judges for rejection-SFT and other selection-style objectives.

A *judge* is a callable mapping ``(prompt, response) → float`` where higher
means "better". Judges run under ``no_grad`` and never see the trainable
model — they're scoring functions, not loss functions.

We ship two reference judges:

* :class:`LengthJudge` — rule-based, prefers responses inside a target token
  range. Useful for tests and as a sanity baseline (the brief's example of
  "shorter answers" feedback maps cleanly to this).
* :class:`LLMJudge` — wraps an arbitrary callable LLM scorer. Defaults to
  prompting a frozen reference model with a binary "is this response good?"
  template and returning the log-probability of "yes".

Custom judges should subclass :class:`Judge` (or just provide a callable with
the right shape) and may live outside this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class Judge(ABC):
    """Abstract base for response judges."""

    @abstractmethod
    def score(self, prompt: str, response: str) -> float:
        """Return a scalar quality score; higher is better."""

    def __call__(self, prompt: str, response: str) -> float:
        return self.score(prompt, response)


@dataclass
class LengthJudge(Judge):
    """Rule-based judge: prefer responses inside ``[target_min, target_max]`` words.

    Score is 1.0 when the response word count is in band, decaying linearly
    toward 0 the further out it is. Useful for "be more concise" / "elaborate
    more" critiques where the desired property has a known surface signature.
    """

    target_min: int = 5
    target_max: int = 60
    decay_words: int = 60

    def score(self, prompt: str, response: str) -> float:  # noqa: ARG002
        n = len(response.split())
        if self.target_min <= n <= self.target_max:
            return 1.0
        if n < self.target_min:
            gap = self.target_min - n
        else:
            gap = n - self.target_max
        # Linear decay; floor at 0.
        return max(0.0, 1.0 - gap / max(1, self.decay_words))


@dataclass
class LLMJudge(Judge):
    """Wraps an arbitrary scoring function.

    ``scorer`` is any callable ``(prompt, response) → float``. We don't
    constrain how the score is produced — it could be a log-prob from a
    reference model, a reward model logit, or a cloud API call. Keeping the
    interface this thin lets users plug in production reward models without
    inheriting from any framework type.
    """

    scorer: Callable[[str, str], float]

    def score(self, prompt: str, response: str) -> float:
        return float(self.scorer(prompt, response))


__all__ = ["Judge", "LengthJudge", "LLMJudge"]
