import random
from typing import Dict, List
import numpy as np

class ThompsonBandit:
    """
    Simple Beta‑distributed Thompson Sampling bandit.
    Each source has (α, β) parameters that model success/failure counts.
    """
    def __init__(self, sources: List[str]):
        self.stats: Dict[str, Dict[str, int]] = {
            src: {"alpha": 1, "beta": 1} for src in sources
        }

    def sample_weights(self) -> Dict[str, float]:
        """Draw a sample from each Beta distribution and normalize."""
        samples = {
            src: random.betavariate(v["alpha"], v["beta"])
            for src, v in self.stats.items()
        }
        total = sum(samples.values())
        return {src: val / total for src, val in samples.items()}

    def update(self, source: str, reward: bool):
        """Reward = True (relevant) or False (non‑relevant)."""
        if reward:
            self.stats[source]["alpha"] += 1
        else:
            self.stats[source]["beta"] += 1

    def get_params(self) -> Dict[str, Dict[str, int]]:
        return self.stats
