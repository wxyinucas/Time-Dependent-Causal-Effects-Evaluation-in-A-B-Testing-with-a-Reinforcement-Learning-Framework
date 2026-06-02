# -*- coding: utf-8 -*-
"""Lightweight test monitor (for development / debugging)."""

from __future__ import annotations

import numpy as np


class TestMonitor:
    """Stub monitor that rejects with probability 0.01.

    Used for fast development iterations; does not perform real estimation.
    """

    def __init__(self, **kwargs: object):
        pass

    def estimate_test(self, *args: object) -> bool:
        """Return random rejection decision."""
        return bool(np.random.binomial(1, 0.01))

    def cal_q_arr(self, *args: object) -> bool:
        """Always return False (no Q-value computation)."""
        return False

    def reset(self) -> None:
        """No-op."""
        pass
