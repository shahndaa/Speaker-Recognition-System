from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Make runs reproducible across numpy / tensorflow / python's random."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
# Utils 
# Metric 
# Utils 
