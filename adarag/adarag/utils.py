# -*- coding: utf-8 -*-
"""
Utility helpers.

This module intentionally keeps dependencies minimal.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import os
import time
import random
import numpy as np
from typing import Optional


def set_global_seed(seed: int) -> None:
    """Best-effort global seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def now_ms() -> int:
    """Current wall time in milliseconds."""
    return int(time.time() * 1000)


def soft_normalize_nonneg(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a non-negative vector to sum to 1.

    If all-zeros, return uniform.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= eps:
        return np.ones_like(x) / max(1, x.size)
    return x / s


def project_prob_vector_sum_leq_one(v: np.ndarray) -> np.ndarray:
    """
    A simple (approximate) projection to the feasible set:
        v_i in [0,1], and sum(v) <= 1.

    For research prototypes this works well:
    1) clip to [0,1]
    2) if sum > 1, scale down proportionally so sum becomes 1

    Note: This is not the exact Euclidean projection when upper bounds are active,
    but it's stable, fast, and keeps feasibility.
    """
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0.0, 1.0)
    s = float(v.sum())
    if s > 1.0:
        v = v / s
    return v


def project_z_to_domain(z: np.ndarray, n_docs: int) -> np.ndarray:
    """
    Project z = [x(1..n), y(1..n), p] to domain Z in the paper:

    - x_i in [0,1], sum(x) <= 1
    - y_i in [0,1], sum(y) <= 1
    - p in [0,1]

    Args:
        z: shape (2*n_docs + 1,)
        n_docs: n

    Returns:
        z_proj: feasible vector
    """
    z = np.asarray(z, dtype=float).copy()
    assert z.shape[0] == 2 * n_docs + 1, "Bad z dimension"
    x = project_prob_vector_sum_leq_one(z[:n_docs])
    y = project_prob_vector_sum_leq_one(z[n_docs:2 * n_docs])
    p = float(np.clip(z[-1], 0.0, 1.0))
    return np.concatenate([x, y, np.array([p], dtype=float)])


def to_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append one line of JSON to a JSONL file."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sample_unit_sphere(dim: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if rng is None:
        u = np.random.randn(dim).astype(np.float32)   # 用全局 np.random，能被 np.random.seed 控制
    else:
        u = rng.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(u))
    if norm < 1e-12:
        u = np.zeros(dim, dtype=np.float32)
        u[-1] = 1.0
        return u
    return (u / norm).astype(np.float32)