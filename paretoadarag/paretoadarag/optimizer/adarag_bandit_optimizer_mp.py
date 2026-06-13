from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence, List

import numpy as np


@dataclass
class AdaRAGState:
    w_hat: np.ndarray
    lambda_: float
    t: int


def _as_doc_choices(v: Optional[Sequence[int]], fallback_n: int) -> List[int]:
    if v is None:
        return list(range(1, int(fallback_n) + 1))
    out = [int(x) for x in v]
    if not out:
        raise ValueError("doc choices must not be empty.")
    if min(out) < 1:
        raise ValueError(f"doc choices must be >= 1, got {out}")
    return out


def _project_prob_mass(v: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """
    Project to original AdaRAG-style probability-mass domain:
        0 <= v_i <= 1, sum(v_i) <= cap.

    The remaining probability mass, 1 - sum(v), is top0.
    """
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0.0, 1.0)
    cap = float(np.clip(cap, 0.0, 1.0))

    s = float(v.sum())
    if s > cap:
        v = v / max(s, 1e-12) * cap
    return v


def _init_prob_mass(policy: str, n: int, total_mass: float) -> np.ndarray:
    """
    Initialize probabilities over nonzero top-k choices.

    total_mass < 1 leaves top0 probability:
        p(top0) = 1 - total_mass.
    """
    policy = str(policy or "increasing").strip().lower()
    if n <= 0:
        raise ValueError("n must be positive.")

    total_mass = float(np.clip(total_mass, 0.0, 1.0))

    if policy == "uniform":
        base = np.full(n, 1.0 / n, dtype=float)
    elif policy == "increasing":
        base = np.arange(1, n + 1, dtype=float)
        base = base / float(base.sum())
    elif policy in ("fixed_max", "max"):
        base = np.zeros(n, dtype=float)
        base[-1] = 1.0
    elif policy in ("fixed_min", "min"):
        base = np.zeros(n, dtype=float)
        base[0] = 1.0
    else:
        raise ValueError(f"Unknown init_doc_policy={policy}")

    return total_mass * base


class AdaRAGBanditOptimizerMP:
    """
    Original AdaRAG-style one-point bandit optimizer.

    Decision vector:
        z = [x, y, p]

    x:
        probabilities over light doc choices; top0 probability = 1 - sum(x)
    y:
        probabilities over heavy doc choices; top0 probability = 1 - sum(y)
    p:
        heavy retrieval proportion

    Difference from ParetoAdaRAG-3P:
        - no CVaR auxiliary variable nu
        - no psi objective
        - one perturbed execution per slot obtains both d_feedback and q_feedback
        - objective is average end-to-end latency d only
    """

    def __init__(
        self,
        *,
        n_docs: Optional[int] = None,
        n_docs_light: Optional[int] = None,
        n_docs_heavy: Optional[int] = None,
        doc_choices_light: Optional[Sequence[int]] = None,
        doc_choices_heavy: Optional[Sequence[int]] = None,
        init_doc_policy: str = "increasing",
        init_top0_prob: float = 0.0,
        q_target: float,
        alpha: float = 0.0005,
        mu: float = 0.005,
        delta: float = 0.05,
        gamma: float = 0.05,
        p_init: float = 0.35,
        p_min: float = 0.0,
        p_max: float = 1.0,
        seed: int = 0,
        normalize_enabled: bool = False,
        d_ref: float = 1.0,
        g_ref: float = 1.0,
    ) -> None:
        if n_docs_light is None:
            n_docs_light = int(n_docs or 4)
        if n_docs_heavy is None:
            n_docs_heavy = int(n_docs or 6)

        self.doc_choices_light = _as_doc_choices(doc_choices_light, int(n_docs_light))
        self.doc_choices_heavy = _as_doc_choices(doc_choices_heavy, int(n_docs_heavy))

        self.n_docs_light = len(self.doc_choices_light)
        self.n_docs_heavy = len(self.doc_choices_heavy)
        self.max_light_docs = max(self.doc_choices_light)
        self.max_heavy_docs = max(self.doc_choices_heavy)

        self.dim_z = self.n_docs_light + self.n_docs_heavy + 1
        self.dim_w = self.dim_z

        self.q_target = float(q_target)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.delta = float(delta)
        self.gamma = float(gamma)

        self.p_min = float(np.clip(p_min, 0.0, 1.0))
        self.p_max = float(np.clip(p_max, self.p_min, 1.0))

        self.normalize_enabled = bool(normalize_enabled)
        self.d_ref = float(d_ref)
        self.g_ref = float(g_ref)

        self.rng = np.random.RandomState(seed)

        init_mass = 1.0 - float(np.clip(init_top0_prob, 0.0, 1.0))
        x0 = _init_prob_mass(init_doc_policy, self.n_docs_light, init_mass)
        y0 = _init_prob_mass(init_doc_policy, self.n_docs_heavy, init_mass)
        p0 = np.array([float(np.clip(p_init, self.p_min, self.p_max))], dtype=float)

        w0 = np.concatenate([x0, y0, p0])
        self.state = AdaRAGState(
            w_hat=self._project_to_scaled_domain(w0, 1.0 - self.gamma),
            lambda_=0.0,
            t=0,
        )

    def _sample_unit_sphere(self) -> np.ndarray:
        u = self.rng.randn(self.dim_w).astype(float)
        norm = float(np.linalg.norm(u))
        if norm < 1e-12:
            u[:] = 0.0
            u[-1] = 1.0
            return u
        return u / norm

    def _project_to_domain(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        x = _project_prob_mass(w[: self.n_docs_light], cap=1.0)
        y = _project_prob_mass(w[self.n_docs_light: self.n_docs_light + self.n_docs_heavy], cap=1.0)
        p = float(np.clip(w[-1], self.p_min, self.p_max))
        return np.concatenate([x, y, np.array([p], dtype=float)])

    def _project_to_scaled_domain(self, w: np.ndarray, scale: float) -> np.ndarray:
        """
        Shrunk domain used by one-point perturbation:
            sum(x) <= scale, sum(y) <= scale,
            p in [p_min, p_min + scale*(p_max-p_min)].
        """
        scale = float(np.clip(scale, 0.0, 1.0))
        w0 = self._project_to_domain(w)
        x, y, p = self.unpack_w(w0)

        x = _project_prob_mass(x, cap=scale)
        y = _project_prob_mass(y, cap=scale)

        p_upper = self.p_min + scale * (self.p_max - self.p_min)
        p = float(np.clip(p, self.p_min, p_upper))

        return np.concatenate([x, y, np.array([p], dtype=float)])

    def unpack_w(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        w = np.asarray(w, dtype=float)
        x = np.asarray(w[: self.n_docs_light], dtype=float)
        y = np.asarray(w[self.n_docs_light: self.n_docs_light + self.n_docs_heavy], dtype=float)
        p = float(w[-1])
        return x, y, p

    def propose(self):
        """
        One-point AdaRAG action proposal.

        Returns:
            w_exec: actual perturbed action after safety projection
            u: sampled unit direction used by the estimator
            w_hat: center decision before update
        """
        w_hat = self.state.w_hat.copy()
        u = self._sample_unit_sphere()
        w_exec = self._project_to_domain(w_hat + self.delta * u)
        return w_exec, u, w_hat

    def _one_point_grad(self, value: float, u_dir: np.ndarray) -> np.ndarray:
        return (self.dim_w / self.delta) * float(value) * np.asarray(u_dir, dtype=float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = max(1e-12, float(np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.dot(a, b) / denom)

    def update(
        self,
        *,
        u_dir: np.ndarray,
        d_feedback: float,
        q_feedback: float,
    ) -> Dict[str, float]:
        """
        Original AdaRAG-style update.

        Normalized objective/constraint used by this implementation:
            objective: d / d_ref
            constraint: (Q - q) / g_ref <= 0

        Lagrangian direction:
            grad_d - lambda * grad_q

        Multiplier update follows the paper-style first-order correction:
            lambda <- [lambda + mu * ((Q-q)/g_ref - grad_q·(z_next-z_prev))]_+
        """
        st = self.state
        w_prev = st.w_hat.copy()

        g_feedback = self.q_target - float(q_feedback)

        if self.normalize_enabled:
            d_value = float(d_feedback) / max(1e-12, self.d_ref)
            q_value = float(q_feedback) / max(1e-12, self.g_ref)
            g_value = float(g_feedback) / max(1e-12, self.g_ref)
        else:
            d_value = float(d_feedback)
            q_value = float(q_feedback)
            g_value = float(g_feedback)

        grad_d = self._one_point_grad(d_value, u_dir)
        grad_q = self._one_point_grad(q_value, u_dir)

        v_dir = grad_d - st.lambda_ * grad_q
        w_next = self._project_to_scaled_domain(w_prev - self.alpha * v_dir, 1.0 - self.gamma)

        correction_q = float(np.dot(grad_q, w_next - w_prev))
        lambda_next = max(0.0, st.lambda_ + self.mu * (g_value - correction_q))

        st.w_hat = w_next
        st.lambda_ = float(lambda_next)
        st.t += 1

        x_prev, y_prev, p_prev = self.unpack_w(w_prev)
        x_next, y_next, p_next = self.unpack_w(w_next)

        return {
            "t": float(st.t),
            "lambda": float(st.lambda_),

            "d_feedback": float(d_feedback),
            "q_feedback": float(q_feedback),
            "g_feedback": float(g_feedback),

            "d_value_normed": float(d_value),
            "q_value_normed": float(q_value),
            "g_value_normed": float(g_value),

            "perturbation_norm": float(np.linalg.norm(self.delta * u_dir)),
            "grad_d_norm": float(np.linalg.norm(grad_d)),
            "grad_q_norm": float(np.linalg.norm(grad_q)),
            "v_norm": float(np.linalg.norm(v_dir)),
            "update_norm": float(np.linalg.norm(w_next - w_prev)),
            "grad_cosine_d_q": self._cosine(grad_d, grad_q),

            "p_hat_before": float(p_prev),
            "p_hat_after": float(p_next),

            "x_sum_before": float(np.sum(x_prev)),
            "y_sum_before": float(np.sum(y_prev)),
            "x_sum_after": float(np.sum(x_next)),
            "y_sum_after": float(np.sum(y_next)),

            "x_top0_before": float(max(0.0, 1.0 - np.sum(x_prev))),
            "y_top0_before": float(max(0.0, 1.0 - np.sum(y_prev))),
            "x_top0_after": float(max(0.0, 1.0 - np.sum(x_next))),
            "y_top0_after": float(max(0.0, 1.0 - np.sum(y_next))),

            "E_light_docs_after": float(np.dot(x_next, np.asarray(self.doc_choices_light, dtype=float))),
            "E_heavy_docs_after": float(np.dot(y_next, np.asarray(self.doc_choices_heavy, dtype=float))),
        }
