# -*- coding: utf-8 -*-
"""
Paper-aligned bandit convex optimization for AdaRAG (Algorithm 1).

Alignment points:
- decision vector z = [x(1..n), y(1..n), p] in R^{2n+1}
- one-point bandit gradient estimate using random unit vector u
  (paper Eq. (7)-(8))
- unperturbed base point z_hat_t is updated on (1-gamma)Z
  (paper Eq. (9)-(10))
- lambda update uses the first-order correction term
  (paper Eq. (11))
- executed decision is z_t = z_hat_t + delta * u
  (no post-perturbation projection in strict paper mode)

Notes:
- To keep strict alignment with the paper, gamma should satisfy gamma = delta / r,
  where r is the inscribed-ball radius used in the analysis.
- If strict_perturbation_check=True, we verify that the executed perturbation already
  lies in Z by checking whether projection changes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from adarag.utils import project_z_to_domain


@dataclass
class OptimizerState:
    z_hat: np.ndarray
    lambda_: float
    t: int


class AdaRAGBanditOptimizer:
    def __init__(
        self,
        *,
        n_docs: int,
        Q_target: float,
        alpha: float = 0.05,
        mu: float = 0.05,
        delta: float = 0.05,
        gamma: Optional[float] = None,
        radius_r: Optional[float] = None,
        seed: int = 0,
        strict_perturbation_check: bool = True,
        feasibility_tol: float = 1e-8,
    ) -> None:
        self.n_docs = int(n_docs)
        self.dim = 2 * self.n_docs + 1

        self.Q_target = float(Q_target)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.delta = float(delta)
        self.radius_r, self.gamma = self._resolve_radius_and_gamma(
            delta=self.delta,
            gamma=gamma,
            radius_r=radius_r,
        )

        self.strict_perturbation_check = bool(strict_perturbation_check)
        self.feasibility_tol = float(feasibility_tol)
        self.rng = np.random.RandomState(seed)

        # Initialize z_hat inside (1-gamma)Z using the correct scaled projection:
        # Proj_{cZ}(a) = c * Proj_Z(a / c), c > 0.
        mass = 0.95
        x0 = np.ones(self.n_docs, dtype=float) * (mass / self.n_docs)
        y0 = np.ones(self.n_docs, dtype=float) * (mass / self.n_docs)
        p0 = 0.2
        z0 = np.concatenate([x0, y0, np.array([p0], dtype=float)])
        z0 = self._project_to_scaled_domain(z0, 1.0 - self.gamma)

        self.state = OptimizerState(z_hat=z0, lambda_=0.0, t=0)

    @staticmethod
    def _resolve_radius_and_gamma(
        *,
        delta: float,
        gamma: Optional[float],
        radius_r: Optional[float],
    ) -> Tuple[float, float]:
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}.")

        # Preferred: provide the analysis radius r, then compute gamma = delta / r.
        if radius_r is not None:
            r = float(radius_r)
            if r <= 0.0:
                raise ValueError(f"radius_r must be positive, got {radius_r}.")
            gamma_from_r = float(delta / r)
            if not (0.0 <= gamma_from_r < 1.0):
                raise ValueError(
                    f"delta / radius_r must lie in [0, 1). Got delta={delta}, "
                    f"radius_r={r}, gamma={gamma_from_r}."
                )
            if gamma is not None and not np.isclose(float(gamma), gamma_from_r, atol=1e-12, rtol=1e-9):
                raise ValueError(
                    f"Inconsistent gamma and radius_r: gamma={gamma}, but delta/r={gamma_from_r}."
                )
            return r, gamma_from_r

        # Fallback: if only gamma is given, algebraically recover r = delta / gamma.
        # This keeps gamma = delta / r internally consistent, but the user should
        # still ensure that this r matches the actual inscribed-ball radius of Z.
        if gamma is not None:
            g = float(gamma)
            if not (0.0 < g < 1.0):
                raise ValueError(f"gamma must lie in (0, 1) when radius_r is absent, got {gamma}.")
            r = float(delta / g)
            return r, g

        raise ValueError(
            "Need either radius_r (preferred) or gamma. For strict paper alignment, "
            "set radius_r and let gamma be computed as delta / radius_r."
        )

    def _sample_unit_sphere(self) -> np.ndarray:
        u = self.rng.randn(self.dim).astype(float)
        norm = float(np.linalg.norm(u))
        if norm <= 1e-12:
            u = np.zeros(self.dim, dtype=float)
            u[-1] = 1.0
            return u
        return u / norm

    def _project_to_scaled_domain(self, z: np.ndarray, scale: float) -> np.ndarray:
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"scale must lie in (0, 1], got {scale}.")
        z = np.asarray(z, dtype=float)
        return scale * project_z_to_domain(z / scale, self.n_docs)

    def _projection_gap_to_Z(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        z_proj = project_z_to_domain(z, self.n_docs)
        return float(np.linalg.norm(z_proj - z))

    def propose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propose the paper-aligned perturbed action:
            z_t = z_hat_t + delta * u.

        Returns:
            z_perturbed: executed perturbed decision z_t
            u:           sampled unit direction
            z_hat:       current base point z_hat_t
        """
        z_hat = self.state.z_hat
        u = self._sample_unit_sphere()
        z_perturbed = z_hat + self.delta * u

        if self.strict_perturbation_check:
            gap = self._projection_gap_to_Z(z_perturbed)
            if gap > self.feasibility_tol:
                raise RuntimeError(
                    "Perturbed action z_t is not in Z under strict paper mode. "
                    f"projection_gap={gap:.6e}, tol={self.feasibility_tol:.6e}. "
                    "Check whether the radius_r used for gamma=delta/r is correct, or whether "
                    "project_z_to_domain matches the paper's feasible domain Z."
                )

        return z_perturbed, u, z_hat

    def update(
        self,
        *,
        z_perturbed: np.ndarray,
        u: np.ndarray,
        d_feedback: float,
        q_feedback: float,
    ) -> Dict[str, Any]:
        st = self.state
        dim = self.dim

        z_hat_old = np.asarray(st.z_hat, dtype=float).copy()

        # Eq. (7)-(8): one-point gradient estimators at z_hat_t using the feedback
        # observed from the executed perturbed action z_t = z_hat_t + delta*u.
        grad_d_hat = (dim / self.delta) * float(d_feedback) * np.asarray(u, dtype=float)
        grad_q_hat = (dim / self.delta) * float(q_feedback) * np.asarray(u, dtype=float)

        # Eq. (10): one-point Lagrangian gradient.
        grad_L_hat = grad_d_hat - st.lambda_ * grad_q_hat

        # Eq. (9): z_hat_{t+1} = Proj_{(1-gamma)Z}( z_hat_t - alpha * grad_L_hat )
        z_update_raw = z_hat_old - self.alpha * grad_L_hat
        z_next = self._project_to_scaled_domain(z_update_raw, 1.0 - self.gamma)

        # Eq. (11): lambda_{t+1} = [lambda_t + mu*(Q - q_t(z_t)
        #                             - grad_q_hat^T (z_hat_{t+1} - z_hat_t))]_+
        corr = float(np.dot(grad_q_hat, (z_next - z_hat_old)))
        lam_next = st.lambda_ + self.mu * (self.Q_target - float(q_feedback) - corr)
        lam_next = float(max(0.0, lam_next))

        st.t += 1
        st.z_hat = z_next
        st.lambda_ = lam_next

        return {
            "t": st.t,
            "d_feedback": float(d_feedback),
            "q_feedback": float(q_feedback),
            "lambda": float(lam_next),
            "gamma": float(self.gamma),
            "radius_r": float(self.radius_r),
            "z_hat_prev": z_hat_old.tolist(),
            "z_hat": z_next.tolist(),
            "z_perturbed": np.asarray(z_perturbed, dtype=float).tolist(),
            "perturbation_norm": float(np.linalg.norm(np.asarray(z_perturbed, dtype=float) - z_hat_old)),
            "projection_gap_perturbed": float(self._projection_gap_to_Z(np.asarray(z_perturbed, dtype=float))),
            "corr": corr,
        }

    def unpack_z(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        z = np.asarray(z, dtype=float)
        x = z[: self.n_docs]
        y = z[self.n_docs : 2 * self.n_docs]
        p = float(z[-1])
        return x, y, p
