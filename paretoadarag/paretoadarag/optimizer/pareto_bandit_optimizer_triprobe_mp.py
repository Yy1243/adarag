from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence, List

import numpy as np


@dataclass
class ParetoState:
    w_hat: np.ndarray
    lambda_: float
    t: int


def _as_doc_choices(v: Optional[Sequence[int]], fallback_n: int) -> List[int]:          #v是用户指定的文档索引列表， fallback_n是当用户未指定时的默认数量
    if v is None:
        return list(range(1, int(fallback_n) + 1))
    out = [int(x) for x in v]       #如果输入的是字符串列表或者其他可迭代对象，把它转换成整数列表
    if not out:
        raise ValueError("doc choices must not be empty.")
    if min(out) < 1:
        raise ValueError(f"doc choices must be >= 1, got {out}")
    return out


def _project_prob_mass(v: np.ndarray, cap: float = 1.0) -> np.ndarray:          #投影质量概率，把一个任意向量变成合法的概率分布片段，允许总和小于1（剩余部分表示"什么都不选"的概率）
    """
    Project to original AdaRAG-style probability-mass domain:
        0 <= v_i <= 1, sum(v_i) <= cap.

    The remaining probability mass, 1 - sum(v), is top0.
    """
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0.0, 1.0)
    cap = float(np.clip(cap, 0.0, 1.0))             #cap的防御编程，cap表示文档总和概率上限，可能会因为扰动导致cap不一定能够为1，总之是小于等于1的

    s = float(v.sum())
    if s > cap:
        v = v / max(s, 1e-12) * cap
    return v


def _init_prob_mass(policy: str, n: int, total_mass: float) -> np.ndarray:      #total_mass = 0表示全部概率给 top0（一定不检索），total_mass = 1：全部概率给 top-k
    """
    Initialize probabilities over nonzero top-k choices.

    total_mass < 1 leaves top0 probability:
        p(top0) = 1 - total_mass.
    """
    policy = str(policy or "increasing").strip().lower()
    if n <= 0:
        raise ValueError("n must be positive.")

    total_mass = float(np.clip(total_mass, 0.0, 1.0))

    if policy == "uniform":                      # [0.25, 0.25, 0.25, 0.25]
        base = np.full(n, 1.0 / n, dtype=float)         
    elif policy == "increasing":                #  [0.1, 0.2, 0.3, 0.4]
        base = np.arange(1, n + 1, dtype=float)
        base = base / float(base.sum())
    elif policy in ("fixed_max", "max"):            # [0, 0, 0, 1]
        base = np.zeros(n, dtype=float)
        base[-1] = 1.0
    elif policy in ("fixed_min", "min"):             # [1, 0, 0, 0]
        base = np.zeros(n, dtype=float)
        base[0] = 1.0
    else:
        raise ValueError(f"Unknown init_doc_policy={policy}")

    return total_mass * base        #关键一步，因为要允许有top_0的概率，所以要把分配给实际文档的概率质量控制在total_mass以内，剩余的概率质量就是top_0的概率了


class ParetoAdaRAGBanditOptimizerTriProbeMP:
    """
    ParetoAdaRAG-3P optimizer with original top0-enabled AdaRAG domain.

    Decision vector:
        w = [x, y, p, nu]

    x:
        probabilities over light doc choices; top0 probability = 1 - sum(x)
    y:
        probabilities over heavy doc choices; top0 probability = 1 - sum(y)
    p:
        heavy retrieval proportion
    nu:
        CVaR auxiliary threshold
    """

    def __init__(
        self,
        *,
        n_docs: Optional[int] = None,           #统一指定light/heavy文档数量的参数，优先级低于n_docs_light/n_docs_heavy
        n_docs_light: Optional[int] = None,         #轻量检索最大文档数
        n_docs_heavy: Optional[int] = None,
        doc_choices_light: Optional[Sequence[int]] = None,      #轻量可选topn的数量列表
        doc_choices_heavy: Optional[Sequence[int]] = None,
        init_doc_policy: str = "increasing",        #概率初始化策略
        init_top0_prob: float = 0.0,              #初始top0概率，即1-sum(init_prob_mass)，优先级高于init_doc_policy，即先设定init_doc_policy然后生成每个文档概率分布。
        q_target: float,
        alpha: float = 0.0005,          #学习率α
        mu: float = 0.005,                  #对偶变量λ的更新步长μ
        delta: float = 0.05,                    #一阶梯度估计的扰动幅度δ    
        gamma: float = 0.05,                        #域缩放参数γ，越大越保守地使用内缩域进行一阶估计和更新，γ=0即不使用内缩域
        nu_min: float = 0.0,
        nu_max: float = 60.0,
        nu_init: float = 8.0,                       #CVaR辅助变量νt的初始值
        p_init: float = 0.35,               #重检比例p的初始值
        p_min: float = 0.0,
        p_max: float = 1.0,
        seed: int = 0,
        normalize_enabled: bool = False,
        d_ref: float = 1.0,
        psi_ref: float = 1.0,
        g_ref: float = 1.0,
    ) -> None:
        if n_docs_light is None:            #未单独指定轻重检索数量用默认的整体的数量
            n_docs_light = int(n_docs or 4)
        if n_docs_heavy is None:
            n_docs_heavy = int(n_docs or 6)

        self.doc_choices_light = _as_doc_choices(doc_choices_light, int(n_docs_light))
        self.doc_choices_heavy = _as_doc_choices(doc_choices_heavy, int(n_docs_heavy))

        self.n_docs_light = len(self.doc_choices_light)
        self.n_docs_heavy = len(self.doc_choices_heavy)
        self.max_light_docs = max(self.doc_choices_light)
        self.max_heavy_docs = max(self.doc_choices_heavy)
        self.n_docs = self.n_docs_light + self.n_docs_heavy             #对核心逻辑没影响，挂名属性

        self.dim_z = self.n_docs_light + self.n_docs_heavy + 1
        self.dim_w = self.dim_z + 1

        self.q_target = float(q_target)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)

        self.p_min = float(np.clip(p_min, 0.0, 1.0))        #裁剪函数，如果p_min<0则设为0，如果p_min>1则设为1，否则不变
        self.p_max = float(np.clip(p_max, self.p_min, 1.0))

        self.normalize_enabled = bool(normalize_enabled)
        self.d_ref = float(d_ref)
        self.psi_ref = float(psi_ref)
        self.g_ref = float(g_ref)

        self.rng = np.random.RandomState(seed)

        init_mass = 1.0 - float(np.clip(init_top0_prob, 0.0, 1.0))          #分配给实际检索文档的总概率质量
        x0 = _init_prob_mass(init_doc_policy, self.n_docs_light, init_mass)             #x0是初始轻检索决策向量，不是单纯的一个数字
        y0 = _init_prob_mass(init_doc_policy, self.n_docs_heavy, init_mass)
        p0 = np.array([float(np.clip(p_init, self.p_min, self.p_max))], dtype=float)
        nu0 = np.array([nu_init], dtype=float)

        w0 = np.concatenate([x0, y0, p0, nu0])      #把四个独立组件拼接成一个长向量
        self.state = ParetoState(
            w_hat=self._project_to_scaled_domain(w0, 1.0 - self.gamma),             #投影后的 w0，当前决策估计
            lambda_=0.0,
            t=0,
        )

    def _sample_unit_sphere(self) -> np.ndarray:
        u = self.rng.randn(self.dim_w).astype(float)        #randn是random normal 的缩写，生成服从标准正态分布 N(0,1) 的随机数
        norm = float(np.linalg.norm(u))             #np.linalg.norm(u)计算向量u的欧几里得范数，即sqrt(sum(u_i^2))，也就是u的长度
        if norm < 1e-12:
            u[:] = 0.0
            u[-1] = 1.0     #设最后一个为1，保证整体范数长度为1，然后的话这一次采样只对最后一个进行扰动
            return u
        return u / norm         #归一化后的u，即在dim_w维空间中均匀分布的单位向量

    def _project_to_domain(self, w: np.ndarray) -> np.ndarray:          # 投影到标准可行域，执行探针，对应论文的公式5
        w = np.asarray(w, dtype=float)          #把列表、元组等转成 np.ndarray即numpy数组。
        x = _project_prob_mass(w[: self.n_docs_light], cap=1.0)
        y = _project_prob_mass(w[self.n_docs_light: self.n_docs_light + self.n_docs_heavy], cap=1.0)
        p = float(np.clip(w[-2], self.p_min, self.p_max))
        nu = float(np.clip(w[-1], self.nu_min, self.nu_max))
        return np.concatenate([x, y, np.array([p, nu], dtype=float)])

    def _project_to_scaled_domain(self, w: np.ndarray, scale: float) -> np.ndarray:            #先调用前者，再进一步收缩，把变量压入更小的安全区域。应用于梯度更新后。对应论文的公式12更新梯度那里
        """
        Shrunk domain used by the one-point perturbation logic:
            sum(x) <= scale, sum(y) <= scale,
            p in [p_min, p_min + scale*(p_max-p_min)],
            nu in [nu_min, scale*nu_max].
        """
        scale = float(np.clip(scale, 0.0, 1.0))             #scale参数控制内缩域的大小，假设 gamma = 0.05，scale = 0.95。
        w0 = self._project_to_domain(w)
        x, y, p, nu = self.unpack_w(w0)

        x = _project_prob_mass(x, cap=scale)        #收紧范围
        y = _project_prob_mass(y, cap=scale)

        p_upper = self.p_min + scale * (self.p_max - self.p_min)        #p的上界也要收紧，原来是 [p_min, p_max]，现在是 [p_min, p_min + scale*(p_max-p_min)]，当scale=0.95时，p的上界就从p_max收紧到接近p_max的位置。
        p = float(np.clip(p, self.p_min, p_upper))                  #重检索概率收紧，如果p_min也为0的话，那么可以直接写成下面p = float(np.clip(p, self.p_min, scale*self.p_max))，这种写法是为了保证在p_min不为0的情况下也能正确收紧。
        nu = float(np.clip(nu, self.nu_min, scale * self.nu_max))           #辅助变量收紧

        return np.concatenate([x, y, np.array([p, nu], dtype=float)])

    def unpack_w(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        w = np.asarray(w, dtype=float)
        x = np.asarray(w[: self.n_docs_light], dtype=float)
        y = np.asarray(w[self.n_docs_light: self.n_docs_light + self.n_docs_heavy], dtype=float)
        p = float(w[-2])
        nu = float(w[-1])
        return x, y, p, nu

    def propose_triple(self):
        w_hat = self.state.w_hat.copy()         #创建副本保证不影响原先的状态，w_hat 和 self.state.w_hat 是同一个对象，虽然这段代码没有直接修改 w_hat，但防御性编程确保万一后续有人加了修改操作，不会破坏原始状态。w_hat的作用可以构造扰动点：w_hat + delta * u。

        u_d = self._sample_unit_sphere()        #采样三个随机方向，分别用于d、psi、g的梯度估计，要求三者之间不能平行，否则会导致梯度估计不稳定，所以如果采样到的方向之间的余弦相似度过高，就重新采样直到满足条件。
        u_psi = self._sample_unit_sphere()
        u_g = self._sample_unit_sphere()

        for _ in range(8):
            if abs(float(np.dot(u_d, u_psi))) > 0.999:          #保证不是平行的情况
                u_psi = self._sample_unit_sphere()
            if abs(float(np.dot(u_d, u_g))) > 0.999 or abs(float(np.dot(u_psi, u_g))) > 0.999:
                u_g = self._sample_unit_sphere()

        # Keep original-paper style: use sampled unit directions in the estimator.
        # Projection is retained as a practical safety guard.
        w_exec_d = self._project_to_domain(w_hat + self.delta * u_d)            #梯度估计乘的是原始的，但在实际直击执行的时候用的是投影后的决策向量，保证执行的决策是合法的
        w_exec_psi = self._project_to_domain(w_hat + self.delta * u_psi)
        w_exec_g = self._project_to_domain(w_hat + self.delta * u_g)

        return w_exec_d, u_d, w_exec_psi, u_psi, w_exec_g, u_g, w_hat

    def _one_point_grad(self, value: float, u_dir: np.ndarray) -> np.ndarray:               #梯度估计公式
        return (self.dim_w / self.delta) * float(value) * np.asarray(u_dir, dtype=float)

    @staticmethod           #不需要访问self 的任何属性或方法，所以可以声明为静态方法
    def _solve_theta(
        grad_d: np.ndarray,
        grad_psi: np.ndarray,
        grad_g: np.ndarray,
        lam: float,
    ) -> Tuple[float, float]:
        a = np.asarray(grad_d, dtype=float)
        b = np.asarray(grad_psi, dtype=float)
        c = float(lam) * np.asarray(grad_g, dtype=float)        #约束惩罚方向（已乘 λ）

        delta = a - b
        base = b + c
        denom = float(np.dot(delta, delta))     #||Δ||²

        if denom < 1e-12:           #意味着两者差值很近，两个梯度几乎同方向。此时无法通过方向差异来判断该偏重哪个目标，最优解不确定。
            theta1 = 0.5
        else:
            theta1 = -float(np.dot(base, delta)) / denom            #推导算出来的值
            theta1 = float(np.clip(theta1, 0.0, 1.0))

        theta2 = 1.0 - theta1
        return theta1, theta2

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:             #计算两个向量的余弦相似度，衡量它们的方向相似程度，值域为[-1, 1]，越接近1表示越同向，越接近-1表示越反向，接近0表示几乎正交。
        denom = max(1e-12, float(np.linalg.norm(a) * np.linalg.norm(b)))            #利用max函数是为了防止除以0，两个向量长度的乘积 ∥a∥⋅∥b∥ 
        return float(np.dot(a, b) / denom)                  #np.dot(a, b)是点积函数

    def update_triple(
        self,
        *,
        u_dir_d: np.ndarray,            #探针的三个随机方向，决策向量 w_hat 的扰动方向，分别用于 d、psi、g 的梯度估计，原始方向，因为是用于梯度估计的，所以不需要投影，保持原样即可。
        u_dir_psi: np.ndarray,
        u_dir_g: np.ndarray,
        d_feedback: float,
        psi_feedback: float,
        q_feedback: float,
    ) -> Dict[str, float]:
        st = self.state             #当前状态对象
        w_prev = st.w_hat.copy()             # 复制当前决策（防止修改原状态）

        g_feedback = self.q_target - float(q_feedback)

        if self.normalize_enabled:
            d_value = float(d_feedback) / max(1e-12, self.d_ref)
            psi_value = float(psi_feedback) / max(1e-12, self.psi_ref)
            g_value = float(g_feedback) / max(1e-12, self.g_ref)
        else:
            d_value = float(d_feedback)
            psi_value = float(psi_feedback)
            g_value = float(g_feedback)

        grad_d = self._one_point_grad(d_value, u_dir_d)             #零阶梯度估计得到三采样的梯度值
        grad_psi = self._one_point_grad(psi_value, u_dir_psi)
        grad_g = self._one_point_grad(g_value, u_dir_g)

        theta1, theta2 = self._solve_theta(grad_d, grad_psi, grad_g, st.lambda_)
        v_dir = theta1 * grad_d + theta2 * grad_psi + st.lambda_ * grad_g       #论文算法中的11式

        w_next = self._project_to_scaled_domain(w_prev - self.alpha * v_dir, 1.0 - self.gamma)

        correction = float(np.dot(grad_g, w_next - w_prev))
        lambda_next = max(0.0, st.lambda_ + self.mu * (g_value - correction))           #对偶变量更新

        st.w_hat = w_next
        st.lambda_ = float(lambda_next)
        st.t += 1

        x_prev, y_prev, p_prev, nu_prev = self.unpack_w(w_prev)
        x_next, y_next, p_next, nu_next = self.unpack_w(w_next)

        return {
            "t": float(st.t),
            "theta1": float(theta1),
            "theta2": float(theta2),
            "lambda": float(st.lambda_),

            "d_feedback": float(d_feedback),
            "psi_feedback": float(psi_feedback),
            "q_feedback": float(q_feedback),
            "g_feedback": float(g_feedback),

            "perturbation_norm_d": float(np.linalg.norm(self.delta * u_dir_d)),
            "perturbation_norm_psi": float(np.linalg.norm(self.delta * u_dir_psi)),
            "perturbation_norm_g": float(np.linalg.norm(self.delta * u_dir_g)),

            "grad_d_norm": float(np.linalg.norm(grad_d)),
            "grad_psi_norm": float(np.linalg.norm(grad_psi)),
            "grad_g_norm": float(np.linalg.norm(grad_g)),
            "v_norm": float(np.linalg.norm(v_dir)),
            "update_norm": float(np.linalg.norm(w_next - w_prev)),

            "u_cosine_d_psi": self._cosine(u_dir_d, u_dir_psi),
            "u_cosine_d_g": self._cosine(u_dir_d, u_dir_g),
            "u_cosine_psi_g": self._cosine(u_dir_psi, u_dir_g),

            "grad_cosine_d_psi": self._cosine(grad_d, grad_psi),
            "grad_cosine_d_g": self._cosine(grad_d, grad_g),
            "grad_cosine_psi_g": self._cosine(grad_psi, grad_g),

            "p_hat_before": float(p_prev),
            "nu_hat_before": float(nu_prev),
            "p_hat_after": float(p_next),
            "nu_hat_after": float(nu_next),

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
