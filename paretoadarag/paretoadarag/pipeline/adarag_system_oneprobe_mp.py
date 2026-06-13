# /home/yy/adarag_repro/paretoadarag/pipeline/adarag_system_oneprobe_mp.py
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import concurrent.futures as cf
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

from adarag.data import QAItem
from adarag.eval.evaluator import batch_accuracy
from adarag.pipeline.prompt_builder import build_prompt

from scripts.eval_light_heavy_retrieval_generation import (
    build_light_retriever,
    build_heavy_retriever,
    build_llm,
)


def _as_doc_choices(v: Optional[Sequence[int]], fallback_n: int) -> List[int]:
    if v is None:
        return list(range(1, int(fallback_n) + 1))
    out = [int(x) for x in v]
    if not out:
        raise ValueError("doc choices must not be empty.")
    if min(out) < 1:
        raise ValueError(f"doc choices must be >= 1, got {out}")
    return out


def _get_field(x: Any, names: List[str], default=None):         #names相当于备选项，然后x可以是一个字典或者一个对象，函数会按照names的顺序去x里找对应的字段，如果找到就返回对应的值，如果找不到就返回默认值default。这个函数的作用是为了方便地从不同结构的数据中提取出我们需要的信息。
    if x is None:
        return default
    if isinstance(x, dict):
        for n in names:
            if n in x and x[n] is not None:
                return x[n]
        return default
    for n in names:
        if hasattr(x, n):
            v = getattr(x, n)
            if v is not None:
                return v
    return default


def _as_answers(a: Any) -> List[str]:
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    if isinstance(a, (list, tuple)):
        return [str(z) for z in a if z is not None]
    return [str(a)]


def _make_qa(x: Any) -> QAItem:
    q = _get_field(x, ["q", "question", "query"], default="") or ""
    a = _get_field(x, ["a", "answers", "answer", "gold", "ground_truth"], default=None)
    qid = _get_field(x, ["qid", "id", "example_id"], default="") or ""
    return QAItem(q=str(q), a=_as_answers(a), qid=str(qid))


def _norm(s: str) -> str:           #标准化输出，将字符串转换为小写，去掉首尾空格，并将连续的空格替换为单个空格。这有助于在比较文本时忽略大小写和多余的空格，从而更准确地判断文本是否包含某些内容。
    return " ".join(str(s).lower().strip().split())


def _to_text(x: Any) -> str:            #处理 LLM 返回的不确定格式的输出，尝试从中提取出文本内容。它支持字符串、字典（从特定字段提取）、列表（取第一个元素）等多种输入格式，并将其转换为纯文本字符串。这对于处理 LLM 可能返回的复杂结构非常有用，可以确保我们最终得到一个可用于评估或其他处理的文本字符串。
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("text", "output", "answer", "generated_text"):          #text优先级最高的设定
            if key in x:
                return str(x[key])
        return str(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_text(x[0])
    return str(x)


def _postprocess_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"```.*?```", "", s, flags=re.S)     #re.sub(pattern, replacement, string, flags):在字符串 s 中找到所有匹配 pattern 的部分，替换成 replacement（这里是空串 ""，即删除）。?表示非贪婪的意思，即遇到第一个成对的 ``` 就停止然后继续寻找匹配，而不是一直匹配到最后一个 ```。flags=re.S 的作用是. 默认不匹配换行符，不加 re.S，. 遇到换行就停了，匹配失败。加了 re.S，. 就能匹配换行了。r""表示原始字符串，不需要对反斜杠进行转义。为什么要对三引号里面的内容给进行去除呢：因为三引号 ```多是 Markdown 代码块的标记。LLM 输出里经常出现这种格式。当前这种情况下不支持嵌套的三引号，如果有嵌套的三引号，可能会导致正则表达式匹配不正确。不过在实际使用中，LLM 输出里通常不会有复杂嵌套的三引号，所以这种简单的正则表达式已经足够了。
    m = re.search(r"final\s*answer\s*[:：]\s*(.*)$", s, flags=re.I | re.S)          #re.search(pattern, string, flags) 的作用，在字符串中搜索第一个匹配位置，返回匹配对象或 None。\s表示空白字符串的含义，*表示前面的元素可以重复零次或多次，[:：]表示匹配冒号（英文或中文）其中一个，(.*)表示匹配任意字符并捕获为一个组，$表示字符串的结尾。flags=re.I 表示忽略大小写，re.S 表示让 . 能匹配换行符。这个正则表达式的目的是尝试从 LLM 的输出中提取出 "final answer" 后面的内容，作为最终的答案。如果找到了这样的模式，就返回冒号后面的内容；如果没有找到，就返回整个字符串经过清理后的版本。
    if m:
        s = m.group(1)          #group(1) 表示获取第一个捕获组的内容，也就是冒号后面的部分，作为最终答案。group(0) 则表示整个匹配到的字符串，包括 "final answer" 和冒号等。通过使用 group(1)，我们可以直接获取我们关心的答案部分，而不需要额外的字符串处理。如何知道有几个组，方法是从左到右数正则里的 (，每个 ( 对应一个组。在这个正则表达式里，只有一个 (，所以只有一个组，就是我们想要提取的答案部分。
    return " ".join(s.strip().split())


def _doc_to_text(d: Any) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        title = str(d.get("title") or "").strip()
        text = str(d.get("text") or d.get("contents") or d.get("passage") or d.get("document") or "").strip()
        return f"{title}\n{text}" if title and text else (title or text)
    if hasattr(d, "full_text"):
        return str(d.full_text or "")
    title = str(getattr(d, "title", "") or "").strip()
    text = str(getattr(d, "text", "") or "").strip()
    return f"{title}\n{text}" if title and text else (title or text)


def _contains_any(text: str, gold: Any) -> bool:
    if text is None:
        return False
    t = _norm(text)
    if isinstance(gold, str):
        g = _norm(gold)
        return bool(g) and (g in t)
    if isinstance(gold, list):
        for a in gold:
            g = _norm(a)
            if g and (g in t):
                return True
    return False


def _oracle_hit(docs: List[Any], gold: Any) -> bool:
    if not docs:
        return False
    texts = [_norm(_doc_to_text(d)) for d in docs]
    if isinstance(gold, str):
        g = _norm(gold)
        return any(g and (g in t) for t in texts)
    if isinstance(gold, list):
        for a in gold:
            g = _norm(a)
            if any(g and (g in t) for t in texts):
                return True
    return False


def _safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]


def _project_prob_mass(v: np.ndarray) -> np.ndarray:            #投影质量概率到合理的区间内，和前面的paretoadarag/optimizer/pareto_bandit_optimizer_triprobe_mp.py略有不同因为这里只需要负责执行时采样合法就可以了
    """
    Project to {0 <= v_i <= 1, sum(v_i) <= 1}.
    The remaining probability mass 1-sum(v) is top0.
    """
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0.0, 1.0)
    s = float(v.sum())
    if s > 1.0:
        v = v / max(s, 1e-12)
    return v


def sample_doc_count_from_choices(              
    probs: np.ndarray,              #各选项概率分布
    doc_choices: Sequence[int],         #可选文档数量列表
    rng: np.random.RandomState,             #随机数生成器
) -> int:
    """
    Sample from top0 plus configured nonzero top-k choices.

    Example:
        doc_choices=[5,6,7,8,9,10]
        probs=[p5,p6,p7,p8,p9,p10], sum(probs)<=1
        top0 probability = 1 - sum(probs)
    """
    choices = np.asarray([int(x) for x in doc_choices], dtype=int)
    p = _project_prob_mass(np.asarray(probs[: len(choices)], dtype=float))
    p0 = max(0.0, 1.0 - float(p.sum()))

    cat = np.concatenate([[p0], p])
    total = float(cat.sum())
    if total <= 1e-12:
        return 0
    cat = cat / total

    all_choices = np.concatenate([[0], choices])            #拼接概率包括文档为0的概率
    return int(rng.choice(all_choices, p=cat))          #按cat概率进行随机抽样得到相应的文档数量


def _mix32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x & 0xFFFFFFFF


def _rng_for(seed: int, slot_id: int, i: int, is_heavy: bool) -> np.random.RandomState:
    x = int(seed)
    x ^= (int(slot_id) * 0x9E3779B1) & 0xFFFFFFFF           
    x ^= (int(i) * 0x85EBCA6B) & 0xFFFFFFFF             #混入 slot_id和查询索引以确保不同的查询有不同的随机数序列
    if is_heavy:
        x ^= 0x27D4EB2F                 #混入heavy 路径额外扰动，三重特征保证每一个问题都有一个独特的随机数生成器，且heavy和light路径的随机数生成器也不同，避免了不同查询和路径之间的随机数序列相关性，从而提高了采样的多样性和公平性。
    return np.random.RandomState(_mix32(x))                     #用混合后的种子初始化RNG，mix32函数是一个简单的整数混合函数，可以将输入的整数 x 混合成一个看起来更随机的整数(即使输入只差了 1，输出也会天差地别)。通过这种方式，我们可以从一个初始种子生成多个不同的随机数生成器实例，每个实例都有一个独特的种子，确保了不同查询和路径之间的随机数序列的独立性和多样性。返回的是一个随机数生成对象，专门用来生产随机数的实例，可以调用它的各种方法来生成随机数。


def _softplus_kappa(x: np.ndarray, kappa: float) -> np.ndarray:         #x在这个函数里代表的是每个请求的延迟相对于阈值 nu 的超出量（excess），是一个 numpy 数组。
    """
    Stable softplus approximation:
        s_kappa(x) = log(1 + exp(kappa*x)) / kappa
    """
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    z = kappa * np.asarray(x, dtype=float)          #这一步是算kx
    return np.logaddexp(0.0, z) / kappa         #np.logaddexp(a, b) = log(exp(a) + exp(b))故logaddexp(0, z) = log(1 + exp(z)) ，故返回的就是论文中的sκ(x)


def _cvar_surrogate(
    tau: List[float],
    nu: float,
    beta: float,
    kappa: float = 5.0,
) -> float:
    """
    Smoothed CVaR surrogate:
        psi = nu + 1 / ((1-beta)|U|) * sum s_kappa(tau_i - nu)
    """
    if not tau:
        return float(nu)
    arr = np.asarray(tau, dtype=float)          #arr = np.asarray(tau)每个元素是请求的延时，是一个numpy数组
    excess = arr - float(nu)
    smooth_excess = _softplus_kappa(excess, kappa=float(kappa))
    return float(float(nu) + np.mean(smooth_excess) / max(1e-12, 1.0 - float(beta)))        #np.mean(...) 就是 sum(...) / len(...)里面包含了求和和除以整体请求数，论文中的公式6。


class AdaRAGSystemOneProbeMP:
    def __init__(
        self,
        *,
        light_retriever,
        heavy_retriever,
        llm,
        doc_choices_light: Sequence[int],
        doc_choices_heavy: Sequence[int],
        beta: float = 0.95,
        acc_mode: str = "token",
        prompt_max_doc_chars: int = 1600,
        seed: int = 0,
        force_heavy: bool = False,
        judge_llm=None,
        exec_mode: str = "overlap",
        heavy_max_workers: int = 2,
        llm_max_workers: int = 8,
        sim_score_topn: int = 4,
        q_target: float = 0.60,
        cvar_kappa: float = 5.0,
    ) -> None:
        self.light_retriever = light_retriever
        self.heavy_retriever = heavy_retriever
        self.llm = llm
        self.judge_llm = judge_llm

        self.doc_choices_light = [int(x) for x in doc_choices_light]
        self.doc_choices_heavy = [int(x) for x in doc_choices_heavy]
        self.n_docs_light = len(self.doc_choices_light)
        self.n_docs_heavy = len(self.doc_choices_heavy)
        self.max_light_docs = max(self.doc_choices_light)
        self.max_heavy_docs = max(self.doc_choices_heavy)

        self.beta = float(beta)
        self.acc_mode = str(acc_mode)
        self.prompt_max_doc_chars = int(prompt_max_doc_chars)
        self.cvar_kappa = float(cvar_kappa)

        self.seed = int(seed)
        self.force_heavy = bool(force_heavy)
        self.exec_mode = str(exec_mode).strip().lower()
        if self.exec_mode not in ("serial", "overlap"):
            raise ValueError(f"exec_mode must be serial or overlap, got {exec_mode}")

        self.heavy_max_workers = int(max(1, heavy_max_workers))
        self.llm_max_workers = int(max(1, llm_max_workers))
        self.sim_score_topn = int(max(1, sim_score_topn))
        self.q_target = float(q_target)

    def split_w(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        AdaRAG native policy vector is w = [x, y, p].

        This system also accepts w = [x, y, p, nu] for backward compatibility
        with older scripts. In AdaRAG, nu is not a decision variable; it is only
        used to report psi_value and does not affect routing, retrieval,
        document sampling, generation, average latency, empirical p95 latency,
        or quality.
        """
        w = np.asarray(w, dtype=float)
        base_dim = self.n_docs_light + self.n_docs_heavy

        if len(w) == base_dim + 1:
            # AdaRAG native format: [x, y, p]
            x = np.asarray(w[: self.n_docs_light], dtype=float)
            y = np.asarray(
                w[self.n_docs_light: self.n_docs_light + self.n_docs_heavy],
                dtype=float,
            )
            p = float(w[-1])
            nu = 4.0
            return x, y, p, nu

        if len(w) == base_dim + 2:
            # Backward-compatible format: [x, y, p, nu]
            x = np.asarray(w[: self.n_docs_light], dtype=float)
            y = np.asarray(
                w[self.n_docs_light: self.n_docs_light + self.n_docs_heavy],
                dtype=float,
            )
            p = float(w[-2])
            nu = float(w[-1])
            if (not np.isfinite(nu)) or nu <= 0.0:
                nu = 4.0
            return x, y, p, nu

        raise ValueError(
            f"Invalid AdaRAG w length={len(w)}. Expected {base_dim + 1} for "
            f"[x,y,p] or {base_dim + 2} for backward-compatible [x,y,p,nu]."
        )

    def _select_heavy_mask(self, sim_sums: np.ndarray, p: float) -> np.ndarray:
        n = len(sim_sums)
        if n == 0:
            return np.zeros((0,), dtype=bool)
        if self.force_heavy:
            return np.ones((n,), dtype=bool)

        p = float(np.clip(p, 0.0, 1.0))
        k = int(np.round(p * n))
        if k <= 0:
            return np.zeros((n,), dtype=bool)

        order = np.argsort(sim_sums)
        mask = np.zeros((n,), dtype=bool)
        mask[order[:k]] = True
        return mask

    def _judge_correct(self, question: str, pred: str, gold: Any) -> bool:
        if self.judge_llm is None:
            return False

        gold_list = gold if isinstance(gold, list) else [gold]
        gold_list = [str(x) for x in gold_list if x is not None]

        prompt = (
            "You are a strict QA evaluator.\n"
            "Decide whether the model answer is semantically equivalent to ANY gold answer.\n"
            "Return ONLY 1 or 0.\n\n"
            f"Question: {question}\n"
            f"Gold answers: {gold_list}\n"
            f"Model answer: {pred}\n"
        )

        out = self.judge_llm.generate(prompt)
        out = _to_text(out).strip()
        m = re.search(r"[01]", out)
        return (m is not None) and (m.group(0) == "1")

    def run_probe(
        self,
        *,
        batch: List[QAItem],
        w: np.ndarray,
        slot_id: int,
        probe_type: str,
        judge_enabled: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        x, y, p, nu = self.split_w(w)
        B = len(batch)
        slot_t0 = time.perf_counter()

        if B == 0:
            return {
                "slot": int(slot_id),
                "probe_type": str(probe_type),
                "request_count": 0,
                "d_value": 0.0,
                "psi_value": float(nu),
                "g_value": self.q_target,
                "q_value": 0.0,
                "accuracy": 0.0,
                "judge_accuracy": None,
                "p95_latency_s": 0.0,
                "batch_wall_time_s": 0.0,
                "judge_wall_time_s": 0.0,
                "wall_time_with_judge_s": 0.0,
                "query_rows": [],
            }

        light_docs_all: List[List[Any]] = [[] for _ in range(B)]
        sim_sums = np.zeros(B, dtype=float)

        light_retrieve_s = np.zeros(B, dtype=float)
        heavy_retrieve_s = np.zeros(B, dtype=float)
        prompt_build_s = np.zeros(B, dtype=float)
        llm_generate_s = np.zeros(B, dtype=float)
        llm_client_wait_s = np.zeros(B, dtype=float)
        tau_e2e_s = np.zeros(B, dtype=float)

        route_heavy = np.zeros(B, dtype=bool)
        n_take_docs = np.zeros(B, dtype=int)
        prompt_chars = np.zeros(B, dtype=int)

        preds: List[str] = [""] * B
        golds: List[Any] = [qa.a for qa in batch]
        prompt_has_gold = np.zeros(B, dtype=int)
        correct_contains = np.zeros(B, dtype=int)
        oracle_light_full = np.zeros(B, dtype=int)
        oracle_heavy_full = np.zeros(B, dtype=int)

        # Batch light retrieval. All requests enter the slot at slot_t0 and wait for
        # the same batch-light-retrieval wall time. This time is NOT divided by B,
        # because tau_e2e_s is per-request wall-clock latency, not amortized cost.
        if not hasattr(self.light_retriever, "retrieve_batch"):
            raise RuntimeError(
                "light_retriever must implement retrieve_batch(queries). "
                "FaissHNSWRetriever has to be updated before running batch-light mode."
            )

        light_batch_t0 = time.perf_counter()
        queries = [qa.q for qa in batch]
        docs_batch, scores_batch = self.light_retriever.retrieve_batch(queries)
        light_batch_wall_time_s = float(time.perf_counter() - light_batch_t0)

        if len(docs_batch) != B or len(scores_batch) != B:
            raise RuntimeError(
                f"retrieve_batch returned wrong length: "
                f"docs={len(docs_batch)}, scores={len(scores_batch)}, B={B}"
            )

        # Every request in this slot waits until the batch light retrieval finishes.
        light_retrieve_s[:] = light_batch_wall_time_s

        for i, qa in enumerate(batch):
            docs_l = _safe_list(docs_batch[i])
            scores_l = _safe_list(scores_batch[i])
            light_docs_all[i] = docs_l

            score_used = scores_l[: self.sim_score_topn]
            sim_sums[i] = float(np.sum(score_used)) if len(score_used) else 0.0
            oracle_light_full[i] = int(_oracle_hit(docs_l[: self.max_light_docs], qa.a))

        heavy_mask = self._select_heavy_mask(sim_sums, p)
        route_heavy[:] = heavy_mask

        if verbose:
            print(
                f"[probe={probe_type}] slot={slot_id} B={B} "
                f"p={p:.4f} heavy={int(heavy_mask.sum())}/{B} "
                f"heavy_workers={self.heavy_max_workers} llm_workers={self.llm_max_workers}"
            )

        def _heavy_retrieve_timed(i: int, q: str):
            t0 = time.perf_counter()
            docs_h, scores_h = self.heavy_retriever.retrieve(q)
            return i, _safe_list(docs_h), _safe_list(scores_h), time.perf_counter() - t0

        def _generate_one(i: int, qa: QAItem, docs: List[Any], submit_rel_s: float, route: str):
            gen_start_rel = time.perf_counter() - slot_t0
            client_wait = max(0.0, gen_start_rel - submit_rel_s)

            t_prompt = time.perf_counter()
            prompt = build_prompt(qa.q, docs, max_doc_chars=self.prompt_max_doc_chars)
            pb_s = time.perf_counter() - t_prompt

            t_gen = time.perf_counter()
            pred_raw = self.llm.generate(prompt)
            gen_s = time.perf_counter() - t_gen

            finish_rel = time.perf_counter() - slot_t0
            pred = _postprocess_answer(_to_text(pred_raw))

            return {
                "i": i,
                "route": route,
                "prompt": prompt,
                "prompt_build_s": pb_s,
                "llm_generate_s": gen_s,
                "llm_client_wait_s": client_wait,
                "finish_rel_s": finish_rel,
                "pred": pred,
            }

        heavy_executor = cf.ThreadPoolExecutor(max_workers=self.heavy_max_workers)
        llm_executor = cf.ThreadPoolExecutor(max_workers=self.llm_max_workers)
        heavy_futures: Dict[cf.Future, int] = {}
        gen_futures: Dict[cf.Future, int] = {}

        def _submit_generation(i: int, qa: QAItem, docs: List[Any], route: str):
            submit_rel_s = time.perf_counter() - slot_t0
            fut = llm_executor.submit(_generate_one, i, qa, docs, submit_rel_s, route)
            gen_futures[fut] = i

        try:
            if self.exec_mode == "serial":
                for i, qa in enumerate(batch):
                    if bool(heavy_mask[i]) and self.heavy_retriever is not None:
                        t0 = time.perf_counter()
                        docs_h, _ = self.heavy_retriever.retrieve(qa.q)
                        heavy_retrieve_s[i] = time.perf_counter() - t0
                        docs_h = _safe_list(docs_h)
                        oracle_heavy_full[i] = int(_oracle_hit(docs_h[: self.max_heavy_docs], qa.a))

                        k = sample_doc_count_from_choices(y, self.doc_choices_heavy, _rng_for(self.seed, slot_id, i, True))
                        docs = docs_h[: min(k, len(docs_h))]
                        n_take_docs[i] = len(docs)
                        _submit_generation(i, qa, docs, "heavy")
                    else:
                        docs_l = light_docs_all[i]
                        k = sample_doc_count_from_choices(x, self.doc_choices_light, _rng_for(self.seed, slot_id, i, False))
                        docs = docs_l[: min(k, len(docs_l))]
                        n_take_docs[i] = len(docs)
                        _submit_generation(i, qa, docs, "light")
            else:
                for i, qa in enumerate(batch):
                    if bool(heavy_mask[i]) and self.heavy_retriever is not None:
                        heavy_futures[heavy_executor.submit(_heavy_retrieve_timed, i, qa.q)] = i

                for i, qa in enumerate(batch):
                    if not bool(heavy_mask[i]) or self.heavy_retriever is None:
                        docs_l = light_docs_all[i]
                        k = sample_doc_count_from_choices(x, self.doc_choices_light, _rng_for(self.seed, slot_id, i, False))
                        docs = docs_l[: min(k, len(docs_l))]
                        n_take_docs[i] = len(docs)
                        _submit_generation(i, qa, docs, "light")

                for fut in cf.as_completed(list(heavy_futures.keys())):
                    i = heavy_futures[fut]
                    qa = batch[i]
                    try:
                        _i, docs_h, _scores_h, h_s = fut.result()
                    except Exception as e:
                        print(f"[WARN] heavy retrieval failed slot={slot_id} i={i} qid={qa.qid}: {repr(e)}")
                        docs_h, h_s = [], 0.0

                    heavy_retrieve_s[i] = float(h_s)
                    oracle_heavy_full[i] = int(_oracle_hit(docs_h[: self.max_heavy_docs], qa.a))

                    k = sample_doc_count_from_choices(y, self.doc_choices_heavy, _rng_for(self.seed, slot_id, i, True))
                    docs = docs_h[: min(k, len(docs_h))]
                    n_take_docs[i] = len(docs)
                    _submit_generation(i, qa, docs, "heavy")

            for fut in cf.as_completed(list(gen_futures.keys())):
                i = gen_futures[fut]
                qa = batch[i]
                try:
                    res = fut.result()
                    pred = str(res["pred"])
                    prompt = str(res["prompt"])

                    preds[i] = pred
                    prompt_build_s[i] = float(res["prompt_build_s"])
                    llm_generate_s[i] = float(res["llm_generate_s"])
                    llm_client_wait_s[i] = float(res["llm_client_wait_s"])
                    tau_e2e_s[i] = float(res["finish_rel_s"])
                    prompt_chars[i] = len(prompt)
                    prompt_has_gold[i] = int(_contains_any(prompt, qa.a))
                    correct_contains[i] = int(_contains_any(pred, qa.a))
                except Exception as e:
                    print(f"[WARN] generation failed slot={slot_id} i={i} qid={qa.qid}: {repr(e)}")
                    preds[i] = ""
                    tau_e2e_s[i] = float(time.perf_counter() - slot_t0)
        finally:
            heavy_executor.shutdown(wait=True, cancel_futures=False)
            llm_executor.shutdown(wait=True, cancel_futures=False)

        batch_wall_time_s = float(time.perf_counter() - slot_t0)
        accuracy = float(batch_accuracy(preds, golds, mode=self.acc_mode))

        judge_accuracy = None
        judge_wall_time_s = 0.0
        if judge_enabled and self.judge_llm is not None:
            tj = time.perf_counter()
            judge_hits = 0
            for qa, pred in zip(batch, preds):
                judge_hits += int(self._judge_correct(qa.q, pred, qa.a))
            judge_wall_time_s = float(time.perf_counter() - tj)
            judge_accuracy = float(judge_hits / max(1, B))
            q_value = judge_accuracy
        else:
            q_value = accuracy

        tau_list = [float(x) for x in tau_e2e_s.tolist()]
        d_value = float(np.mean(tau_e2e_s))
        p95_latency_s = float(np.quantile(tau_e2e_s, self.beta))
        psi_value = _cvar_surrogate(tau_list, nu=nu, beta=self.beta, kappa=self.cvar_kappa)
        g_value = self.q_target - float(q_value)

        query_rows: List[Dict[str, Any]] = []
        for i, qa in enumerate(batch):
            query_rows.append({
                "slot": int(slot_id),
                "probe_type": str(probe_type),
                "query_index": int(i),
                "qid": qa.qid,
                "route": "heavy" if bool(route_heavy[i]) else "light",
                "tau_e2e_s": float(tau_e2e_s[i]),
                "light_retrieve_s": float(light_retrieve_s[i]),
                "heavy_retrieve_s": float(heavy_retrieve_s[i]),
                "prompt_build_s": float(prompt_build_s[i]),
                "llm_client_wait_s": float(llm_client_wait_s[i]),
                "llm_generate_s": float(llm_generate_s[i]),
                "n_take_docs": int(n_take_docs[i]),
                "prompt_chars": int(prompt_chars[i]),
                "correct_contains": int(correct_contains[i]),
                "prompt_has_gold": int(prompt_has_gold[i]),
                "heavy_selected": int(route_heavy[i]),
                "sim_sum": float(sim_sums[i]),
            })

        light_mask = ~route_heavy
        heavy_mask_final = route_heavy

        return {
            "slot": int(slot_id),
            "probe_type": str(probe_type),
            "request_count": int(B),
            "d_value": d_value,
            "psi_value": psi_value,
            "g_value": float(g_value),
            "q_value": float(q_value),
            "accuracy": accuracy,
            "judge_accuracy": judge_accuracy,
            "p95_latency_s": p95_latency_s,
            # batch_wall_time_s is measured before judge evaluation.
            # Therefore judge latency is not counted into optimization/runtime latency.
            "batch_wall_time_s": batch_wall_time_s,
            "judge_wall_time_s": judge_wall_time_s,
            "wall_time_with_judge_s": float(batch_wall_time_s + judge_wall_time_s),
            "p_exec": float(p),
            "nu_exec": float(nu),
            "heavy_frac_real": float(np.mean(route_heavy)),
            "avg_docs_light": float(np.mean(n_take_docs[light_mask])) if np.any(light_mask) else 0.0,
            "avg_docs_heavy": float(np.mean(n_take_docs[heavy_mask_final])) if np.any(heavy_mask_final) else 0.0,
            "mean_light_retrieve_s": float(np.mean(light_retrieve_s)),
            "light_batch_wall_time_s": float(light_batch_wall_time_s),
            "mean_heavy_retrieve_s": float(np.mean(heavy_retrieve_s)),
            "mean_prompt_build_s": float(np.mean(prompt_build_s)),
            "mean_llm_client_wait_s": float(np.mean(llm_client_wait_s)),
            "mean_llm_generate_s": float(np.mean(llm_generate_s)),
            "oracle_recall_light_full": float(np.mean(oracle_light_full)),
            "oracle_recall_heavy_full": float(np.mean(oracle_heavy_full[route_heavy])) if np.any(route_heavy) else 0.0,
            "prompt_gold_rate": float(np.mean(prompt_has_gold)),
            "tau_per_request": tau_list,
            "query_rows": query_rows,
        }

    def run_slot(
        self,
        batch: List[QAItem],
        w: np.ndarray,
        return_timings: bool = False,
        verbose: bool = False,
        slot_id_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        slot_id = int(slot_id_override) if slot_id_override is not None else 0
        return self.run_probe(
            batch=batch,
            w=w,
            slot_id=slot_id,
            probe_type="legacy",
            judge_enabled=(self.judge_llm is not None),
            verbose=verbose,
        )


def build_system_from_cfg(cfg: dict, *, judge_enabled: bool) -> AdaRAGSystemOneProbeMP:
    seed = int(cfg.get("seed", 42))
    beta = float(cfg.get("beta", 0.95))

    doc_choices_light = _as_doc_choices(cfg.get("doc_choices_light", None), int(cfg.get("n_docs_light", 4)))
    doc_choices_heavy = _as_doc_choices(cfg.get("doc_choices_heavy", None), int(cfg.get("n_docs_heavy", 6)))
    max_light_docs = max(doc_choices_light)
    max_heavy_docs = max(doc_choices_heavy)

    sys_cfg = cfg.get("system", {}) or {}
    q_target = float((cfg.get("optimizer", {}) or {}).get("q_target", cfg.get("quality_target", 0.60)))

    light_cfg = cfg.get("light_retriever", {}) or {}
    heavy_cfg = cfg.get("heavy_retriever", {}) or {}

    light_topn = max(
        int(light_cfg.get("top_n", 10)),
        max_light_docs,
        max_heavy_docs,
        int(heavy_cfg.get("dense_top_n", 0) or 0),
    )

    print(f"[build_system] light_topn={light_topn}")
    light = build_light_retriever(cfg, topn=light_topn)

    print(f"[build_system] heavy_topn={max_heavy_docs}")
    heavy = build_heavy_retriever(cfg, topn=max_heavy_docs, light=light, llm_rewriter=None)

    try:
        heavy.retrieve("warmup query")
    except Exception as e:
        print(f"[WARN] heavy warmup failed: {repr(e)}")

    print("[build_system] loading generator llm...")
    llm = build_llm(cfg.get("llm", {}) or {})
    print("[build_system] generator llm loaded.")

    judge_llm = None
    if judge_enabled and cfg.get("judge_llm", None):
        print("[build_system] loading judge llm...")
        judge_llm = build_llm(cfg.get("judge_llm", {}) or {})
        print("[build_system] judge llm loaded.")

    return AdaRAGSystemOneProbeMP(
        light_retriever=light,
        heavy_retriever=heavy,
        llm=llm,
        doc_choices_light=doc_choices_light,
        doc_choices_heavy=doc_choices_heavy,
        beta=beta,
        acc_mode=str(cfg.get("acc_mode", "token")),
        prompt_max_doc_chars=int(sys_cfg.get("prompt_max_doc_chars", 1600)),
        seed=seed,
        force_heavy=bool(sys_cfg.get("force_heavy", False)),
        judge_llm=judge_llm,
        exec_mode=str(sys_cfg.get("exec_mode", "overlap")).strip().lower(),
        heavy_max_workers=int(sys_cfg.get("heavy_max_workers", 2)),
        llm_max_workers=int(sys_cfg.get("llm_max_workers", 8)),
        sim_score_topn=int(sys_cfg.get("sim_score_topn", max_light_docs)),
        q_target=q_target,
        cvar_kappa=float(sys_cfg.get("cvar_kappa", 5.0)),
    )


_WORKER_SYSTEM_WITH_JUDGE = None
_WORKER_SYSTEM_NO_JUDGE = None


def run_probe_worker(
    cfg: dict,
    batch_payload: List[Dict[str, Any]],
    w: np.ndarray,
    slot_id: int,
    *,
    judge_enabled: bool,
    verbose: bool = False,
) -> Dict[str, Any]:
    global _WORKER_SYSTEM_WITH_JUDGE, _WORKER_SYSTEM_NO_JUDGE

    if judge_enabled:
        if _WORKER_SYSTEM_WITH_JUDGE is None:
            _WORKER_SYSTEM_WITH_JUDGE = build_system_from_cfg(cfg, judge_enabled=True)
        system = _WORKER_SYSTEM_WITH_JUDGE
    else:
        if _WORKER_SYSTEM_NO_JUDGE is None:
            _WORKER_SYSTEM_NO_JUDGE = build_system_from_cfg(cfg, judge_enabled=False)
        system = _WORKER_SYSTEM_NO_JUDGE

    batch = [_make_qa(x) for x in batch_payload]
    return system.run_probe(
        batch=batch,
        w=np.asarray(w, dtype=float),
        slot_id=int(slot_id),
        probe_type="worker",
        judge_enabled=judge_enabled,
        verbose=verbose,
    )


# Compatibility aliases for old imports.
ParetoAdaRAGSystemOneProbeMP = AdaRAGSystemOneProbeMP
ParetoAdaRAGSystemTriProbeMP = AdaRAGSystemOneProbeMP