# -*- coding: utf-8 -*-
import re
from collections import Counter
from typing import Any, Iterable, List


def _normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(z) for z in x if z is not None]
    return [str(x)]


def exact_match(pred: str, gold: Any) -> float:
    p = _normalize(pred)
    for g in _as_list(gold):
        if p == _normalize(g):
            return 1.0
    return 0.0


def contains_match(pred: str, gold: Any) -> float:
    p = _normalize(pred)
    for g in _as_list(gold):
        gg = _normalize(g)
        if gg and (gg in p):
            return 1.0
    return 0.0


def token_subset_match(pred: str, gold: Any) -> float:
    """
    更“宽松”的 match：gold 的 token 集合 ⊆ pred 的 token 集合 就算对。
    能解决日期顺序、逗号、UTC 等格式差异导致 contains=0 的问题。
    """
    p_tokens = set(_normalize(pred).split())
    if not p_tokens:
        return 0.0
    for g in _as_list(gold):
        g_tokens = set(_normalize(g).split())
        if g_tokens and g_tokens.issubset(p_tokens):
            return 1.0
    return 0.0


def token_f1(pred: str, gold: Any) -> float:
    """
    SQuAD 风格 token-F1（取 gold 候选里最大 F1）
    """
    p = _normalize(pred).split()
    if not p:
        return 0.0
    p_cnt = Counter(p)

    best = 0.0
    for g in _as_list(gold):
        gt = _normalize(g).split()
        if not gt:
            continue
        g_cnt = Counter(gt)
        common = sum((p_cnt & g_cnt).values())
        if common <= 0:
            continue
        precision = common / max(1, sum(p_cnt.values()))
        recall = common / max(1, sum(g_cnt.values()))
        f1 = 2 * precision * recall / max(1e-12, (precision + recall))
        if f1 > best:
            best = f1
    return float(best)


def batch_accuracy(preds: List[str], golds: List[Any], mode: str = "contains") -> float:
    if len(preds) == 0:
        return 0.0
    assert len(preds) == len(golds), "preds/golds length mismatch"

    scores = []
    for p, g in zip(preds, golds):
        if mode == "exact":
            scores.append(exact_match(p, g))
        elif mode == "contains":
            scores.append(contains_match(p, g))
        elif mode == "token":
            scores.append(token_subset_match(p, g))
        elif mode == "f1":
            scores.append(token_f1(p, g))
        else:
            raise ValueError(f"Unknown mode={mode}")
    return float(sum(scores) / max(1, len(scores)))
