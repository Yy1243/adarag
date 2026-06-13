# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple

from adarag.data import QAItem
from adarag.eval.evaluator import batch_accuracy
from adarag.pipeline.prompt_builder import build_prompt

def sample_topk_from_probs(probs: np.ndarray, n_docs: int, rng: np.random.RandomState) -> int:
    """
    probs: x or y, shape >= n_docs, each in [0,1]
    Interpret as:
      P(k=0) = 1 - sum_{i=1..n_docs} probs[i-1]
      P(k=i) = probs[i-1]   for i=1..n_docs
    Then choose k and take top-k docs.
    """
    p = np.clip(probs[:n_docs].astype(float), 0.0, 1.0)
    p0 = 1.0 - float(p.sum())
    if p0 < 0.0:
        # 如果你的 optimizer 产出了 sum(p)>1，这里做一个保险归一化
        # （理论上不该发生，但工程上避免崩）
        p0 = 0.0
    cat = np.concatenate([[p0], p])
    s = float(cat.sum())
    if s <= 0:
        return 0
    cat = cat / s
    k = int(rng.choice(np.arange(0, n_docs + 1), p=cat))
    return k

class AdaRAGSystemCDF:
    """
    Real-ish AdaRAG system:
      - light: dense retriever
      - heavy: bm25 / ES bm25
      - heavy selection: lowest similarity sum (CDF-style), deterministic
      - x,y: per-rank inclusion probs
      - p: fraction of queries using heavy
    """

    def __init__(
        self,
        light_retriever,
        heavy_retriever,
        llm,
        n_docs: int = 5,
        acc_mode: str = "contains",
        seed: int = 42,
        force_heavy: bool = False,   # debug 用：强制每个 query 都走 heavy
        prompt_max_doc_chars: int = 1600,
        judge_llm=None,
    ):
        self.light_retriever = light_retriever
        self.heavy_retriever = heavy_retriever
        self.llm = llm
        self.n_docs = int(n_docs)
        self.acc_mode = acc_mode
        self.force_heavy = bool(force_heavy)
        self.prompt_max_doc_chars = int(prompt_max_doc_chars)
        self.rng = np.random.RandomState(seed)
        self.judge_llm = judge_llm

    def _judge_correct(self, question: str, pred: str, gold: Any) -> bool:
        if self.judge_llm is None:
            return False

        # gold 统一成列表
        gold_list = gold if isinstance(gold, list) else [gold]
        gold_list = [str(x) for x in gold_list if x is not None]

        prompt = (
            "You are a strict QA evaluator.\n"
            "Decide whether the model answer is semantically equivalent to ANY gold answer.\n"
            "Return ONLY 1 or 0.\n\n"
            f"Question: {question}\n"
            f"Gold answers: {gold_list}\n"
            f"Model answer: {pred}\n"
            "Output:"
        )
        out = self.judge_llm.generate(prompt)
        txt = self._llm_to_text(out).strip()
        return txt.startswith("1")

    #分割参数向量Z，返回x,y,p
    @staticmethod
    def split_z(z: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray, float]:
        x = z[:n_docs].clip(0.0, 1.0)
        y = z[n_docs:2 * n_docs].clip(0.0, 1.0)
        p = float(np.clip(z[-1], 0.0, 1.0))
        return x, y, p

    #dubug判断是否heavy选的就是最小那 k 个
    def _select_heavy_indices(self, sim_scores: np.ndarray, p: float) -> np.ndarray:
        B = len(sim_scores)
        if B == 0:
            return np.zeros((0,), dtype=bool)
        if self.force_heavy:
            return np.ones((B,), dtype=bool)    #(B,) 是 元组语法，表示"形状为5的一维数组",结果为结果：[False, False, False, False, False]
        if p <= 0.0:
            return np.zeros((B,), dtype=bool)

        k = int(np.round(p * B))
        k = max(1, min(B, k))
        order = np.argsort(sim_scores)    #获取从低到高的索引排序，临时改成负的做测试
        heavy_mask = np.zeros((B,), dtype=bool)
        heavy_mask[order[:k]] = True     #order[:k] 是 最小的 k 个索引，order是数组对象 
        return heavy_mask       #返回 np.ndarray，元素类型是 bool（布尔值），判断是否走重检索路线

    #转小写、去首尾空格、合并连续空格为标准单空格，用于答案匹配时消除格式差异
    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(str(s).lower().strip().split())
    
    #兼容不同数据源返回的文档格式（字典、对象、字符串），提取其中的文本内容。
    @staticmethod
    def _doc_to_text(d: Any) -> str:
        if d is None:
            return ""
        if isinstance(d, str):
            return d
        if isinstance(d, dict):
            return str(d.get("text") or d.get("contents") or d.get("passage") or d.get("document") or "")
        for attr in ("text", "contents", "passage", "document"):
            if hasattr(d, attr):
                return str(getattr(d, attr) or "")
        return str(d)

    #统一提取出llm生成的文本字符串。
    @staticmethod
    def _llm_to_text(x: Any) -> str:
        """
        兼容：
        - vllm wrapper 返回 str
        - 返回 [str]
        - 返回 vllm RequestOutput / 其列表
        """
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, list) and len(x) > 0:
            # [str]
            if isinstance(x[0], str):
                return x[0]
            # [RequestOutput]
            o = x[0]
            if hasattr(o, "outputs") and o.outputs:
                out0 = o.outputs[0]
                if hasattr(out0, "text"):
                    return str(out0.text or "")
            return str(o)
        # RequestOutput
        if hasattr(x, "outputs") and getattr(x, "outputs", None):
            outs = x.outputs
            if outs and hasattr(outs[0], "text"):
                return str(outs[0].text or "")
        return str(x)

    #处理LLM输出不稳定
    @staticmethod
    def _postprocess_answer(s: str) -> str:
        s = (s or "").strip()  #防止 None
        # 常见：Answer: xxx
        m = re.search(r"(?i)\banswer\s*:\s*(.+)", s)   #(?i)：忽略大小写，\b：单词边界，answer：匹配字面量 "answer，\s*：匹配零个或多个空白字符，:：匹配冒号，(.+)：捕获组，匹配冒号后的所有内容（.+表示一个或多个任意字符）
        if m:
            s = m.group(1).strip()        #提取捕获组也即答案内容并去掉首尾空格
        s = (s or "").strip()
        s = " ".join(s.splitlines()).strip()
        s = s.strip(" \"'`")
        return s

    #匹配判断，判断模型输出是否包含正确答案，比精确匹配更宽松
    @classmethod
    def _contains_any(cls, pred: str, gold: Any) -> bool:   #cls表示类本身，这里是 AdaRAGSystemCDF
        p = cls._norm(pred)     #标准化预测答案，用 cls._norm() 可以调用类的其他静态/类方法，不需要创建实例，普通实例方法用 self（代表实例）
        if isinstance(gold, str):
            g = cls._norm(gold)
            return bool(g) and (g in p)  #判断标准化后的正确答案是否在预测答案中出现，如果 g 是空字符串 ""，bool("") 是 False，直接返回 False
        if isinstance(gold, list):
            for a in gold:
                g = cls._norm(a)
                if g and (g in p):
                    return True
        return False

    #检查检索到的文档中是否本来就有答案
    @classmethod
    def _oracle_hit(cls, docs: List[Any], gold: Any) -> bool:  #和上面项目只是将 pred 换成了 docs，其他逻辑类似
        if not docs:
            return False
        texts = [cls._norm(cls._doc_to_text(d)) for d in docs]  #提取并标准化所有文档文本
        if isinstance(gold, str):    #类比上面的方法基本一致
            g = cls._norm(gold)
            return any(g and (g in t) for t in texts)  
        if isinstance(gold, list):
            for a in gold:
                g = cls._norm(a)
                if any(g and (g in t) for t in texts):
                    return True
        return False
    
    #这是系统的主干逻辑，处理一个批次的查询：
    def run_slot(self, batch: List[QAItem], z: np.ndarray, latency_target_s: float) -> Dict[str, Any]:
        x, y, p = self.split_z(z, self.n_docs)

        preds: List[str] = []
        golds: List[Any] = []

        light_doc_cnt: List[int] = []      # 每个查询用了多少轻文档，是数量
        heavy_doc_cnt: List[int] = []      # 每个查询用了多少重文档

        light_docs_all: List[List[Any]] = []   # 缓存所有轻检索结果，是结果
        sim_sum: List[float] = []             #缓存相似度总和（用于选择heavy）
        examples: List[Dict[str, Any]] = []   #详细案例记录（用于调试）

        oracle_light_hits = 0        #轻文档包含答案的次数，dubug
        oracle_heavy_hits = 0        #重文档包含答案的次数
        oracle_any_hits = 0          #任一路径包含答案的次数
        prompt_has_gold_cnt = 0      #Prompt直接包含答案字符串的次数（信息泄露检查）
        # --- BEFORE sampling: true topN recall of retrievers (full) ---未送进prompt前的检索器召回率检查
        oracle_light_full_hits = 0
        oracle_heavy_full_hits = 0
        oracle_any_full_hits = 0
        oracle_light_full_list: List[bool] = []   # 每条 query light topN 是否命中 gold

        t0_slot = time.time()  #返回从1970年1月1日到现在经过的秒数（浮点数）
        judge_hits = 0      #模型评判器的效果计数
        
        #每个问题都需要经过一轮轻检索    
        for qa in batch:
            docs_l, scores_l = self.light_retriever.retrieve(qa.q)   #这里是没问题的，相当于调用self.light_retriever 这个对象的成员方法，然后self.light_retriever是一个轻检索器实例,所以只要这个实例有相应的方法即可！！！！（python的动态特性） 
            light_docs_all.append(docs_l)        #将每个问题的轻检索结果存起来
            sim_sum.append(float(np.sum(scores_l)) if len(scores_l) else 0.0)    #计算每个问题相似度总和存起来,scores_l是一个numpy数组，可以对他直接利用求和公式np.sum,然后添加到列表中
            hit_l_full = self._oracle_hit(docs_l, qa.a)
            oracle_light_full_hits += int(hit_l_full)
            oracle_light_full_list.append(bool(hit_l_full))

        #根据得分选出重检索的那一批
        sim_sum_arr = np.asarray(sim_sum, dtype=float)  #将列表转换为 numpy 数组给下面进行计算
        heavy_mask = self._select_heavy_indices(sim_sum_arr, p)   #调用方法，传入数组根据相似度总和和比例 p 选择重检索的索引数组形式
        # 调试检查: heavy_mask 是否选择了 sim_sum 最低的那批 ---
        B = len(batch)
        if B > 0:
            if self.force_heavy:
                k_expected = B
            elif p <= 0.0:
                k_expected = 0
            else:
                k_expected = int(np.round(p * B))       # 计算期望的重检索数量
                k_expected = max(1, min(B, k_expected))   # 保证在合理范围内

            order = np.argsort(sim_sum_arr)   # 从低到高的排序索引
            chosen = np.where(heavy_mask)[0]   # np.where返回的是一个元组里面只有一个元素，(heavy_mask)[0]返回true位置的索引
            expected = order[:k_expected] if k_expected > 0 else np.array([], dtype=int)  #期望的重检索索引，相似度最低的索引，返回数组
    
            ok = set(chosen.tolist()) == set(expected.tolist()) #判断实际选择的重检索索引和期望的是否一致，tolist()：NumPy数组转Python列表

            print("\n[SanityCheck] heavy selection")
            print(f"  B={B}, p={p:.4f}, k_expected={k_expected}, chosen_cnt={len(chosen)}, ok={ok}")

            # 打印最小的几个 / 最大的几个 sim_sum，看重检索是否在低端
            m = min(5, B)   
            print("  lowest sim_sum:")
            for idx in order[:m]:
                print(f"    idx={idx:2d}, sim_sum={sim_sum_arr[idx]:.6f}, heavy={bool(heavy_mask[idx])}")
            print("  highest sim_sum:")
            for idx in order[-m:][::-1]:
                print(f"    idx={idx:2d}, sim_sum={sim_sum_arr[idx]:.6f}, heavy={bool(heavy_mask[idx])}")

            # 如果不一致，额外把 chosen 和 expected 打出来
            if not ok:
                print("  chosen idx:", chosen.tolist())
                print("  expected idx:", expected.tolist())

        #对每个问题，根据是否走重检索路径，构建 Prompt 并调用 LLM 生成答案
        for i, qa in enumerate(batch):
            docs_l = light_docs_all[i]    #取出对应问题的轻检索结果
            if (not heavy_mask[i]) or (self.heavy_retriever is None):
                # -------- light path: only light docs --------
                oracle_any_full_hits += int(oracle_light_full_list[i])

                k_l = sample_topk_from_probs(x, self.n_docs, self.rng)  #根据 x 概率向量采样决定取多少轻文档！！！！！！
                k_l = max(1, k_l)
                take_l = docs_l[: min(k_l, len(docs_l))]  #取出对应数量的轻文档
                take_h = []        

                light_doc_cnt.append(len(take_l))    #记录轻文档数量
                heavy_doc_cnt.append(0)            #重文档数量为0

                try:
                    prompt = build_prompt(     #调用prompt_builder下的方法进行prompt构建
                        question=qa.q,
                        docs=take_l,
                        max_doc_chars=self.prompt_max_doc_chars, #限制提示词文档字符数
                    )
                except TypeError:
                    prompt = build_prompt(question=qa.q, docs=take_l)  #兼容旧接口

                pred_raw = self.llm.generate(prompt) 

            else:
                # -------- heavy path: retrieve heavy docs then infer only on heavy docs --------
                docs_h, _scores_h = self.heavy_retriever.retrieve(qa.q)   #得到的文档和分数是由高到低排序的，retrieve方法返回的是一个元组，第一个元素是文档列表，第二个元素是对应的分数列表
                hit_h_full = self._oracle_hit(docs_h, qa.a)
                oracle_heavy_full_hits += int(hit_h_full)
                oracle_any_full_hits += int(oracle_light_full_list[i] or hit_h_full)

                k_h = sample_topk_from_probs(y, self.n_docs, self.rng)
                k_h = max(1, k_h)
                take_h = docs_h[: min(k_h, len(docs_h))]
                take_l = []  

                light_doc_cnt.append(0)
                heavy_doc_cnt.append(len(take_h))

                try:
                    prompt = build_prompt(
                        question=qa.q,
                        docs=take_h,
                        max_doc_chars=self.prompt_max_doc_chars,
                    )
                except TypeError:
                    prompt = build_prompt(question=qa.q, docs=take_h)

                pred_raw = self.llm.generate(prompt)   #实际运行中LLM这里是被实例化了，他实例化的对象来源于hf_llm.py里的HFTextLLM类，然后在HFTextLLM类里有generate方法

            # 提取并清洗答案
            pred_text = self._postprocess_answer(self._llm_to_text(pred_raw))  #类中自己的私有方法
            preds.append(pred_text)      #添加到预测答案列表
            golds.append(qa.a)       #添加到正确答案列表
            
            if self.judge_llm is not None:
                judge_hits += int(self._judge_correct(qa.q, pred_text, qa.a))

            hit_l = self._oracle_hit(take_l, qa.a)   #返回布尔类型的值
            hit_h = self._oracle_hit(take_h, qa.a)
            hit_any = hit_l or hit_h     
            oracle_light_hits += int(hit_l)  #轻检索包含答案的次数累加，整形数字
            oracle_heavy_hits += int(hit_h)  #重检索包含答案的次数累加，整形数字
            oracle_any_hits += int(hit_any)  #任一路径包含答案的次数累加，整形数字
            prompt_has_gold = self._contains_any(prompt, qa.a)  
            prompt_has_gold_cnt += int(prompt_has_gold)  

            examples.append({
                "i": i,
                "question": qa.q,
                "golds": qa.a,
                "pred": pred_text,
                "pred_raw_head": str(pred_raw)[:200],
                "is_correct_contains": self._contains_any(pred_text, qa.a),
                "heavy_selected": bool(heavy_mask[i]),
                "sim_sum": float(sim_sum_arr[i]),
                "n_take_light": int(len(take_l)),
                "n_take_heavy": int(len(take_h)),
                "oracle_hit_any": bool(hit_any),
                "prompt_has_gold": bool(prompt_has_gold),
                "prompt_chars": int(len(prompt)),
                "prompt_head": prompt[:400],
            })

        latency_s = time.time() - t0_slot
        latency_per_q = latency_s / max(1, len(batch))
        acc = batch_accuracy(preds, golds, mode=self.acc_mode)   #调用eval下的batch_accuracy计算准确率

        g = latency_per_q - float(latency_target_s)
        B = max(1, len(batch))

        return {
            "latency_s": float(latency_per_q),
            "accuracy": float(acc),
            "constraint_g": float(g),
            "avg_docs_light": float(np.mean(light_doc_cnt) if light_doc_cnt else 0.0),
            "avg_docs_heavy": float(np.mean(heavy_doc_cnt) if heavy_doc_cnt else 0.0),
            "p": float(p),
            "heavy_frac_real": float(np.mean(heavy_mask) if len(heavy_mask) else 0.0),

            # after-sampling oracle (docs actually sent to prompt)
            "oracle_recall_any": float(oracle_any_hits / B),
            "oracle_recall_light": float(oracle_light_hits / B),
            "oracle_recall_heavy": float(oracle_heavy_hits / B),

            # before-sampling oracle (retriever topN)
            "oracle_recall_any_full": float(oracle_any_full_hits / B), 
            "oracle_recall_light_full": float(oracle_light_full_hits / B),
            "oracle_recall_heavy_full": float(oracle_heavy_full_hits / B),

            "prompt_gold_rate": float(prompt_has_gold_cnt / B),
            "judge_accuracy": float(judge_hits / B) if self.judge_llm is not None else None,

            "examples": examples,
        }
