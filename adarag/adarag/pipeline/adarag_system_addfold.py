#/T20050013/adarag_repro/adarag/pipeline/adarag_system_addfold.py
from __future__ import annotations

import re
import time
import numpy as np
import concurrent.futures as cf
from typing import List, Dict, Any, Tuple, Optional
from adarag.data import QAItem
from adarag.eval.evaluator import batch_accuracy
from adarag.pipeline.prompt_builder import build_prompt


def sample_topk_from_probs(probs: np.ndarray, n_docs: int, rng: np.random.RandomState) -> int:   #传入的参数是一个概率数组，一个整数n_docs表示可选文档数量的上限数量，rng表示一个随机生成数对象，类型注解了比较清晰，实际调入的是my_rng = np.random.RandomState(42) 类似这种创建的实例对象
    p = np.clip(probs[:n_docs].astype(float), 0.0, 1.0)      #切片取其中索引0-n_docs-1的元素转化为浮点数值在0-1之间
    p0 = 1.0 - float(p.sum())       #计算k=0时的概率，p.sum（）返回一个标量
    if p0 < 0.0:
        p0 = 0.0
    cat = np.concatenate([[p0], p])   #拼接数组，生成一个长度为n_docs+1的一维数组，索引0的元素是p0，索引1到n_docs的元素是p中的元素
    s = float(cat.sum())
    if s <= 0:
        return 0
    cat = cat / s   #归一化操作，使得cat数组的元素之和为1，形成一个合法的概率分布
    k = int(rng.choice(np.arange(0, n_docs + 1), p=cat))    #在 rng.choice 中第一个参数第一个参数 a：候选元素的集合，即要从哪些值中进行选择；第二个参数 p：每个候选元素被选中的概率，必须与 a 的长度相同且归一化。
    return k


class AdaRAGSystemCDF:
    def __init__(
        self,
        light_retriever,         #两个检索器对象，实现 retrieve(query) 方法，返回 (docs, scores)
        heavy_retriever,
        llm,        #生成器对象，实现 generate(prompt) 方法，返回生成的文本（或包含文本的结构）
        n_docs: int = 5,        #每个查询最多考虑的文档数量，即top-n
        acc_mode: str = "contains",
        seed: int = 42,
        force_heavy: bool = False,      #如果为 True，则强制所有查询都走 heavy 路径（用于调试或消融实验）
        prompt_max_doc_chars: int = 1600,   #用于控制提示词长度
        judge_llm=None,
        exec_mode: str = "overlap",            # "serial" | "overlap"两种模式
        heavy_max_workers: int = 8,           #重叠模式下的线程数
    ):
        self.light_retriever = light_retriever
        self.heavy_retriever = heavy_retriever
        self.llm = llm
        self.n_docs = int(n_docs)
        self.acc_mode = acc_mode
        self.force_heavy = bool(force_heavy)
        self.prompt_max_doc_chars = int(prompt_max_doc_chars)
        self.rng = np.random.RandomState(seed)      #全局的随机数生成器，用于类级别的随机需求，当前的设定中后续其实并没有用到这个属性，因为线程问题所以我们不能通过共享一个随机生成数对象来确保最后随机的可重复性
        self.seed = int(seed)   #保存种子值，用于派生每个查询的独立随机数生成器
        self._slot_counter = 0  #用于为每个 run_slot 调用分配一个递增的 ID，记录批次。
        self.judge_llm = judge_llm
        exec_mode = str(exec_mode).lower().strip()
        if exec_mode not in ("serial", "overlap"):
            raise ValueError(f"exec_mode must be 'serial' or 'overlap', got: {exec_mode}")
        self.exec_mode = exec_mode
        self.heavy_max_workers = int(max(1, heavy_max_workers))

    @staticmethod
    def _mix32(x: int) -> int:  #为了确保在重叠模式代码下可重复的随机性，代码实现了每个查询独立的随机数生成器。在单线程中，任务 1 先拿随机数，任务 2 后拿，顺序固定。但在多线程中，任务可能同时请求，谁先拿到随机数取决于操作系统调度。因此我们不再让任务共享一个 RNG，而是给每个任务创建自己的 RNG，且这个 RNG 的种子由任务的“身份证”决定。任务 A 的种子 = 函数(全局种子, 任务编号, 其他信息)。在当前代码中每个查询就是一个任务，这样，无论任务执行顺序如何，任务 A 内部使用的随机数序列始终由它自己的 RNG 产生，固定不变。整个实验的结果就可重复了。在这里是确保同一个批次下的问题每个问题抽样应该在每一次实验中是抽的相当数目的top-n，而不会因为线程拥堵等问题每次抽取的不一样
        x &= 0xFFFFFFFF     #将x限制为 32 位无符号整数，按位与操作将 x 的高于 32 的位全部清零，只保留低 32 位，这确保后续运算都在 32 位空间中进行，表示32位全1
        x ^= (x >> 16)      #把 x 右移 16 位。例如 x 原来的高 16 位现在移动到低 16 位的位置。让高 16 位的信息与低 16 位混合。原来高 16 位的位被传播到低 16 位，增加位之间的依赖。
        x = (x * 0x7feb352d) & 0xFFFFFFFF       #将输入信息充分“搅拌”，每一位都开始影响输出的许多位。
        x ^= (x >> 15)          #再次右移 15 位（这次移 15 位，不是 16），然后异或。与之前的移位错开，进一步混合，避免规律性
        x = (x * 0x846ca68b) & 0xFFFFFFFF       #使用两个不同的常数可以增强随机性，防止单一乘法的潜在模式。
        x ^= (x >> 16)  
        return x & 0xFFFFFFFF       #最终确保返回值是 32 位，总的来说这个函数它的任务是把输入信息（比如全局种子、批次 ID、查询索引等）充分“打乱”，生成一个高质量的 32 位种子，用于初始化每个查询独立的随机数生成器。

    def _rng_for(self, slot_id: int, i: int, is_heavy: bool) -> np.random.RandomState:  #传入当前批次的唯一id，该查询在当前批次中的索引，该查询是否走 heavy 路径的布尔值返回的是一个随机数对象实例，专用于该查询
        x = self.seed
        x ^= (slot_id * 0x9E3779B1) & 0xFFFFFFFF    #将 slot_id 乘以这个常数。由于乘数很大，即使 slot_id 很小（如 0、1、2），乘积也会在 32 位空间内均匀分布，避免了线性相关。& 0xFFFFFFFF：截断到 32 位。x ^= ...：将当前 x 与该乘积异或。异或是一种可逆、均匀的混合操作，能把 slot_id 的信息融入种子中。
        x ^= (i * 0x85EBCA6B) & 0xFFFFFFFF  #0x85EBCA6B 是另一个大奇数，作用与上一个常数类似，但专门用于混合查询索引 i。将 i 乘以此常数并截断，然后异或到当前种子。这样每个索引值的信息也被独立地混入。
        if is_heavy:
            x ^= 0x27D4EB2F     #0x27D4EB2F 是一个专门用于 heavy 路径的常数。如果查询是 heavy，就将这个常数异或进去；否则跳过。这样，同一个查询如果是 heavy 路径和 light 路径，最终种子就会不同，从而产生不同的随机数序列，避免路径间的干扰。
        return np.random.RandomState(self._mix32(x))

    def _judge_correct(self, question: str, pred: str, gold: Any) -> bool:
        if self.judge_llm is None:
            return False

        gold_list = gold if isinstance(gold, list) else [gold]      #规范化标准答案，如果是列表则直接用，否则包成只有一个元素的列表
        gold_list = [str(x) for x in gold_list if x is not None]        #然后遍历列表，将每个元素转换为字符串，并过滤掉 None 值。这样 gold_list 就是一个由字符串组成的列表，便于在提示中使用。

        prompt = (
            "You are a strict QA evaluator.\n"
            "Decide whether the model answer is semantically equivalent to ANY gold answer.\n"
            "Return ONLY 1 or 0.\n\n"
            f"Question: {question}\n"
            f"Gold answers: {gold_list}\n"
            f"Model answer: {pred}\n"
        )
        out = self.judge_llm.generate(prompt)   #调用评判模型的 generate 方法，传入提示，得到原始输出。
        out = self._llm_to_text(out).strip()    #它能将LLM返回的各种类型统一转换为字符串，方法在后面进行定义的，编译代码是整体编译的，故即使在这段代码后面编译也是没有任何问题的
        m = re.search(r"[01]", out)     #使用正则表达式 [01] 在输出字符串中搜索第一个出现的 '0' 或 '1' 字符。re.search 返回一个匹配对象，如果没找到则返回 None。
        return (m is not None) and (m.group(0) == "1")  

    @staticmethod
    def split_z(z: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray, float]:    #分割整体的策略参数向量z为三个部分：前 n_docs 个元素对应 light 路径的文档数量参数 x；接下来的 n_docs 个元素对应 heavy 路径的文档数量参数 y；最后一个元素对应选择 heavy 路径的概率 p。函数首先将 z 的前 2*n_docs 个元素切片出来，分别进行截断（clip）操作，确保它们的值在 0.0 到 1.0 之间。然后将 z 的最后一个元素也进行截断，得到 p。最终返回 x、y 和 p 三个值。这个函数的作用是从整体的策略参数 z 中提取出针对 light 和 heavy 路径的具体参数，以及选择 heavy 路径的概率，为后续的决策提供依据。    
        x = z[:n_docs].clip(0.0, 1.0)       #取前 n_docs 个元素，并将它们的值限制在 0.0 到 1.0 之间，得到 light 路径的文档数量参数 x，clip 是 NumPy 中用于将数值限制在指定范围内的函数。它会把所有小于下界的值替换为下界，大于上界的值替换为上界，其余的值保持不变
        y = z[n_docs:2 * n_docs].clip(0.0, 1.0)
        p = float(np.clip(z[-1], 0.0, 1.0))     #z[-1] 是数组的最后一个元素，它是一个标量（例如一个浮点数），不是数组。标量没有 .clip() 方法，所以不能写成 z[-1].clip(0.0, 1.0)（会抛出 AttributeError）。因此必须使用全局函数 np.clip(z[-1], 0.0, 1.0) 来处理标量。两种用法
        return x, y, p      

    def _select_heavy_indices(self, sim_scores: np.ndarray, p: float) -> np.ndarray:    #传入的参数sim_scores是一个数组，长度是当前批次中查询的数量，表示每个查询的相似度分数；p是一个浮点数，表示选择 heavy 路径的概率。函数的目的是根据 sim_scores 中的分数和概率 p 来决定哪些查询应该走 heavy 路径。具体来说，它会选择 sim_scores 中分数最低的 p-fraction 的查询作为 heavy 路径。
        B = len(sim_scores)     #获取批次查询问题数量的大小
        if B == 0:
            return np.zeros((0,), dtype=bool)   #如果是空则直接返回空布尔数组0
        if self.force_heavy:
            return np.ones((B,), dtype=bool)    #如果使用了强制重检索模式，则此时返回全 True 即 1 的掩码。
        if p <= 0.0:
            return np.zeros((B,), dtype=bool)   #如果概率小于0则返回全为0即全部不走重检索
        k = int(np.round(p * B))      #np.round表示对乘积进行一个四舍五入然后再取整数
        order = np.argsort(sim_scores)      # low -> high，np.argsort 返回按值升序排列的索引数组。也就是说，order[0] 对应分数最低的查询（light 检索效果最差）对应的索引，order[-1] 对应分数最高的查询对应的索引。
        heavy_mask = np.zeros((B,), dtype=bool)     #先创建一个全为false的数组，一维数组长度为B，(B, 1)则变成二维数组，B行1列，(1, B)则代表二维数组1行B列
        heavy_mask[order[:k]] = True
        return heavy_mask

    @staticmethod
    def _norm(s: str) -> str:       #文本规范与预处理
        return " ".join(str(s).lower().strip().split())     #先转成字符串，再小写化，再移除字符串开头和结尾的所有空白字符串，再以空白字符为分隔符将字符串拆分成一个单词列表，标点符合仍会在字符串里面不会去除的

    @staticmethod
    def _doc_to_text(d: Any) -> str:
        if d is None:
            return ""
        if isinstance(d, str):
            return d
        if isinstance(d, dict):
            return str(d.get("text") or d.get("contents") or d.get("passage") or d.get("document") or "")
        for attr in ("text", "contents", "passage", "document"):        #处理对象类型，遍历可能出现的属性名
            if hasattr(d, attr):
                return str(getattr(d, attr) or "")      #如果有的话获取该属性的值
        return str(d)

    @staticmethod           #装饰器这里将方法标记为了静态方法。这意味着该方法不依赖于类的实例（不需要self参数），可以直接通过类名调用，如 AdaRAGSystemCDF._llm_to_text(x)。
    def _llm_to_text(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ("text", "output", "answer", "generated_text"):
                if k in x:
                    return str(x[k])
            return str(x)
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return AdaRAGSystemCDF._llm_to_text(x[0])       #取列表元组中的第一个元素，然后递归的调用自身以用于处理嵌套结构，层层剥开
        return str(x)

    @staticmethod
    def _postprocess_answer(s: str) -> str:     #传入的参数为大模型生成的原始回答，作用是将数据变得更干净
        if s is None:
            return ""
        s = str(s)
        s = re.sub(r"```.*?```", "", s, flags=re.S)     #正则表达式替换，r"```.*?```"表示需要匹配的格式，""表示替换为空的也即删除的含义。具体来说```表示匹配开头的三个反引号，.*?：非贪婪匹配任意字符（代码内容），```匹配结尾的三个反引号。flags=re.S：单行模式（DOTALL），使.能匹配换行符，确保多行代码块也能被匹配，如果是.*就代表是贪婪的，?跟在量词后面，将其变为非贪婪（懒惰）模式。
        m = re.search(r"final\s*answer\s*[:：]\s*(.*)$", s, flags=re.I | re.S)      #提取最终答案，其中re.search()表示搜索匹配正则的第一个位置，final表示匹配单词"final"，\s*表示匹配零个或多个空白字符（空格、制表符等），其中\s是特殊转义符，代表任何空白字符，answer表示匹配单词，"answer"，\s*表示再次匹配可选空白，[:：]表示匹配英文冒号:或中文冒号：，\s*表示冒号后的可选空白，(.*)表示捕获组，捕获冒号后的所有内容（即答案本身），()代表捕获组，将括号内的匹配内容保存起来，可以用group(1)提取，.通配符，匹配任意单个字符（除了换行符，除非有re.S标志），*量词，0次或多次，$：匹配字符串结尾（确保拿到最后的答案部分），re.I表示忽略大小写（匹配"Final Answer"、"FINAL ANSWER"等），re.S表示单行模式（使.匹配换行符，确保多行答案也能捕获）。总的来说就是匹配final answer冒号后面的内容
        if m:
            s = m.group(1)  #如果匹配成功（m不为None），提取第1个捕获组（即(.*)匹配到的内容），赋值给s。
        s = s.strip()   #去除首尾空白，删除字符串开头和结尾的空格、换行、制表符等。
        s = " ".join(s.split())     #s.split()表示按任意空白字符（空格、换行、制表符等）分割，返回单词列表，自动去除多余空白，" ".join(...)表示用单个空格将单词重新连接
        return s

    @staticmethod
    def _contains_any(text: str, gold: Any) -> bool:    #text是模型生成的文本，其实在后面永不上，因为我们用的是模型判断正误
        if text is None:
            return False
        t = AdaRAGSystemCDF._norm(text)     #调用静态方法标准化一下
        if isinstance(gold, list):
            for g in gold:
                if g is None:
                    continue
                if AdaRAGSystemCDF._norm(g) in t:   #子串即判断对
                    return True
            return False
        if gold is None:
            return False
        return AdaRAGSystemCDF._norm(gold) in t     #如果是单个答案的话则直接判断

    def _oracle_hit(self, docs: List[Any], gold: Any) -> bool:      #检查检索到的文档是否包含正确答案，测试召回
        if docs is None:
            return False
        for d in docs:
            if self._contains_any(self._doc_to_text(d), gold):      #对于静态方法两种调用都可以一种是类方法调用，一种是实例调用
                return True
        return False

    def run_slot(
        self,
        batch: List[QAItem],    #批处理的回答对列表
        z: np.ndarray,          #控制参数向量 [x, y, p]
        latency_target_s: float,
        *,
        return_timings: bool = False,   # 是否返回详细耗时统计（默认关闭）
        verbose: bool = False,       # 是否打印调试信息（默认关闭）
    ) -> Dict[str, Any]:
        slot_id = self._slot_counter    #每一个run_slot 调用分配唯一递增的 slot_id，确保可复现以及多线程环境下避免种子竞争
        self._slot_counter += 1     #根据执行模式调用不同的内部方法来处理批次。serial 模式下，所有查询按顺序处理；overlap 模式下，重叠执行重检索和轻路径处理以节省时间。
        if self.exec_mode == "overlap":
            return self._run_slot_overlap(batch, z, latency_target_s, slot_id=slot_id, return_timings=return_timings, verbose=verbose)
        return self._run_slot_serial(batch, z, latency_target_s, slot_id=slot_id, return_timings=return_timings, verbose=verbose)

    def _run_slot_serial(
        self,
        batch: List[QAItem],
        z: np.ndarray,
        latency_target_s: float,    #目标延迟约束可以忽略
        slot_id: int,
        *,
        return_timings: bool,       # 是否返回详细耗时统计（默认关闭）
        verbose: bool,          # 是否打印调试信息（默认关闭），* 强制之后的参数必须使用关键字调用，提高代码可读性和安全性。
    ) -> Dict[str, Any]:
        x, y, p = self.split_z(z, self.n_docs)      #调用类里面的方法
        B = len(batch)      #获取批次中查询的数量，如果是0则直接返回一个包含所有指标为0或None的字典，避免后续处理中的除零错误或索引错误。
        if B == 0:
            return {
                "latency_s": 0.0,
                "accuracy": 0.0,
                "constraint_g": -float(latency_target_s),
                "avg_docs_light": 0.0,
                "avg_docs_heavy": 0.0,
                "p": float(p),
                "heavy_frac_real": 0.0,
                "oracle_recall_any": 0.0,
                "oracle_recall_light": 0.0,
                "oracle_recall_heavy": 0.0,
                "oracle_recall_any_full": 0.0,
                "oracle_recall_light_full": 0.0,
                "oracle_recall_heavy_full": 0.0,
                "prompt_gold_rate": 0.0,
                "judge_accuracy": None,
                "examples": [],
            }

        preds: List[str] = []
        golds: List[Any] = []
        light_doc_cnt: List[int] = []   #每个查询使用的轻量文档数量，记录批次处理过程中每个查询的轻量检索结果中实际使用了多少文档
        heavy_doc_cnt: List[int] = []   
        light_docs_all: List[List[Any]] = []    #所有查询的轻量检索结果，文档列表的列表，和下面一个参数的配合使用
        sim_sum: List[float] = []       #每个查询的轻量检索相似度分数之和，用于后续的 heavy 路径选择
        examples: List[Dict[str, Any]] = []     # 记录每个查询的完整处理记录，包括问题、答案、检索结果、路径选择等信息，便于分析和调试
        oracle_light_hits = oracle_heavy_hits = oracle_any_hits = 0     #实际送入LLM的文档中是否命中，记录数量
        prompt_has_gold_cnt = 0         #记录数量，用于计算提示词中包含正确答案的比例
        oracle_light_full_hits = oracle_heavy_full_hits = oracle_any_full_hits = 0      #检索器返回的全部文档中是否命中，不是实际送入，记录数量	
        oracle_light_full_list: List[bool] = []     #每个查询的轻检索结果中是否包含正确答案的布尔值列表，长度与批次查询数量相同，元素为 True 或 False，表示对应查询的轻检索结果是否包含正确答案，记录具体批次数据
        judge_hits = 0
        #耗时统计可不选
        t_light_retrieve_s: Optional[List[float]] = [] if return_timings else None
        t_heavy_retrieve_s: Optional[List[float]] = [] if return_timings else None
        t_prompt_build_s: Optional[List[float]] = [] if return_timings else None
        t_llm_generate_s: Optional[List[float]] = [] if return_timings else None
        path_is_heavy: Optional[List[bool]] = [] if return_timings else None
        t0_slot = time.time()       #槽位开始计时  

        # 1) light retrieval for all
        for qa in batch:
            t0 = time.time()
            docs_l, scores_l = self.light_retriever.retrieve(qa.q)      #调用轻检索器里面的方法
            if return_timings:
                t_light_retrieve_s.append(time.time() - t0)      #记录轻检索耗时，如果设置了return_timings，则在对应的列表中添加耗时数据
            light_docs_all.append(docs_l)       #追加至列表后方，形成一个文档列表的列表，每个元素对应一个查询的轻检索结果
            sim_sum.append(float(np.sum(scores_l)) if len(scores_l) else 0.0)       #记录查询的轻检索相似度并将其追加至列表最后
            hit_l_full = self._oracle_hit(docs_l, qa.a)     #调用类里面的方法检查轻检索结果中是否包含正确答案，返回布尔值
            oracle_light_full_hits += int(hit_l_full)       #如果轻检索结果中包含正确答案，则 oracle_light_full_hits 计数器加 1。int(hit_l_full) 将布尔值转换为整数，True 转换为 1，False 转换为 0。
            oracle_light_full_list.append(bool(hit_l_full))     #将查询的轻检索结果是否包含正确答案的布尔值追加至列表最后，形成一个布尔列表，每个元素对应一个查询的轻检索结果是否包含正确答案

        # 2) heavy selection
        sim_sum_arr = np.asarray(sim_sum, dtype=float)   #转化为数组，数组里每个元素是一个查询的轻检索相似度分数之和。
        heavy_mask = self._select_heavy_indices(sim_sum_arr, p)     #调用类里面方法返回一个数组，长度为批处理的数量，值为true和false。
        if verbose:     #判断是否打印调试信息
            chosen = np.where(heavy_mask)[0]
            print(f"[AdaRAG][serial] B={B} p={p:.4f} heavy_cnt={len(chosen)}")

        # 3) infer each query
        for i, qa in enumerate(batch):
            docs_l = light_docs_all[i]      #拿到当前查询的轻检索结果，文档列表
            is_heavy = bool(heavy_mask[i]) and (self.heavy_retriever is not None)     #根据之前的 heavy_mask 判断当前查询是否走 heavy 路径，且只有当 heavy_retriever 不为 None 时才允许走 heavy 路径
            take_l: List[Any] = []
            take_h: List[Any] = []
            prompt = ""
            if not is_heavy:    #轻检索的生成
                oracle_any_full_hits += int(oracle_light_full_list[i])      #更新数据，表示查询器中返回的是否命中了正确文档，不走重检索那么直接就看轻检索当前检索出的文档便可以了
                k_l = sample_topk_from_probs(x, self.n_docs, self._rng_for(slot_id, i, is_heavy=False))     #采样决定取在n_docs上取出多少个文档
                take_l = docs_l[: min(k_l, len(docs_l))]
                light_doc_cnt.append(len(take_l))   #每个查询使用轻检索文档的数量，列表存储
                heavy_doc_cnt.append(0)     #同理
                if return_timings:      #如果统计了耗时，则记录重检索时间为0
                    t_heavy_retrieve_s.append(0.0)  
                    path_is_heavy.append(False)

                t1 = time.time()
                try:        #构建prompt兼容新旧接口
                    prompt = build_prompt(question=qa.q, docs=take_l, max_doc_chars=self.prompt_max_doc_chars)
                except TypeError:
                    prompt = build_prompt(question=qa.q, docs=take_l)
                if return_timings:
                    t_prompt_build_s.append(time.time() - t1)   #记录t1主要是为了记录prompt的构建时长

                t2 = time.time()
                pred_raw = self.llm.generate(prompt)    #LLM类下面的方法，在初始化的时候已经有定义
                if return_timings:
                    t_llm_generate_s.append(time.time() - t2)   #记录t2主要是为了记录LLM生成回答的时长

            else:       #重检索的生成
                tH0 = time.time()
                docs_h, _scores_h = self.heavy_retriever.retrieve(qa.q)        
                tH = time.time() - tH0      #记录重检索的耗时
                if return_timings:
                    t_heavy_retrieve_s.append(tH)   
                    path_is_heavy.append(True)

                hit_h_full = self._oracle_hit(docs_h, qa.a)       #调用判断一下重检索的语料是否在答案中，返回布尔值
                oracle_heavy_full_hits += int(hit_h_full)       #将其加入到重检索器返回的结果是否在答案中
                oracle_any_full_hits += int(oracle_light_full_list[i] or hit_h_full)    #加入到任意中

                k_h = sample_topk_from_probs(y, self.n_docs, self._rng_for(slot_id, i, is_heavy=True))
                take_h = docs_h[: min(k_h, len(docs_h))]
                light_doc_cnt.append(0)
                heavy_doc_cnt.append(len(take_h))

                t1 = time.time()
                try:
                    prompt = build_prompt(question=qa.q, docs=take_h, max_doc_chars=self.prompt_max_doc_chars)
                except TypeError:
                    prompt = build_prompt(question=qa.q, docs=take_h)
                if return_timings:
                    t_prompt_build_s.append(time.time() - t1)   #提示词构建时间

                t2 = time.time()
                pred_raw = self.llm.generate(prompt)
                if return_timings:
                    t_llm_generate_s.append(time.time() - t2)   #生成最后答案时间

            pred_text = self._postprocess_answer(self._llm_to_text(pred_raw))   #对LLM生成的原始回答进行文本化和后处理，得到最终的预测文本
            preds.append(pred_text)     #将答案添加至preds列表当中
            golds.append(qa.a)          #将问题的答案添加至golds列表里面

            if self.judge_llm is not None:      
                judge_hits += int(self._judge_correct(qa.q, pred_text, qa.a))   #调用类里面定义的方法

            hit_l = self._oracle_hit(take_l, qa.a)      #判断实际送进去的轻检索文档有没有和最后的答案击中
            hit_h = self._oracle_hit(take_h, qa.a)
            hit_any = hit_l or hit_h
            oracle_light_hits += int(hit_l)     #中的话数量加1
            oracle_heavy_hits += int(hit_h)
            oracle_any_hits += int(hit_any)
            prompt_has_gold = self._contains_any(prompt, qa.a)      #判断提示词里是否有答案
            prompt_has_gold_cnt += int(prompt_has_gold)         #计数提示词里有答案的问题数量

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

        slot_total_s = time.time() - t0_slot        #统计该批次整体的处理时间   
        latency_per_q = slot_total_s / max(1, B)        #计算每个问题平均时间
        acc = batch_accuracy(preds, golds, mode=self.acc_mode)  #计算批次的准确率，调用之前定义的方法，传入预测和答案列表以及评测模式
        g = latency_per_q - float(latency_target_s)

        out: Dict[str, Any] = {         #输出汇总，字典类型，为返回值做准备
            "latency_s": float(latency_per_q),
            "accuracy": float(acc),
            "constraint_g": float(g),
            "avg_docs_light": float(np.mean(light_doc_cnt) if light_doc_cnt else 0.0),
            "avg_docs_heavy": float(np.mean(heavy_doc_cnt) if heavy_doc_cnt else 0.0),
            "p": float(p),
            "heavy_frac_real": float(np.mean(heavy_mask) if len(heavy_mask) else 0.0),
            "oracle_recall_any": float(oracle_any_hits / B),
            "oracle_recall_light": float(oracle_light_hits / B),
            "oracle_recall_heavy": float(oracle_heavy_hits / B),
            "oracle_recall_any_full": float(oracle_any_full_hits / B),
            "oracle_recall_light_full": float(oracle_light_full_hits / B),
            "oracle_recall_heavy_full": float(oracle_heavy_full_hits / B),
            "prompt_gold_rate": float(prompt_has_gold_cnt / B),
            "judge_accuracy": float(judge_hits / B) if self.judge_llm is not None else None,    #利用大模型判断正确得到的准确率大小
            "examples": examples,
        }

        if return_timings:      #如果这个设置了还要返回具体的时间的话，那么输出里面额外需要加属性 
            out["timings"] = {
                "light_retrieve_s": t_light_retrieve_s,
                "heavy_retrieve_s": t_heavy_retrieve_s,
                "prompt_build_s": t_prompt_build_s,
                "llm_generate_s": t_llm_generate_s,
                "path_is_heavy": path_is_heavy,
                "slot_total_s": float(slot_total_s),
            }
            out["timing_summary"] = _summarize_timings(out["timings"])      #将原始的时间列表进行汇总统计，得到平均值、中位数、分位数等指标，方便分析和比较。summarize_timings（）是一个函数

        return out

    def _run_slot_overlap(
        self,
        batch: List[QAItem],
        z: np.ndarray,
        latency_target_s: float,
        slot_id: int,
        *,
        return_timings: bool,
        verbose: bool,
    ) -> Dict[str, Any]:
        x, y, p = self.split_z(z, self.n_docs)
        B = len(batch)
        if B == 0:
            return {
                "latency_s": 0.0,
                "accuracy": 0.0,
                "constraint_g": -float(latency_target_s),
                "avg_docs_light": 0.0,
                "avg_docs_heavy": 0.0,
                "p": float(p),
                "heavy_frac_real": 0.0,
                "oracle_recall_any": 0.0,
                "oracle_recall_light": 0.0,
                "oracle_recall_heavy": 0.0,
                "oracle_recall_any_full": 0.0,
                "oracle_recall_light_full": 0.0,
                "oracle_recall_heavy_full": 0.0,
                "prompt_gold_rate": 0.0,
                "judge_accuracy": None,
                "examples": [],
            }
        #结果按索引进行存储，为什么用 by_i 结构？因为Heavy 检索是异步并行的，完成顺序不确定，要按原始索引 i 存储，最后统一汇总
        pred_by_i: List[Optional[str]] = [None] * B     #预测结果，预分配 B 个空槽位，List[Optional[str]] 标注"字符串或 None 的列表"，合起来表示"长度为 B 的结果容器，初始为空，待填入字符串"。
        gold_by_i: List[Any] = [None] * B       #准确结果
        raw_by_i: List[Any] = [None] * B    #原始llm的输出
        take_l_by_i: List[List[Any]] = [[] for _ in range(B)]   #每个查询实际使用的轻检索文档列表，列表的列表，长度为批次查询数量，每个元素是一个列表，包含该查询实际使用的轻检索文档
        take_h_by_i: List[List[Any]] = [[] for _ in range(B)]
        prompt_by_i: List[str] = ["" for _ in range(B)]     #每个查询构建的提示词，列表长度为批次查询数量，每个元素是对应查询的提示词字符串
        light_doc_cnt: List[int] = []
        heavy_doc_cnt: List[int] = []
        examples: List[Dict[str, Any]] = []
        #召回的数据的初始化，与第一种模式基本一致
        oracle_light_hits = oracle_heavy_hits = oracle_any_hits = 0
        prompt_has_gold_cnt = 0
        oracle_light_full_hits = oracle_heavy_full_hits = oracle_any_full_hits = 0
        oracle_light_full_list: List[bool] = []
        judge_hits = 0
        # timings (optional)
        t_light_retrieve_s: Optional[List[float]] = [0.0] * B if return_timings else None
        t_heavy_retrieve_s: Optional[List[float]] = [0.0] * B if return_timings else None
        t_prompt_build_s: Optional[List[float]] = [0.0] * B if return_timings else None
        t_llm_generate_s: Optional[List[float]] = [0.0] * B if return_timings else None
        path_is_heavy: Optional[List[bool]] = [False] * B if return_timings else None

        t0_slot = time.time()
        # 1) light retrieval for all queries，对所有问题先轻检索
        light_docs_all: List[List[Any]] = []
        sim_sum: List[float] = []
        for i, qa in enumerate(batch):
            t0 = time.time()
            docs_l, scores_l = self.light_retriever.retrieve(qa.q)      #docs_l是轻检索返回得到的一个文档列表
            if return_timings:
                t_light_retrieve_s[i] = time.time() - t0
            light_docs_all.append(docs_l)       #将列表追加到列表里面形成嵌套列表形式
            sim_sum.append(float(np.sum(scores_l)) if len(scores_l) else 0.0)       #将列表中的分数求和得到一个总分数，如果没有分数则为0.0，并追加到 sim_sum 列表中
            hit_l_full = self._oracle_hit(docs_l, qa.a)     #调用类里面的方法检查轻检索结果中是否包含正确答案，返回布尔值
            oracle_light_full_hits += int(hit_l_full)       #如果轻检索结果中包含正确答案，则 oracle_light_full_hits 计数器加 1。int(hit_l_full) 将布尔值转换为整数，True 转换为 1，False 转换为 0。
            oracle_light_full_list.append(bool(hit_l_full))     #将查询的轻检索结果是否包含正确答案的布尔值追加至列表最后，形成一个布尔列表，每个元素对应一个查询的轻检索结果是否包含正确答案，记录具体批次数据
        sim_sum_arr = np.asarray(sim_sum, dtype=float)      #将python列表转化为numpy数组
        heavy_mask = self._select_heavy_indices(sim_sum_arr, p)     #将需要进行重检索的标记为true，前面的几个
        if verbose:
            chosen = np.where(heavy_mask)[0]        #np.where(heavy_mask)找出数组中 True 元素的索引位置，返回的是一个元组，然后heavy_mask是一维数组，则返回的是(array([0, 2]),)类似这种，取【0】即代表取全部索引，如果heavy_mask是二维数组，则返回的是(array([0, 1]), array([0, 1])),即第一个元素是行索引数组，第二个元素为列索引数组。np.where(condition)的意义是返回 condition 中 True 元素的索引，所以在这里是返回数组中为true的索引
            print(f"[AdaRAG][overlap] B={B} p={p:.4f} heavy_cnt={len(chosen)} workers={self.heavy_max_workers}")

        # 2) schedule heavy retrievals in background，划分轻重检索并设置异步对象
        futures: Dict[cf.Future, int] = {}      #cf是concurrent.futures的别名，Future对象表示一个异步执行的操作，可能还没有完成。这里用一个字典来存储Future对象和对应的查询索引，方便后续获取结果时知道是哪个查询的重检索结果。因为异步任务完成顺序 ≠ 提交顺序，所以需要记录索引。
        def _heavy_retrieve_timed(q: str) -> Tuple[List[Any], Any, float]:  #定义重检索封装返回时间
            t0 = time.time()
            docs_h, scores_h = self.heavy_retriever.retrieve(q)   
            return docs_h, scores_h, (time.time() - t0)

        executor: Optional[cf.ThreadPoolExecutor] = None    #初始化执行器对象将其设置为none，后续如果需要执行重检索才会实例化这个执行器对象
        if self.heavy_retriever is not None:
            executor = cf.ThreadPoolExecutor(max_workers=self.heavy_max_workers)    #cf.ThreadPoolExecutor为线程池执行器类，max_workers=self.heavy_max_workers代表最多同时运行 N 个线程
            for i, qa in enumerate(batch):
                if bool(heavy_mask[i]):
                    futures[executor.submit(_heavy_retrieve_timed, qa.q)] = i    #将请求提交到线性池里面，executor.submit的功能是返回一个未来对象，则返回值变成了futures[Future对象] = i表示把这个 Future 和查询索引 i 关联起来，方便后续找回。ThreadPoolExecutor下面的submit参数大概是这样的def submit(self, fn, /, *args, **kwargs)，fn: 要执行的函数，*args: 传给 fn 的位置参数， **kwargs: 传给 fn 的关键字参数。相当于字典的赋值意思。
                    if return_timings:
                        path_is_heavy[i] = True

        #定义通过文档进行推理构建提示词加生成最终答案的函数
        def _infer_with_docs(i: int, qa: QAItem, docs: List[Any]) -> Any:
            t1 = time.time()
            try:
                prompt = build_prompt(question=qa.q, docs=docs, max_doc_chars=self.prompt_max_doc_chars)
            except TypeError:
                prompt = build_prompt(question=qa.q, docs=docs)
            if return_timings:
                t_prompt_build_s[i] = time.time() - t1      #构建提示词的时间
            prompt_by_i[i] = prompt
            t2 = time.time()
            pred_raw = self.llm.generate(prompt)
            if return_timings:
                t_llm_generate_s[i] = time.time() - t2
            raw_by_i[i] = pred_raw
            pred_text = self._postprocess_answer(self._llm_to_text(pred_raw))
            pred_by_i[i] = pred_text
            gold_by_i[i] = qa.a
            return pred_text

        #轻检索搞完推理先行，重检索后到达的结果可能会更快，所以先把轻检索的结果进行推理生成，重检索的结果到达后再进行推理生成，这样可以节省时间。
        light_indices = [i for i in range(B) if (not bool(heavy_mask[i])) or (self.heavy_retriever is None)]
        for i in light_indices:
            qa = batch[i]
            oracle_any_full_hits += int(oracle_light_full_list[i])      #对于轻路径来说全量召回就等于实际召回了，所以直接看之前记录的轻检索结果是否包含正确答案的布尔值即可
            docs_l = light_docs_all[i]
            k_l = sample_topk_from_probs(x, self.n_docs, self._rng_for(slot_id, i, is_heavy=False))
            take_l = docs_l[: min(k_l, len(docs_l))]
            take_l_by_i[i] = take_l
            if return_timings:
                t_heavy_retrieve_s[i] = 0.0
                path_is_heavy[i] = False    #前面的是设为ture，这里设置为false
            _infer_with_docs(i, qa, take_l)

        if futures:         #检查是否有heavy任务
            for fut in cf.as_completed(list(futures.keys())):       #futures.keys()代表所有 Future 对象（提交时的凭证），cf.as_completed(...)代表按完成顺序返回 Future，谁先完成谁先出
                i = futures.pop(fut)        #查字典，找这个 Future 对应哪个查询索引
                qa = batch[i]       #找回到原始数据
                try:
                    docs_h, _scores_h, tH= fut.result()        #阻塞获取异步任务返回值,这里返回的是executor.submit(_heavy_retrieve_timed, qa.q, light_docs_all[i])中_heavy_retrieve_timed函数的值，executor.submit相当于把函数和相应的参数交给线程池里的线程去执行，fut.result()就是获取这个函数的返回值，如果这个函数还没有执行完，那么就会阻塞等待直到这个函数执行完毕并返回结果。这里的结果是一个元组，包含重检索得到的文档列表、分数、耗时和使用的bm25查询串。
                except Exception as e:
                    print(f"[AdaRAG][overlap] heavy retrieve failed i={i} qid={getattr(qa,'qid','')}: {repr(e)}")
                    docs_h, _scores_h, tH = [], [], 0.0     #失败的话给空结果

                if return_timings:
                    t_heavy_retrieve_s[i] = float(tH)       #重检索耗时

                hit_h_full = self._oracle_hit(docs_h, qa.a)
                oracle_heavy_full_hits += int(hit_h_full)
                oracle_any_full_hits += int(oracle_light_full_list[i] or hit_h_full)

                k_h = sample_topk_from_probs(y, self.n_docs, self._rng_for(slot_id, i, is_heavy=True))
                take_h = docs_h[: min(k_h, len(docs_h))]
                take_h_by_i[i] = take_h

                # key change: infer immediately when a heavy result is ready (GPU is free here)
                _infer_with_docs(i, qa, take_h)

        if executor is not None:
            executor.shutdown(wait=True)

        # 5) finalize metrics in index order
        preds: List[str] = []
        golds: List[Any] = []

        for i, qa in enumerate(batch):
            pred_text = pred_by_i[i] if pred_by_i[i] is not None else ""
            pred_raw = raw_by_i[i]
            gold = gold_by_i[i]
            preds.append(pred_text)
            golds.append(gold)

            is_heavy = bool(heavy_mask[i]) and (self.heavy_retriever is not None)
            take_l = take_l_by_i[i]
            take_h = take_h_by_i[i]
            prompt = prompt_by_i[i]

            if is_heavy:
                light_doc_cnt.append(0)
                heavy_doc_cnt.append(len(take_h))
            else:
                light_doc_cnt.append(len(take_l))
                heavy_doc_cnt.append(0)

            if self.judge_llm is not None:
                judge_hits += int(self._judge_correct(qa.q, pred_text, qa.a))

            hit_l = self._oracle_hit(take_l, qa.a)
            hit_h = self._oracle_hit(take_h, qa.a)
            hit_any = hit_l or hit_h
            oracle_light_hits += int(hit_l)
            oracle_heavy_hits += int(hit_h)
            oracle_any_hits += int(hit_any)
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

        slot_total_s = time.time() - t0_slot
        latency_per_q = slot_total_s / max(1, B)
        acc = batch_accuracy(preds, golds, mode=self.acc_mode)
        g = latency_per_q - float(latency_target_s)

        out: Dict[str, Any] = {
            "latency_s": float(latency_per_q),
            "accuracy": float(acc),
            "constraint_g": float(g),
            "avg_docs_light": float(np.mean(light_doc_cnt) if light_doc_cnt else 0.0),
            "avg_docs_heavy": float(np.mean(heavy_doc_cnt) if heavy_doc_cnt else 0.0),
            "p": float(p),
            "heavy_frac_real": float(np.mean(heavy_mask) if len(heavy_mask) else 0.0),
            "oracle_recall_any": float(oracle_any_hits / B),
            "oracle_recall_light": float(oracle_light_hits / B),
            "oracle_recall_heavy": float(oracle_heavy_hits / B),
            "oracle_recall_any_full": float(oracle_any_full_hits / B),
            "oracle_recall_light_full": float(oracle_light_full_hits / B),
            "oracle_recall_heavy_full": float(oracle_heavy_full_hits / B),
            "prompt_gold_rate": float(prompt_has_gold_cnt / B),
            "judge_accuracy": float(judge_hits / B) if self.judge_llm is not None else None,
            "examples": examples,
        }

        if return_timings:
            out["timings"] = {
                "light_retrieve_s": t_light_retrieve_s,
                "heavy_retrieve_s": t_heavy_retrieve_s,
                "prompt_build_s": t_prompt_build_s,
                "llm_generate_s": t_llm_generate_s,
                "path_is_heavy": path_is_heavy,
                "slot_total_s": float(slot_total_s),
            }
            out["timing_summary"] = _summarize_timings(out["timings"])

        return out


def _summarize_timings(t: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact timing summary to keep logs clean."""
    out: Dict[str, Any] = {}

    def _stats(arr: List[float]) -> Dict[str, float]:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
        return {
            "mean": float(a.mean()),
            "p50": float(np.quantile(a, 0.50)),
            "p90": float(np.quantile(a, 0.90)),
        }

    for k in ("light_retrieve_s", "heavy_retrieve_s", "prompt_build_s", "llm_generate_s"):
        if k in t and isinstance(t[k], list):
            out[k] = _stats(t[k])
    if "path_is_heavy" in t and isinstance(t["path_is_heavy"], list):
        out["heavy_frac_real"] = float(np.mean(np.asarray(t["path_is_heavy"], dtype=float)))
    if "slot_total_s" in t:
        out["slot_total_s"] = float(t["slot_total_s"])
    return out
