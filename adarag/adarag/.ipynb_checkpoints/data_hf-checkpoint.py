from __future__ import annotations

from typing import List, Optional
import os
import glob
import json

import numpy as np
from datasets import load_dataset
from adarag.data import QAItem


def _find_split_parquets(local_path: str, split: str) -> List[str]:     #在指定目录下递归查找某个 split 对应的 parquet 文件，传入制定目录和相应的split集合，传出目录列表
    # 优先找更“像 split shard”的命名（如果存在）
    cand = sorted(glob.glob(os.path.join(local_path, "**", f"{split}-*.parquet"), recursive=True))  #os.path.join()是os板块中的函数，用于拼接路径，这里是将local_path、"**"和f"{split}-*.parquet"拼接成一个完整的路径模式，比如说os.path.join("/data", "nq-open", "validation-0001.parquet")得到的结果就变成了/data/nq-open/validation-0001.parquet，再比如os.path.join("C:\\data", "nq-open", "validation-0001.parquet")得到的结果就变成了C:\\data\\nq-open\\validation-0001.parquet。glob.glob则是用于文件模式匹配，前面的glob是标准库，后面的glob.glob()函数用于根据指定的模式查找文件，这里是根据上面拼接的路径模式来查找符合条件的parquet文件，**表示递归查找所有子目录，f"{split}-*.parquet"表示文件名以split开头，后面跟着任意字符，最后以.parquet结尾的文件；recursive=True参数表示启用递归查找。sorted()函数则是对找到的文件路径列表进行排序，按照字母顺序排列。
    if not cand:        # 放宽条件（1）：只要文件名里包含 split 就行（兼容一些不规范命名）
        cand = sorted(glob.glob(os.path.join(local_path, "**", f"*{split}*.parquet"), recursive=True))  #os可以理解为一个特殊的板块（不要把他立即为单个模块），他是用c语言编写的，然后os.path相当于是调用板块中的py文件

    if not cand:        #再次放宽条件（2）：只要文件名里面有parquet就可以了
        all_pq = sorted(glob.glob(os.path.join(local_path, "**", "*.parquet"), recursive=True))     #用sorted按照字母排序保证了可复现性
        msg = "\n".join(all_pq[:50])    #只取前50个文件路径不多取，不同文件路径之间用\n换行符进行连接，glob.glob（）返回的是一个列表形式
        raise FileNotFoundError(
            f"Cannot find parquet for split='{split}' under: {local_path}\n"
            f"First 50 parquet files:\n{msg}"
        )
    return cand


def _load_json_or_jsonl(path: str, max_examples: int, seed: int) -> List[QAItem]:    #jsonl和json文件的区别：jsonl是每行一个json对象，整个文件是文本格式；json文件通常是一个列表里面可能会嵌入字典形式的数据。这个函数根据文件后缀来判断是jsonl还是json，然后分别处理。对于jsonl文件，它逐行读取，每行解析成一个json对象；对于json文件，它一次性读取整个文件，解析成一个json对象（通常是列表）。无论哪种情况，最终都会得到一个QAItem的列表。这个函数还支持对数据进行随机打乱和截断，以控制加载的数据量。
    """
    支持 jsonl（每行一个 dict）或 json（整体 list[dict]）。
    字段兼容：question/q, answer/a
    """
    items: List[QAItem] = []

    if path.endswith(".jsonl"):             #str.endswith()是字符串方法，用于判断字符串是否以指定的后缀结尾，这里是判断文件路径是否以.jsonl结尾
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)      #json.loads()是json模块中的函数，用于将json格式的字符串解析成Python对象，这里是将每行文本解析成一个json对象（通常是字典），json.loads()解析字符串
                q = obj.get("question") or obj.get("q") or ""
                a = obj.get("answer") or obj.get("a") or []
                answers = [a] if isinstance(a, str) else (list(a) if a is not None else [])
                items.append(QAItem(q=q, a=answers))

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)     #json.load()是json模块中的函数，用于将json格式的文件解析成Python对象，这里是将整个json文件解析成一个Python对象（通常是列表或者字典），json.load()解析文件对象,和上面的函数区分好了
        if isinstance(data, dict) and "data" in data:
            data = data["data"]     #处理嵌套情况为了得到真正的列表对象，如果解析出来是一个字典，并且这个字典里有一个叫"data"的键，那么就把data变量更新为这个键对应的值，这样就可以处理一些json文件中数据被嵌套在一个叫"data"的字段里的情况了
        if not isinstance(data, list):
            raise ValueError(f"Bad json format, expect list[dict]: {path}")
        for obj in data:        #没有嵌套直接解析逐个取出来就好
            q = obj.get("question") or obj.get("q") or ""
            a = obj.get("answer") or obj.get("a") or []
            answers = [a] if isinstance(a, str) else (list(a) if a is not None else [])
            items.append(QAItem(q=q, a=answers))
    else:
        raise ValueError(f"Unsupported file type for local_path: {path}")

    if max_examples and max_examples < len(items):      # shuffle + truncate（与 parquet 分支行为保持一致）
        rng = np.random.RandomState(seed)       # np是numpy包的别名，np.random相当于子包或者说模块即一个py文件，Randomstate相当于是类大写开头，那么rng相当于实例类对象
        idx = rng.permutation(len(items))[:max_examples]    #permutation是RandomState类的方法，用于生成一个随机排列的整数数组，这里是生成一个长度为len(items)的随机排列，然后取前max_examples个索引，达到随机打乱和截断的目的
        items = [items[i] for i in idx]         #根据随机索引列表idx重新排列items，实现随机打乱和截断
    return items

def load_nq_open_stream(         #load_nq_open_stream函数是用来加载Natural Questions Open数据集的，返回一个列表
    split: str = "validation",
    max_examples: int = 1000,    #默认最多条数为1000条，加载的数据集中的数量
    seed: int = 42,
    local_path: Optional[str] = None,
    ##cache_dir: Optional[str] = None, 保留参数以兼容旧调用；local_path 给了就忽略
) -> List[QAItem]:
    if not local_path:
        raise ValueError("local_path is required in offline mode.")

    # 1) local_path 是文件：json/jsonl
    if os.path.isfile(local_path):
        return _load_json_or_jsonl(local_path, max_examples=max_examples, seed=seed)     #调用内部函数读取 JSON/JSONL 文件

    # 2) local_path 是目录：parquet
    if os.path.isdir(local_path):
        pqs = _find_split_parquets(local_path, split)    #返回该 split 对应的所有.parquet 文件路径列表
        ds = load_dataset("parquet", data_files={split: pqs}, split=split)  #加载数据集，huggingfaceload中dataset包的库函数,ds是Dataste类方法（函数），指定格式为 Parquet，Parquet是一种数据存储方式，它相比于csv的整行来读取某一列的信息，他可以直接实现读取某一列，同类型数据放在一块；只加载指定的 split，比如说训练集，验证集，测试集，然后每个 split 可能对应多个 shard 的 parquet 文件；进行遍历；split: pqs这个参数的意思是告诉 load_dataset 函数，数据集的 split 是什么，数据文件是哪些；比如说 split 是 validation，那么就会把 pqs 这个列表里的 parquet 文件加载到 validation 这个 split 下面，pqs是一个列表形式，举个例子比如说data_files = {"validation": ["/data/val-0001.parquet", "/data/val-0002.parquet"]}，返回的ds一个 Dataset 对象，里面封装了属性，类似列表可以用for ex in ds，但不是列表

        if max_examples and max_examples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_examples))   #随机打乱顺序，种子参数确保可复现，只选前max_examples条数据,shuffle和select都是huggingface dataset对象的方法，shuffle方法是用来打乱数据集的顺序的，select方法是用来选择数据集中的特定条目的，这里是选择前max_examples条数据

        items: List[QAItem] = []      
        for ex in ds:   #ex是字典形式，ds里面封装了属性，然后ds[0]就相当于是取出第一组数据以此类推
            q = ex.get("question", "")
            a = ex.get("answer", [])
            answers = [a] if isinstance(a, str) else (list(a) if a is not None else [])  #处理答案的多种格式，统一成列表形式，比如说如果答案是字符串，就放到列表里，如果已经是列表了，就直接用，如果是元组或者集合则也转化成了列表形式
            items.append(QAItem(q=q, a=answers))
        return items

    raise FileNotFoundError(f"local_path not found: {local_path}")
