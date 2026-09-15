"""FinRAG 命令行问答工具 —— 粘贴财报 + 提问 → 得到可追溯答案。

用法：
  python ask.py            交互式：粘贴财报表格/文本，然后循环提问
  python ask.py --demo     演示模式：从 dev.json 选样本，展示完整检索+推理
"""

import csv
import io
import random
import sys

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

import reasoner as R
from prompt import SYSTEM_PROMPT
from data_loader import load_finga
from retriever import chunk_sample, retrieve


def make_chunks(sample, k=20):
    """构造检索上下文：表格行全部常驻，文本段走 TF-IDF top-k。"""
    chunks = []
    key2text = {key: text for text, key in chunk_sample(sample)}
    for text, key in chunk_sample(sample):
        if key[0] == "table":
            chunks.append((key, text))
    for key in retrieve(sample, k):
        if key[0] == "text" and key in key2text:
            chunks.append((key, key2text[key]))
    return chunks


def solve_explain(sample, k=20):
    """作答并返回 (答案, 生成代码, 检索到的依据块)。"""
    chunks = make_chunks(sample, k)
    user = R.build_prompt(sample["question"], chunks)
    raw = R.call_llm(SYSTEM_PROMPT, user, 0.0)
    code = R.extract_code(raw)
    pred = R.run_code(code)
    return pred, code, chunks


def parse_table(raw):
    """把粘贴的表格文本解析成 rows（第一行是表头）。自动识别逗号/制表符分隔。"""
    raw = raw.strip()
    if not raw:
        return []
    first = raw.splitlines()[0]
    delim = "\t" if first.count("\t") >= first.count(",") else ","
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
    return [r for r in rows if any(c.strip() for c in r)]


def build_sample(question, table, texts):
    return {
        "question": question,
        "table": table,
        "pre_text": texts,
        "post_text": [],
    }


def read_multiline():
    """逐行读入直到遇到空行。"""
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def print_answer(pred, code, chunks):
    print("\n" + "=" * 60)
    print("答案:", pred)
    print("-" * 60)
    print("依据（检索到的行/段落）:")
    for key, text in chunks:
        label = "[表格行]" if key[0] == "table" else "[文本]"
        print(f"  {label} {text[:100]}")
    print("-" * 60)
    print("生成的代码:")
    print(code if code else "(未能生成有效代码)")
    print("=" * 60 + "\n")


def interactive():
    print("=" * 60)
    print("FinRAG 财报数值问答")
    print("=" * 60)
    print("第一步：粘贴财报表格（第一行表头，逗号或制表符分隔）。")
    print("        粘贴完后按回车，再空一行结束。")
    table = parse_table(read_multiline())
    if not table:
        print("未读到表格，退出。")
        return
    print(f"已载入 {len(table) - 1} 行数据、{len(table[0])} 列。\n")

    print("第二步：粘贴财报文本段落（可选，直接空行跳过）")
    texts_raw = read_multiline()
    texts = [l.strip() for l in texts_raw.splitlines() if l.strip()]
    print(f"已载入 {len(texts)} 段文本。\n")

    print("第三步：输入问题（每条约 65s，输入 quit 退出）")
    while True:
        q = input("\n问题> ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        sample = build_sample(q, table, texts)
        pred, code, chunks = solve_explain(sample, k=20)
        print_answer(pred, code, chunks)


def demo():
    samples = load_finga("data/dev.json", limit=100)
    print("演示模式：从 dev.json 前 100 题里挑一个样本，展示完整检索+推理流程。")
    print("可选样本（输入编号 0-4，直接回车=随机）：")
    for i in range(5):
        print(f"  [{i}] {samples[i]['question'][:70]}")
    idx = input("编号> ").strip()
    s = random.choice(samples) if idx == "" else samples[int(idx)]
    print("\n问题:", s["question"])
    pred, code, chunks = solve_explain(s, k=20)
    print_answer(pred, code, chunks)
    print("标准答案(供对照):", repr(s["answer"]))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        interactive()
