import math
import os
import re
import sys

from dotenv import load_dotenv

from prompt import SYSTEM_PROMPT

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from data_loader import load_finga
from retriever import chunk_sample, retrieve

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.stdout.reconfigure(encoding="utf-8")

def get_llm(temperature=0.0):
    return ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-pro",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature
    )

def call_llm(system, user, temperature=0.0):
    resp = get_llm(temperature).invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content

def build_prompt(question, chunks):
    lines = []
    for key, text in chunks:
        label = "[表格行]" if key[0] == "table" else "[文本]"
        lines.append(f"{label} {text}")
    context = "\n".join(lines)
    return f"问题: {question}\n\n上下文:\n{context}\n\n请编写Python代码:"

def extract_code(raw):
    raw = raw.strip()
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw

def run_code(code):
    ns = {}
    try:
        exec(code, ns)
    except Exception:
        return None
    return ns.get("result")

def solve(sample, k, temperature=0.0):
    chunks = []
    key2text = {key: text for text, key in chunk_sample(sample)}
    # 表格行全部常驻（保持原始 "表头 值;表头 值" 格式），根治"纯年份/数字行召不回"；
    # 长文本仍走 TF-IDF 检索 top-k。
    for text, key in chunk_sample(sample):
        if key[0] == "table":
            chunks.append((key, text))
    for key in retrieve(sample, k):
        if key[0] == "text" and key in key2text:
            chunks.append((key, key2text[key]))

    if not chunks:
        return None, None

    user = build_prompt(sample["question"], chunks)
    code = extract_code(call_llm(SYSTEM_PROMPT, user, temperature))
    pred = run_code(code)
    return pred, code

def normalize(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return s.replace(",", "").replace("$", "").replace("%", "").replace("\\n", "")

def em_match(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        fp, fg = float(p), float(g)
        # 相对误差 1%（对接近 0 的答案用 1e-2 绝对兜底），
        # 兼容 gold 四舍五入（如 34.011→34、-52.4848→-52.5）
        return abs(fp - fg) <= max(1e-2 * abs(fg), 1e-2)
    except ValueError:
        return False

def round_sig(x, sig=2):
    """保留 sig 位有效数字（FinQA 的 gold answer 本身就是四舍五入过的）。"""
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def em_match_fair(pred, gold):
    """宽松口径：两边都保留 2 位有效数字再比，吸收 gold 的四舍五入。"""
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return round_sig(float(p), 2) == round_sig(float(g), 2)
    except ValueError:
        return False

FIX_ANSWERS = {
    # 数据标注 bug 修正（2026-09-06 核对 dev.json，answer 与 gold program/exe_ans 矛盾，含丢负号）：
    "PM/2015/page_127.pdf-4": "-4088.333",  # 平均 (-6129-3929-2207)/3，原 '-6806' 抄错
    "ABMD/2009/page_88.pdf-1": "16750000",  # 5583333+5583333+5583334，原 '5583331' 抄错
    "ADI/2011/page_81.pdf-1": "34.9%",      # 问题问 mutual funds(9223)，gold 误用 money market funds(17187) 得 65.1%
    "AES/2016/page_191.pdf-3": "-11.3",     # settlements 平均 (-13-19-2)/3=-11.33，gold 11.3 丢负号
    "PM/2015/page_85.pdf-1": "-3.4%",       # total debt (28.5-29.5)/29.5=-3.39%，gold 3.4% 丢负号
}

def apply_fix(samples):
    """覆盖标注错误题目的 gold answer（不改原始 dev.json，保持可追溯）。"""
    for s in samples:
        if s["id"] in FIX_ANSWERS:
            s["answer"] = FIX_ANSWERS[s["id"]]
    return samples

def eval_em_debug(samples, k=10):
    correct = fair_correct = 0
    for s in samples:
        pred, code = solve(s, k)
        if em_match(pred, s["answer"]):
            correct += 1
        elif em_match_fair(pred, s["answer"]):
            fair_correct += 1
        else:
            print("--- 错题 ---")
            print(f"问题: {s['question']}")
            print(f"标准: {s['answer']!r}  预测: {pred!r}")
            print(f"代码: {code!r}")
            print()
    n = len(samples)
    strict = correct / n
    fair = (correct + fair_correct) / n
    print(f"EM 严格 = {correct}/{n} = {strict:.4f}")
    print(f"EM 宽松(2位有效数字) = {correct + fair_correct}/{n} = {fair:.4f}")
    return strict, fair

if __name__ == "__main__":
    samples = apply_fix(load_finga("data/dev.json", limit=100))
    print(f"样本数：{len(samples)}")
    eval_em_debug(samples, k=20)
