"""FinRAG 财报数值问答 Web 应用（Streamlit）。

运行：
    streamlit run app.py
浏览器打开 http://localhost:8501
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import streamlit as st

import ask
from ask import build_sample, parse_table, solve_explain

st.set_page_config(page_title="FinRAG 财报数值问答", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(1200px 600px at 85% -10%, #16233f 0%, #0b0f1a 55%);
    }
    section[data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid #1e293b;
    }
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 22px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 30px; font-weight: 700; color: #fff; letter-spacing: .3px;
    }
    .hero-sub {
        font-size: 14px; color: rgba(255,255,255,.72); margin-top: 6px;
    }
    .badge {
        display: inline-block; background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.20); color: #fff;
        border-radius: 999px; padding: 4px 14px; font-size: 12px;
        margin: 12px 8px 0 0; backdrop-filter: blur(4px);
    }
    .step-label {
        font-size: 12px; font-weight: 700; color: #93c5fd;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 2px;
    }
    .stTextArea textarea, .stTextInput input {
        background: #0f1626 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        color: #e8ecf4 !important;
    }
    .stButton > button {
        border-radius: 10px; border: none; font-weight: 600;
        transition: all .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(59,130,246,.35);
    }
    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden; border: 1px solid #1e293b;
    }
    .answer-card {
        background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 100%);
        border-radius: 16px; padding: 22px 28px;
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
        margin: 6px 0 4px 0;
    }
    .answer-label {
        font-size: 12px; color: rgba(255,255,255,.7);
        letter-spacing: 2px; text-transform: uppercase;
    }
    .answer-value {
        font-size: 42px; font-weight: 700; color: #fff; margin-top: 2px;
        font-family: 'Consolas','Menlo',monospace;
    }
    h1, h2, h3, h4 { color: #e8ecf4 !important; }
    .footer {
        text-align: center; color: #475569; font-size: 12px; margin-top: 30px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_TABLE = (
    "指标,2014,2013\n"
    "notional/contract amount,36197,29270\n"
    "asset fair value (a),1189,1872\n"
    "liability fair value (b),364,152\n"
)
EXAMPLE_TEXT = (
    "The company's derivatives designated as hedging instruments under GAAP had "
    "a notional amount of $36,197 million at the end of 2014, with an asset fair "
    "value of $1,189 million."
)
EXAMPLE_QUESTION = "at the end of 2014, the notional value of derivatives was what percent of the fair value?"

# ---- Hero ----
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📊 FinRAG 财报数值问答</div>
        <div class="hero-sub">粘贴财报 → 提问 → 得到可追溯答案，每个答案都能追到「取了哪两个数、怎么算」</div>
        <div>
            <span class="badge">严格 EM 0.84</span>
            <span class="badge">宽松 EM 0.90</span>
            <span class="badge">TF-IDF + LLM 代码生成</span>
            <span class="badge">DeepSeek v4-pro</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ 参数")
    k = st.slider("文本检索 top-k", 1, 50, 20)
    st.divider()
    st.markdown("### 🧭 工作流程")
    st.markdown(
        "1. 🔍 **检索** — TF-IDF 召回相关行/文本\n"
        "2. ✍️ **生成** — LLM 编写 Python 代码\n"
        "3. ⚙️ **执行** — `exec` 得到精确数值\n"
        "4. ✅ **可追溯** — 答案 + 依据 + 代码"
    )
    st.divider()
    if st.button("🧪 加载示例数据", use_container_width=True):
        st.session_state.table_raw = EXAMPLE_TABLE
        st.session_state.text_raw = EXAMPLE_TEXT
        st.session_state.question = EXAMPLE_QUESTION
        st.rerun()

# ---- 输入区 ----
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="step-label">① 财报表格</div>', unsafe_allow_html=True)
    table_raw = st.text_area(
        "第一行表头，逗号或制表符分隔（可从 Excel 复制粘贴）",
        key="table_raw", height=170,
    )
with c2:
    st.markdown('<div class="step-label">② 财报文本（可选）</div>', unsafe_allow_html=True)
    text_raw = st.text_area(
        "题干之外的背景段落",
        key="text_raw", height=170,
    )

st.markdown('<div class="step-label">③ 问题</div>', unsafe_allow_html=True)
q1, q2 = st.columns([5, 1])
with q1:
    question = st.text_input("输入数值问题", key="question", label_visibility="collapsed")
with q2:
    run = st.button("🚀 计算答案", type="primary", use_container_width=True)

# ---- 表格预览 ----
table = parse_table(table_raw)
if table:
    st.markdown("#### 📋 解析后的表格")
    st.dataframe(pd.DataFrame(table[1:], columns=table[0]))

# ---- 计算 ----
if run:
    if not table:
        st.error("请先粘贴财报表格")
    elif not question.strip():
        st.error("请输入问题")
    else:
        texts = [l.strip() for l in text_raw.splitlines() if l.strip()]
        sample = build_sample(question.strip(), table, texts)
        with st.spinner("检索 + 推理中（约 65s）..."):
            pred, code, chunks = solve_explain(sample, k=k)
        st.session_state.result = {"pred": pred, "code": code, "chunks": chunks}

# ---- 结果 ----
if "result" in st.session_state:
    r = st.session_state.result
    st.divider()
    pred_display = "—" if r["pred"] is None else str(r["pred"])
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">答案 Answer</div>
            <div class="answer-value">{pred_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["📋 推导依据", "🐍 生成代码"])
    with tab1:
        table_rows = [c for c in r["chunks"] if c[0][0] == "table"]
        text_rows = [c for c in r["chunks"] if c[0][0] == "text"]
        if table_rows:
            st.markdown("**表格行**（全部常驻）")
            for _, text in table_rows:
                st.markdown(f"- `{text}`")
        if text_rows:
            st.markdown("**文本段落**（TF-IDF top-k）")
            for _, text in text_rows:
                st.markdown(f"- `{text}`")
    with tab2:
        st.code(r["code"] if r["code"] else "(未能生成有效代码)", language="python")

st.markdown(
    '<div class="footer">FinRAG · TF-IDF 检索 + LLM 代码生成 · FinQA 严格 EM 0.84 / 宽松 EM 0.90</div>',
    unsafe_allow_html=True,
)
