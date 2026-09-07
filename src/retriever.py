import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_finga

sys.stdout.reconfigure(encoding="utf-8")

def chunk_sample(sample):
    chunks = []

    all_text = sample["pre_text"] + sample["post_text"]
    for i,t in enumerate(all_text):
        if t.strip():
            chunks.append((t,("text",i)))

    table = sample["table"]
    if table:
        header = table[0]
        for r in range(1, len(table)):
            row_text = ";".join(f"{h} {v}" for h, v in zip(header, table[r]))
            chunks.append((row_text, ("table", r)))

    return chunks

def format_full_table(sample):
    table = sample.get("table")
    if not table:
        return None
    rows = [" | ".join(str(c) for c in row) for row in table]
    return "\n".join(rows)

def retrieve(sample, k=20):
    chunks = chunk_sample(sample)
    if not chunks:
        return[]

    corpus = [text for text, _ in chunks]
    vec =  TfidfVectorizer()
    corpus_vec = vec.fit_transform(corpus)
    q_vec = vec.transform([sample["question"]])

    sim = cosine_similarity(q_vec, corpus_vec)[0]
    order = sim.argsort()[::-1][:k]
    return [chunks[i][1] for i in order]

def eval_recall(samples, k):
    total, n = 0.0, 0
    for s in samples:
        gold = {("text", i) for i in s["ann_text_rows"]} | {("table", r) for r in s["ann_table_rows"]}
        if not gold:
            continue
        hit = set(retrieve(s, k))
        total += len(gold & hit) / len(gold)
        n += 1
    return total / n if n else 0.0

if __name__ == "__main__":
    samples = load_finga("data/dev.json", limit=None)
    print(f"样本数：{len(samples)}")
    for k in (5, 10, 20):
        print(f"recall@{k} = {eval_recall(samples, k):.4f}")