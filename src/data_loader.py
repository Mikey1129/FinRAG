import json

def load_finga(path, limit=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        qa = item.get("qa", {})
        samples.append({
            "id": item.get("id"),
            "question": qa.get("question",""),
            "answer": qa.get("answer",""),
            "program": qa.get("program",""),
            "table": item.get("table") or [],
            "pre_text":item.get("pre_text") or [],
            "post_text":item.get("post_text") or [],
            "ann_table_rows": qa.get("ann_table_rows") or [],
            "ann_text_rows": qa.get("ann_text_rows") or []
        })
        if limit and len(samples) >= limit:
            break
    return samples
