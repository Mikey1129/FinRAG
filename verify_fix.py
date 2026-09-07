import sys

sys.stdout.reconfigure(encoding="utf-8")

def normalize(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return s.replace(",", "").replace("$", "").replace("%", "")

def em_match(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        fp, fg = float(p), float(g)
        return abs(fp - fg) <= max(1e-2 * abs(fg), 1e-2)
    except ValueError:
        return False

# (id, 模型预测值[来自 eval_100_v3.txt], 原gold, 修正gold)
cases = [
    ("PM/2015/page_127.pdf-4", -4088.3333333333335, '-6806', '-4088.333'),
    ("ABMD/2009/page_88.pdf-1", 16750000, '5583331', '16750000'),
    ("ADI/2011/page_81.pdf-1", 34.922377887163954, '65.1%', '34.9%'),
    ("MRO/2006/page_33.pdf-1", 40444920000.0, '$ 40444920\n', None),
]

saved = 0
for cid, pred, old_gold, new_gold in cases:
    old_ok = em_match(pred, old_gold)
    if new_gold is not None:
        new_ok = em_match(pred, new_gold)
        tag = "救回 ✓" if (not old_ok and new_ok) else ("本来就对" if old_ok else "仍错")
        if not old_ok and new_ok:
            saved += 1
        print(f"{cid}\n  预测={pred}\n  原gold={old_gold!r} → {'对' if old_ok else '错'}\n  修正={new_gold!r} → {'对' if new_ok else '错'}   [{tag}]\n")
    else:
        print(f"{cid}\n  预测={pred}\n  原gold={old_gold!r} → {'对' if old_ok else '错'}   [非answer-bug，不修]\n")

print(f"离线验证：改 answer 可救回 {saved} 题 → EM 76 → {76 + saved}（严格口径）")
