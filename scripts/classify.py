#!/usr/bin/env python3
"""Classification script — legacy binary version (first-100 experiment).

History: the very first batch classified the file's first 100 rows with a binary
schema (rating >= 4 -> POSITIVE, else NEGATIVE). Results are reported in README.md
under "The lopsided run". The final project uses the 3-class schema via
emote_llm.py instead; this file is kept for reproducibility of that first run.

Run: python3 classify.py <seed> <n_pos> <n_neg>           (balanced eval mode)
     python3 classify.py 7 3 3                             -> quick smoke test
"""
import csv
import html
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Gift_Cards.jsonl"
if not DATA.exists():  # the repo ships the .gz; decompress if needed
    import gzip
    with gzip.open(DATA.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as src, \
            open(DATA, "w", encoding="utf-8") as dst:
        dst.write(src.read())

API = "http://dobolyi.com:9001/v1/chat/completions"
KEY = "6418"
MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

SYSTEM = ("You are an Amazon review sentiment classifier. Given a product's review "
          "(title and body), classify the overall sentiment as POSITIVE or NEGATIVE. "
          "A mostly positive review that mentions a minor issue is POSITIVE; a mostly "
          "negative review is NEGATIVE. Reply with EXACTLY one token: POSITIVE or NEGATIVE.")
FEWSHOT = [
    ("Great gift", "Having Amazon money is always good.", "POSITIVE"),
    ("Worst gift card", "The card arrived with a zero balance and they never fixed it. "
     "Terrible customer service. Do not buy.", "NEGATIVE"),
    ("Amazon gift card", "Always the perfect gift. They are thrilled and excited "
     "to have a bit of a spree. Arrives in 1 day in most cases.", "POSITIVE"),
    ("Not $10 Gift Cards", "Used one card and it had $6.52 not $10.00. The other had "
     "$5.32. Random amounts on them. I'm embarrassed to have given them as gifts.", "NEGATIVE"),
]


def clean(t):
    t = html.unescape(t or "")
    t = re.sub(r"<br\s*/?>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def classify(title, text, retries=4):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ti, tx, lab in FEWSHOT:
        msgs.append({"role": "user", "content": f"Title: {ti}\nReview: {tx}\nSentiment:"})
        msgs.append({"role": "assistant", "content": lab})
    msgs.append({"role": "user", "content": f"Title: {title}\nReview: {text}\nSentiment:"})
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": 8, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(API, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {KEY}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=120) as r:
                obj = json.loads(r.read().decode())
            up = (obj["choices"][0]["message"].get("content") or "").upper()
            for tok in ("POSITIVE", "NEGATIVE"):
                if tok in up:
                    return tok
        except Exception:
            pass
        time.sleep(1 + i)
    return "ERROR"


def main():
    seed, n_pos, n_neg = (int(a) for a in sys.argv[1:4]) if len(sys.argv) > 3 else (42, 150, 150)
    pos, neg = [], []
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["rating"] >= 4:
                pos.append((clean(r["title"]), clean(r["text"])))
            elif r["rating"] <= 2:
                neg.append((clean(r["title"]), clean(r["text"])))
    rng = random.Random(seed)
    sample = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(sample)
    rows, t0 = [], time.time()
    for i, (title, text) in enumerate(sample, 1):
        pred = classify(title, text)
        rows.append({"true": "POSITIVE" if i <= n_pos else "NEGATIVE",
                     "pred": pred, "title": title, "text": text})
        print(f"[{i}/{len(sample)}] pred={pred} eta~{(len(sample)-i)*(time.time()-t0)/i:.0f}s", flush=True)
    out = ROOT / "results" / "classify_binary_sample.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["true", "pred", "title", "text"])
        w.writeheader()
        w.writerows(rows)
    print("DONE ->", out)


if __name__ == "__main__":
    sys.exit(main())
