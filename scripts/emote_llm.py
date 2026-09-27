#!/usr/bin/env python3
"""Scoring script: run the balanced sample through the LLM sentiment+emotion classifier.

For every review in results/sample_balanced.json, calls the OpenAI-compatible
vLLM endpoint (Qwen3.6-35B) with the prompt in prompts/prompt_sentiment_emotion.md
and writes one JSON line per review to results/sample150_emote_llm.csv.

The review's RATING IS NEVER SENT TO THE MODEL — the input is only title + text.

Environment overrides (optional):
    LLM_API_URL   default http://dobolyi.com:9001/v1/chat/completions
    LLM_API_KEY   default 6418
"""
import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "sample_balanced.json"
OUT = ROOT / "results" / "sample150_emote_llm.csv"

API = os.environ.get("LLM_API_URL", "http://dobolyi.com:9001/v1/chat/completions")
KEY = os.environ.get("LLM_API_KEY", "6418")
MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust", "neutral"]
EMO_LIST = "anger, anticipation, disgust, fear, joy, sadness, surprise, trust"

# --- the prompt (kept in sync with prompts/prompt_sentiment_emotion.md) ----
SYSTEM = (
    "You are an Amazon review classifier. Given a review (title and body) return a "
    "JSON object with exactly two fields: \"sentiment\" = one of \"POSITIVE\", "
    "\"NEUTRAL\", \"NEGATIVE\"; rating scales: 4-5 stars is POSITIVE, 3 stars is "
    "NEUTRAL (mixed, indifferent, or minor gripes without strong praise), 1-2 stars "
    "is NEGATIVE. And \"emotion\" = the single dominant emotion, exactly one of: " +
    EMO_LIST + ", or \"neutral\" when no emotion is clearly expressed. "
    "Return only the JSON, no commentary."
)

FEWSHOT = [
    ("Great gift", "Having Amazon money is always good.",
     {"sentiment": "POSITIVE", "emotion": "joy"}),
    ("Worst gift card", "The card arrived with a zero balance and they never fixed it. "
     "Terrible customer service. Do not buy.",
     {"sentiment": "NEGATIVE", "emotion": "anger"}),
    ("Okay", "It did its job. Nothing special, would not rush to buy again.",
     {"sentiment": "NEUTRAL", "emotion": "neutral"}),
    ("Amazon gift card", "Always the perfect gift. They are thrilled and excited to "
     "have a bit of a spree. Arrives in 1 day.",
     {"sentiment": "POSITIVE", "emotion": "anticipation"}),
    ("Not $10 Gift Cards", "Used one card and it had $6.52 not $10.00. The other had "
     "$5.32. Random amounts on them. Embarrassed to give them as gifts.",
     {"sentiment": "NEGATIVE", "emotion": "disgust"}),
]


def build_messages(title: str, text: str):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ti, tx, ans in FEWSHOT:
        msgs.append({"role": "user", "content": f"Title: {ti}\nReview: {tx}"})
        msgs.append({"role": "assistant", "content": json.dumps(ans)})
    msgs.append({"role": "user", "content": f"Title: {title}\nReview: {text}"})
    return msgs


def parse(content: str):
    m = re.search(r'"sentiment"\s*:\s*"([A-Za-z]+)"', content)
    m2 = re.search(r'"emotion"\s*:\s*"([A-Za-z]+)"', content)
    sent = (m.group(1).upper() if m else "")
    sent = sent if sent in ("POSITIVE", "NEUTRAL", "NEGATIVE") else None
    emo = (m2.group(1).lower() if m2 else "")
    emo = emo if emo in EMOTIONS else None
    return sent, emo


def call(title: str, text: str, retries: int = 4):
    body = json.dumps({
        "model": MODEL,
        "messages": build_messages(title, text),
        "max_tokens": 60,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 thinking off
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {KEY}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode())
            return parse(obj["choices"][0]["message"].get("content") or "")
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            time.sleep(1 + attempt)
    return None, f"ERROR:{last}"


def main() -> None:
    reviews = json.loads(DATA.read_text())
    out = []
    t0 = time.time()
    for i, r in enumerate(reviews, 1):
        sent, emo = call(r["title"], r["text"])
        out.append({"line": r["line"], "sentiment": sent, "emotion": emo,
                    "title": r["title"], "text": r["text"]})
        el = time.time() - t0
        print(f"[{i}/{len(reviews)}] line={r['line']} sent={sent} emo={emo} "
              f"eta~{(len(reviews)-i)*(el/i):.0f}s", flush=True)
        if i % 25 == 0:
            OUT.with_name(OUT.stem + "_partial.json").write_text(json.dumps(out))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["line", "sentiment", "emotion", "title", "text"])
        w.writeheader()
        w.writerows(out)
    print(f"DONE {len(out)} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
