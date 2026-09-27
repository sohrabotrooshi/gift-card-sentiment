#!/usr/bin/env python3
"""Sample a balanced, reproducible evaluation set from the Gift_Cards reviews.

Reads data/raw/Gift_Cards.jsonl.gz (Amazon Reviews '23, Gift_Cards category),
labels every review with the rating key (4-5 -> POSITIVE, 3 -> NEUTRAL,
1-2 -> NEGATIVE), then draws ~50 reviews per class with a fixed random seed
so the same set comes out every run. Writes results/sample_balanced.json.

Why balanced? The corpus is ~88.5% positive, so reading the file head-first
under-represents the rare classes (the first 100 rows contained 0 neutrals).
Equal per-class sampling makes every class equally testable.
"""
import gzip, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Gift_Cards.jsonl.gz"
OUT = ROOT / "results" / "sample_balanced.json"
SEED = 20260914   # fixed: same sample every run
N_PER_CLASS = 50


def gold(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def clean(text: str) -> str:
    import html
    import re
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    bins = {"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []}
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g = gold(r["rating"])
            bins[g].append((i, r["rating"], clean(r["title"]), clean(r["text"])))

    print("corpus pools:", {k: len(v) for k, v in bins.items()})
    rng = random.Random(SEED)
    sample = []
    for g in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        sample += [
            {"line": t[0], "rating": t[1], "title": t[2], "text": t[3], "gold": g}
            for t in rng.sample(bins[g], N_PER_CLASS)
        ]
    rng.shuffle(sample)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sample, ensure_ascii=False))
    print(f"wrote {len(sample)} reviews -> {OUT}"
          f"  ({ {g: sum(1 for r in sample if r['gold'] == g) for g in bins} })")


if __name__ == "__main__":
    sys.exit(main())
