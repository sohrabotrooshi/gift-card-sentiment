#!/usr/bin/env python3
"""Word-list emotion take: primary emotion from the NRC lexicon (no model calls).

For each review, tokenize the text, look up each token in the NRC word-level
emotion lexicon (lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt), sum the
association scores per emotion among the eight NRC emotions, and take the
highest-scoring emotion as the primary. Ties resolve to the first emotion in
NRC order; reviews with no emotional words get "neutral".

Writes results/emote_wordlist.json.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / "lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
DATA = ROOT / "results" / "sample_balanced.json"
OUT = ROOT / "results" / "emote_wordlist.json"

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
EMO = set(EMOTIONS)

TOKEN = re.compile(r"[a-z']+")
# minimal function words with no emotional load, kept in the stop-list so gift-card
# filler ("the", "and") never outscores real emotion words
STOP = {"a", "an", "the", "of", "to", "and", "or", "for", "in", "on", "at", "i",
        "it", "is", "was", "be", "as", "my", "me", "we", "you", "your", "this",
        "that", "with", "have", "had", "has", "not", "but", "so", "if", "they",
        "them", "their", "there", "she", "he", "her", "his", "all", "no", "do",
        "did", "by", "from", "get", "got"}


def load_lexicon():
    lex = defaultdict(set)  # word -> set of emotions with association == 1
    with open(LEXICON, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\r\n").strip("\ufeff").split("\t")
            if len(parts) != 3:
                continue
            word, emo, assoc = (p.strip().lower() for p in parts)
            if emo in EMO and assoc == "1" and word:
                lex[word].add(emo)
    return lex


def primary(review, lex):
    counts = Counter()
    hits = 0
    for tok in TOKEN.findall((review["text"] or "").lower()):
        if len(tok) < 2 or tok in STOP:
            continue
        emos = lex.get(tok)
        if not emos:
            continue
        hits += 1
        for e in emos:
            counts[e] += 1
    if hits == 0:
        return {"line": review["line"], "lexicon_emotion": "neutral",
                "top_score": 0, "all_zero": True, "tie": False, "hits": 0,
                "scores": {e: 0 for e in EMOTIONS}}
    top = max(counts.values())
    winners = [e for e in EMOTIONS if counts[e] == top]
    return {"line": review["line"], "lexicon_emotion": winners[0],
            "top_score": top, "all_zero": False, "tie": len(winners) > 1,
            "hits": hits, "scores": {e: counts[e] for e in EMOTIONS}}


def main() -> None:
    lex = load_lexicon()
    print("lexicon emotions loaded for", len(lex), "words")
    reviews = json.loads(DATA.read_text())
    out = [primary(r, lex) for r in reviews]
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    n_lex = sum(1 for r in out if not r["all_zero"])
    from collections import Counter
    dist = Counter(r["lexicon_emotion"] for r in out)
    print(f"lexicon hits on {n_lex}/{len(out)} reviews, "
          f"all-neutral {len(out)-n_lex}, ties {sum(1 for r in out if r['tie'])}")
    print("primary emotion distribution:", dict(dist))
    print("wrote ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
