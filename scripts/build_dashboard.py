#!/usr/bin/env python3
"""Dashboard generator: merge the run's raw outputs and build the self-contained HTML.

1. Joins results/sample_balanced.json (gold labels), results/sample150_emote_llm.csv
   (LLM sentiment + emotion) and results/emote_wordlist.json (NRC lexicon emotion)
   into results/evaluation_balanced.json — one record per review with
   {line, rating, title, text, gold, pred, emo_llm, emo_wl, emo_agree}.
2. Injects that JSON into dashboard/dashboard_template.html (replacing the
   __DATA__ placeholder) and writes dashboard/sentiment_dashboard.html —
   a single self-contained file: all data baked in, zero requests, works offline.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "results" / "sample_balanced.json"
LLM = ROOT / "results" / "sample150_emote_llm.csv"
WL = ROOT / "results" / "emote_wordlist.json"
MERGED = ROOT / "results" / "evaluation_balanced.json"
TEMPLATE = ROOT / "dashboard" / "dashboard_template.html"
OUT = ROOT / "dashboard" / "sentiment_dashboard.html"


def main() -> None:
    sample = json.loads(SAMPLE.read_text())
    llm = {int(r["line"]): r for r in csv.DictReader(LLM.open())}
    wl = {int(r["line"]): r for r in json.loads(WL.read_text())}

    for rec in sample:
        L = rec["line"]
        rec["pred"] = llm[L]["sentiment"]
        rec["emo_llm"] = llm[L]["emotion"]
        rec["emo_wl"] = wl[L]["lexicon_emotion"]
        rec["emo_agree"] = rec["emo_llm"] == rec["emo_wl"]

    MERGED.write_text(json.dumps(sample, ensure_ascii=False))

    tpl = TEMPLATE.read_text()
    assert "__DATA__" in tpl, "template has no __DATA__ placeholder"
    html = tpl.replace("__DATA__", json.dumps(sample, ensure_ascii=False))
    OUT.write_text(html)
    print(f"merged {len(sample)} records -> {MERGED}")
    print(f"built  {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
