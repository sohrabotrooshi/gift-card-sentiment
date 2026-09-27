# Classification prompt — sentiment + primary emotion

Used by `scripts/emote_llm.py` against the OpenAI-compatible vLLM endpoint
(`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`). The reviewer's **rating is never part of
the prompt** — only `Title:` and `Review:` are sent; the rating is used purely
as the held-out answer key afterwards.

## System message

> You are an Amazon review classifier. Given a review (title and body) return a
> JSON object with exactly two fields: `"sentiment"` = one of `"POSITIVE"`,
> `"NEUTRAL"`, `"NEGATIVE"`; rating scales: 4-5 stars is POSITIVE, 3 stars is
> NEUTRAL (mixed, indifferent, or minor gripes without strong praise), 1-2 stars
> is NEGATIVE. And `"emotion"` = the single dominant emotion, exactly one of:
> anger, anticipation, disgust, fear, joy, sadness, surprise, trust, or
> `"neutral"` when no emotion is clearly expressed. Return only the JSON,
> no commentary.

## Few-shot examples (prepended as user/assistant turns)

| Title | Review | Expected |
|---|---|---|
| Great gift | Having Amazon money is always good. | `{"sentiment": "POSITIVE", "emotion": "joy"}` |
| Worst gift card | The card arrived with a zero balance and they never fixed it. Terrible customer service. Do not buy. | `{"sentiment": "NEGATIVE", "emotion": "anger"}` |
| Okay | It did its job. Nothing special, would not rush to buy again. | `{"sentiment": "NEUTRAL", "emotion": "neutral"}` |
| Amazon gift card | Always the perfect gift. They are thrilled and excited to have a bit of a spree. Arrives in 1 day. | `{"sentiment": "POSITIVE", "emotion": "anticipation"}` |
| Not $10 Gift Cards | Used one card and it had $6.52 not $10.00. The other had $5.32. Random amounts on them. Embarrassed to give them as gifts. | `{"sentiment": "NEGATIVE", "emotion": "disgust"}` |

## Request shape (OpenAI chat completions)

```json
{
  "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "max_tokens": 60,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

`enable_thinking: false` matters: the model is a Qwen3 "thinking" model, and with
thinking enabled the answer lands in the `reasoning` field (content comes back
null). Disabling it keeps the output deterministic, fast, and JSON-only.

---

Earlier prototype (binary, first-100-row experiments): same shape but sentiment
was only `"POSITIVE"` / `"NEGATIVE"` and the system message said
"4-5 stars is POSITIVE, 1-3 stars is NEGATIVE". Superseded by the 3-class schema.
