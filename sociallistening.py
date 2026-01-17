import os
import json
import time
from typing import Dict, Any

import pandas as pd
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
MODEL = "gpt-5-mini"
INPUT_CSV = "input.csv"          # rename your file to input.csv or change this
TEXT_COL = "Message"

META_COLS = ["SenderScreenName", "CreatedTime", "MessageType", "Country", "State", "City"]

OUTPUT_JSONL = "topics.jsonl"
OUTPUT_CSV = "topics.csv"

MAX_CHARS_PER_POST = 2000
SLEEP_BETWEEN_CALLS_SEC = 0.2
MAX_RETRIES = 5

# ----------------------------
# Locked taxonomy (topics)
# ----------------------------
TOPICS = [
    {"topic_id": 1, "topic": "Health & Wellness"},
    {"topic_id": 2, "topic": "Fitness & Physical Activity"},
    {"topic_id": 3, "topic": "Career Advancement"},
    {"topic_id": 4, "topic": "Learning & Upskilling"},
    {"topic_id": 5, "topic": "Income & Financial Growth"},
    {"topic_id": 6, "topic": "Financial Discipline & Security"},
    {"topic_id": 7, "topic": "Relationships & Romantic Life"},
    {"topic_id": 8, "topic": "Family & Parenting"},
    {"topic_id": 9, "topic": "Social Life & Friendships"},
    {"topic_id": 10, "topic": "Housing & Property"},
    {"topic_id": 11, "topic": "Personal Growth & Mindset"},
    {"topic_id": 12, "topic": "Lifestyle & Experiences"},
    {"topic_id": 13, "topic": "Entrepreneurship & Business Building"},
    {"topic_id": 14, "topic": "Community, Purpose & Giving Back"},
    {"topic_id": 15, "topic": "Other / Unclear"},
]
TOPIC_NAME_TO_ID = {t["topic"]: t["topic_id"] for t in TOPICS}
ALLOWED_TOPICS = list(TOPIC_NAME_TO_ID.keys())
OTHER_TOPIC = "Other / Unclear"

# ----------------------------
# Locked taxonomy (sentiment)
# ----------------------------
ALLOWED_SENTIMENT = ["Positive", "Negative", "Neutral", "Mixed", "Unclear"]

SYSTEM_PROMPT = f"""You are a social media text analyst.

Task:
1) I need to analyse what are the top resolutions that people make. You will receive unfiltered raw Reddit messages that contain the keyword "new year resolution". 
2) Classify ONE social media message into EXACTLY ONE topic bucket from the locked list.
3) Assign a NEW sentiment label for the message (ignore any existing vendor sentiment).

Locked topic buckets (choose exactly one string, match spelling/case exactly):
{json.dumps(ALLOWED_TOPICS, ensure_ascii=False)}

Locked sentiment labels (choose exactly one string, match spelling/case exactly):
{json.dumps(ALLOWED_SENTIMENT, ensure_ascii=False)}

Return ONLY valid JSON with this schema:
{{
  "topic": string,                  // must be one of the locked topic buckets exactly
  "subtopic": string|null,          // optional refinement (free text)
  "confidence": number,             // topic confidence 0 to 1
  "rationale": string,              // <= 20 words
  "newSentiment": string,           // must be one of the locked sentiment labels exactly
  "newSentimentConfidence": number  // 0 to 1
}}

Rules:
- If the message is mostly noise, sarcasm, too vague, or you cannot decide: use "{OTHER_TOPIC}" and sentiment "Unclear" with low confidence. This includes messages that do not have clear indication of a new year resolution. Do not infer.
- Do not invent new labels outside the locked lists.
- Pick the dominant intent if multiple goals appear.
"""

client = OpenAI()  # reads OPENAI_API_KEY from environment


def trim_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_CHARS_PER_POST:
        return text
    return text[:MAX_CHARS_PER_POST].rstrip() + "…"


def call_with_retries(fn, max_retries: int = MAX_RETRIES, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def _to_float_clamped(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def validate_and_normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    # ---- topic ----
    topic = result.get("topic")
    if topic not in TOPIC_NAME_TO_ID:
        topic = OTHER_TOPIC

    subtopic = result.get("subtopic", None)
    if subtopic is not None and not isinstance(subtopic, str):
        subtopic = str(subtopic)
    subtopic = subtopic.strip() if isinstance(subtopic, str) else None
    if not subtopic:
        subtopic = None

    confidence = _to_float_clamped(result.get("confidence", 0.0))

    rationale = result.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    rationale = rationale.strip()
    if len(rationale) > 120:
        rationale = rationale[:120].rstrip() + "…"

    # ---- sentiment ----
    new_sent = result.get("newSentiment")
    if new_sent not in ALLOWED_SENTIMENT:
        new_sent = "Unclear"

    new_sent_conf = _to_float_clamped(result.get("newSentimentConfidence", 0.0))

    # If topic is Other/Unclear, nudge sentiment confidence down if it's oddly high
    if topic == OTHER_TOPIC and new_sent_conf > 0.7:
        new_sent_conf = 0.7

    return {
        "topic_id": TOPIC_NAME_TO_ID[topic],
        "topic": topic,
        "subtopic": subtopic,
        "confidence": confidence,
        "rationale": rationale,
        "newSentiment": new_sent,
        "newSentimentConfidence": new_sent_conf,
    }


def analyze_message(message: str) -> Dict[str, Any]:
    message = trim_text(message)

    def _do_call():
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown. No explanation."
                },
                {"role": "user", "content": message},
            ],
        )

        raw_text = (resp.output_text or "").strip()

        # Defensive JSON extraction (handles rare leading/trailing text)
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            raw = json.loads(raw_text[start : end + 1])

        return validate_and_normalize(raw)

    return call_with_retries(_do_call)


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Put it in the same folder or update INPUT_CSV."
        )

    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"CSV must contain '{TEXT_COL}'. Found columns: {list(df.columns)}")

    df = df.reset_index(drop=True)
    df["RowId"] = df.index + 1

    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df["__msg_clean"] = df[TEXT_COL].str.strip()
    df = df[df["__msg_clean"].str.len() > 0].copy()

    total = len(df)
    results = []

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        processed = 0
        for _, row in df.iterrows():
            processed += 1
            row_id = int(row["RowId"])
            msg = row["__msg_clean"]

            meta = {"RowId": row_id}
            for c in META_COLS:
                if c in df.columns and pd.notna(row.get(c)):
                    meta[c] = row.get(c)

            try:
                analysis = analyze_message(msg)
                out_row = {**meta, "Message": row[TEXT_COL], **analysis}
                print(
                    f"[{processed}/{total}] RowId={row_id} "
                    f"msg={msg}"
                    f"topic={analysis['topic']} ({analysis['confidence']:.2f}) "
                    f"newSentiment={analysis['newSentiment']} ({analysis['newSentimentConfidence']:.2f})"
                )
            except Exception as e:
                out_row = {**meta, "Message": row[TEXT_COL], "error": str(e)}
                print(f"[{processed}/{total}] RowId={row_id} ERROR: {e}")

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            results.append(out_row)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    pd.DataFrame(TOPICS).to_csv("topic_lookup.csv", index=False, encoding="utf-8-sig")

    print(f"\nDone. Wrote {OUTPUT_JSONL}, {OUTPUT_CSV}, and topic_lookup.csv")


if __name__ == "__main__":
    main()
