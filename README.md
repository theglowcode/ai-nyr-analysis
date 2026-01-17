# Social Listening Topic & Sentiment Classifier (GPT-5 Mini)

This repository contains a Python-based pipeline for analyzing social media messages using OpenAI’s GPT-5 Mini model.  
It classifies each message into a **locked topic taxonomy** and generates a **new, model-derived sentiment label**, designed for downstream analytics and dashboards (e.g. Power BI, Tableau).

The system is intended for **research, prototyping, and analytics workflows**, not real-time moderation.

---

## Features

- Reads social media data from CSV
- Processes messages **row by row**
- Locked, analyst-defined **topic taxonomy**
- Model-generated `newSentiment` (vendor sentiment ignored)
- Deterministic, schema-enforced JSON output
- Outputs:
  - `topics.jsonl` (stream-friendly)
  - `topics.csv` (BI-friendly)
  - `topic_lookup.csv` (dimension table)
- Built for **GPT-5 Mini** using the OpenAI Responses API

---

## Example Use Cases

- Social listening & trend analysis  
- New Year resolution / aspiration analysis  
- Market research & consumer insight  
- Sentiment benchmarking against vendor tools  
- Academic or exploratory NLP work  

---

## Requirements

- Python 3.9+
- OpenAI Python SDK
- pandas

Install dependencies:

```bash
pip install openai pandas
