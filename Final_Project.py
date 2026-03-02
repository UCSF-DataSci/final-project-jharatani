# Final Project

# Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

# Dataset: data/SYNTHETIC_MENTIONS.csv

# Setup

import os
import json
import random
import numpy as np
from dotenv import load_dotenv
import re
import pandas as pd

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
load_dotenv()

def get_client():
    from openai import OpenAI

    # Local Ollama
    if os.environ.get("LOCAL_LLM") == "1":
        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        return OpenAI(base_url=base, api_key="ollama"), "local"

    # OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        ), "openrouter"

    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "openai"

    raise ValueError("No API key found. Set LOCAL_LLM=1 (for Ollama) or set OPENROUTER_API_KEY / OPENAI_API_KEY.")

def call_llm(prompt, provider, client):
    """Send a prompt to the LLM and return the response text."""
    if provider == "local":
        model = "llama3.2:1b"
    elif provider == "openrouter":
        model = "openai/gpt-4o-mini"
    else:
        model = "gpt-4o-mini"

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No markdown, no commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    # Strict JSON mode for OpenAI (works well)
    if provider == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def get_device():
    """Detect the best available device for local model inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

# Load Data

synthetic = pd.read_csv("data/SYNTHETIC_MENTIONS.csv")

print(f"Loaded {len(synthetic)} synthetic clinical notes")
print(f"Columns: {list(synthetic.columns)}")

print(synthetic.iloc[0])

def extract_best_mention(text):
    mentions = re.findall(r"<1CUI>\s*(.*?)\s*</1CUI>", str(text), flags=re.DOTALL)
    mentions = [m.strip() for m in mentions if m and m.strip()]

    if not mentions:
        return None

    # keep "word-like" mentions (letters, length>=4)
    candidates = [m for m in mentions if re.search(r"[A-Za-z]", m) and len(m) >= 4]

    # pick the longest candidate (more likely meaningful)
    return max(candidates or mentions, key=len)

def clean_text(text):
    text = re.sub(r"</?1CUI>", " ", str(text))
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

synthetic["mention"] = synthetic["matched_output"].apply(extract_best_mention)
synthetic["note"] = synthetic["matched_output"].apply(clean_text)

# (optional) keep only rows that actually have a tagged mention
synth_use = synthetic[synthetic["mention"].notna()].copy()

random.seed(2026)
sample_df = synth_use.sample(n=4, random_state=2026)  # better than random.sample for DataFrames

notes_p1 = sample_df["note"].tolist()

print(f"Selected {len(notes_p1)} notes for extraction")
for i, n in enumerate(notes_p1, 1):
    print(f"\n--- Note {i} ({len(n)} chars) ---")
    print(n[:150] + "...")


def build_prompt(note, few_shot=False):
    prompt = """Extract structured clinical information from the clinical note snippet below.

Return ONLY a JSON object with exactly these fields:

{
  "diagnosis": "<primary diagnosis or main clinical problem (string)>",
  "medications": ["<medication 1>", "<medication 2>", "..."],
  "lab_values": {"<lab name>": "<value with units if present>", "..."},
  "confidence": <float between 0 and 1>
}

Rules:
- Only extract information explicitly stated in the snippet. Do NOT infer or guess.
- If a field is not present, use: diagnosis="" , medications=[], lab_values={}
- Ignore de-identification placeholders like [** ... **].
- <1CUI>...</1CUI> marks a highlighted mention; treat it as normal text (do not assume it is the diagnosis).
- Confidence reflects how complete/clear the snippet is (low if sparse/fragmented).
"""

    if few_shot:
        prompt += """

Example 1:
Snippet: "3. ovarian cysts. 4. tubal epithelial hyperplasia. 5. endometrial polyp. 6. uterine fibroids."
Output:
{
  "diagnosis": "Uterine fibroids",
  "medications": [],
  "lab_values": {},
  "confidence": 0.55
}

Example 2:
Snippet: "sig: one tablet (300 mg) po qd for 7 days. allergies: penicillins / sulfa."
Output:
{
  "diagnosis": "",
  "medications": ["(unspecified medication) 300 mg PO daily x7 days"],
  "lab_values": {},
  "confidence": 0.45
}

Example 3:
Snippet: "white blood cell count of 100,000 on admission, decreased to 70,000 by discharge."
Output:
{
  "diagnosis": "",
  "medications": [],
  "lab_values": {"WBC": "100,000 on admission; 70,000 by discharge"},
  "confidence": 0.7
}
"""

    prompt += f"\nClinical note snippet:\n{note}\n"
    return prompt

def parse_json_response(text):
    """Try to extract a JSON object from LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("```")
        for block in lines[1::2]:  # odd-indexed segments are inside fences
            block = block.strip().removeprefix("json").strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Find outermost braces
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None

def validate_response(response):
    """
    Check that the parsed response contains all required fields.
    Returns True if all present, False otherwise
    """

    parsed = parse_json_response(response)
    if parsed is None:
        return False
    
    required_fields = ["diagnosis", "medications", "lab_values", "confidence"]

    return all(field in parsed for field in required_fields)

def safe_json_loads(s):
    """
    Try to parse JSON even if the model accidentally adds extra text.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to extract the first JSON object in the string
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

client, provider = get_client()

import requests
print(requests.get("http://127.0.0.1:11434/api/tags", timeout=5).json())

def extract_entities(note, client, provider, few_shot=False):

    prompt = build_prompt(note, few_shot=few_shot)
    raw = call_llm(prompt, provider=provider, client=client)
    parsed = parse_json_response(raw)

    if parsed is None or not validate_response(raw):
        return None
    
    return parsed

results_p1 = []
for i, note in enumerate(notes_p1, 1):
    result = extract_entities(note, client, provider, few_shot=True)
    print(f"--- Note {i} ---")
    if result:
        print(json.dumps(result, indent=2))
        results_p1.append(result)
    else:
        print("Extraction failed")
    print()

# ---- Part 1: run extraction on ~50 notes ----

synth_use = synth_use.reset_index(drop=True)

N_P1 = 50
df_p1 = synth_use.sample(n=min(N_P1, len(synth_use)), random_state=2026).reset_index(drop=True)

out_path = "outputs/extractions.jsonl"

with open(out_path, "w", encoding="utf-8") as f:
    for i, row in df_p1.iterrows():
        note_text = row["note"]
        mention = row["mention"]
        cui = row["cui"]

        prompt = build_prompt(note_text, few_shot=True)
        resp_text = call_llm(prompt, provider, client)

        # Try to parse JSON
        try:
            extraction = json.loads(resp_text)
        except:
            extraction = {
                "diagnosis": "",
                "medications": [],
                "lab_values": {},
                "confidence": 0.0,
                "_raw_response": resp_text
            }

        record = {
            "cui": str(cui),
            "mention": "" if pd.isna(mention) else str(mention),
            "note": str(note_text),
            "extraction": extraction,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(df_p1)}")

print(f"\nWrote: {out_path}")

pretty_path = "outputs/extractions_pretty.txt"

with open(out_path, "w", encoding="utf-8") as f_jsonl, \
     open(pretty_path, "w", encoding="utf-8") as f_pretty:

    for idx, row in df_p1.iterrows():
        note_text = row["note"]
        mention = row["mention"]
        cui = row["cui"]

        prompt = build_prompt(note_text, few_shot=True)
        resp_text = call_llm(prompt, provider, client)

        parsed = parse_json_response(resp_text)

        if parsed is None or not validate_response(resp_text):
            extraction = {
                "diagnosis": "",
                "medications": [],
                "lab_values": {},
                "confidence": 0.0,
                "_raw_response": resp_text[:2000],
            }
        else:
            extraction = parsed

        record = {
            "cui": str(cui),
            "mention": "" if pd.isna(mention) else str(mention),
            "note": str(note_text),
            "extraction": extraction,
        }

        # ---- Write machine-readable JSONL ----
        f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ---- Write pretty human-readable format ----
        f_pretty.write(f"--- Note {idx+1} ---\n")
        f_pretty.write(json.dumps(extraction, indent=2))
        f_pretty.write("\n\n")

print(f"Wrote JSONL: {out_path}")
print(f"Wrote pretty file: {pretty_path}")

#---- Part 2: Semantics Search/Embeddings ----

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Build the search corpus from your cleaned notes
# Keep metadata so results are interpretable

import json

records = []

with open("outputs/extractions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df_extract = pd.DataFrame(records)

print(df_extract.head())

# Extracting diagnosis from json file created in Pt. 1
df_extract["diagnosis"] = df_extract["extraction"].apply(
    lambda x: x.get("diagnosis", "") if isinstance(x, dict) else ""
)

corpus = df_extract[["cui", "diagnosis", "note"]].reset_index(drop=True)

N_P2 = 2000  # start small
corpus = corpus.sample(n=min(N_P2, len(corpus)), random_state=2026).reset_index(drop=True)

notes_p2 = corpus["note"].tolist()
print(f"{len(notes_p2)} notes in search corpus")

# Load embedding model (runs locally)
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
print(f"Embedding model loaded on {get_device()}")


def embed_notes(notes):
    """
    Encode a list of notes into embeddings.
    Returns: numpy array (n_notes, embedding_dim)
    """
    return embed_model.encode(notes, show_progress_bar=True)


def find_similar(query, corpus_df, embeddings, top_k=5):
    """
    Returns top_k results with CUI + mention + note snippet + score
    """
    query_embedding = embed_model.encode([query])  # shape (1, dim)

    embeddings = np.asarray(embeddings)
    sims = cosine_similarity(query_embedding, embeddings)[0]  # shape (n_notes,)

    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = corpus_df.iloc[idx]
        results.append({
            "cui": str(row["cui"]),
            "diagnosis": "" if pd.isna(row["diagnosis"]) else str(row["diagnosis"]),
            "score": float(sims[idx]),
            "note_preview": str(row["note"])[:200],
        })
    return results


# ---- Run Part 2 pipeline ----
embeddings = embed_notes(notes_p2)
print(f"Embeddings shape: {embeddings.shape}")

queries = [
    "heart attack symptoms",
    "infectious disease with fever",
    "respiratory illness",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = find_similar(q, corpus, embeddings, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. (score: {r['score']:.3f}) CUI={r['cui']} diagnosis={r['diagnosis']}")
        print(f"     {r['note_preview']}...")

# Save one example output (like the assignment)
os.makedirs("outputs", exist_ok=True)
search_results = find_similar("heart attack symptoms", corpus, embeddings, top_k=5)
with open("outputs/search_results.json", "w", encoding="utf-8") as f:
    json.dump(search_results, f, indent=2)

print(f"Saved {len(search_results)} search results to outputs/search_results.json")