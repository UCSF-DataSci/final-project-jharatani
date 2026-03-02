# --- Part 3: Building a Helpful Medical Agent ---

#imports

import re
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import json
import asyncio
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from agents import (
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem,
    input_guardrail, output_guardrail
)

load_dotenv() 

if os.environ.get("LOCAL_LLM") == "1":
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    client = AsyncOpenAI(base_url=base, api_key="ollama")
    MODEL = OpenAIChatCompletionsModel(model="llama3.2:1b", openai_client=client)
    MODEL_NAME = "local"
    set_tracing_disabled(True)
elif os.environ.get("OPENROUTER_API_KEY"):
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = OpenAIChatCompletionsModel(model="openai/gpt-4o-mini", openai_client=client)
    MODEL_NAME = "openai/gpt-4o-mini"
    set_tracing_disabled(True)
elif os.environ.get("OPENAI_API_KEY"):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)
    MODEL_NAME = "gpt-4o-mini"
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

SETTINGS = ModelSettings(temperature=0, max_tokens=1024)
print(f"Using model: {MODEL_NAME}")

# Load em up!

def get_device():
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"

def load_corpus(path="outputs/extractions.jsonl"):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["diagnosis"] = df["extraction"].apply(
        lambda x: x.get("diagnosis", "") if isinstance(x, dict) else ""
    )
    return df[["cui", "diagnosis", "note", "extraction"]].reset_index(drop=True)

print("Loading corpus and embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
corpus = load_corpus()
embeddings = embed_model.encode(corpus["note"].tolist(), show_progress_bar=True)
print(f"Ready — {len(corpus)} notes loaded")

# Let's get some definitions and functions and tools

# PHI Guardrails - though there is no PHI in my dataset, I want to be able to use these guardrails on future datasets with PHI

PHI_PATTERNS = {
    "ssn":   r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "mrn":   r"\b(MRN|Medical Record)[\s:#]*\d+\b",
}

@input_guardrail
async def phi_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Block input containing PHI before it reaches the LLM."""
    text = input if isinstance(input, str) else str(input)
    found = {k: re.findall(p, text, re.IGNORECASE) for k, p in PHI_PATTERNS.items()}
    found = {k: v for k, v in found.items() if v}
    return GuardrailFunctionOutput(
        output_info={"phi_found": found},
        tripwire_triggered=bool(found),
    )

# I also want to check for hallucinations here, using demo framework (Lecture 8)

from pydantic import BaseModel

class HallucinationCheck(BaseModel):
    hallucination_detected: bool
    reasoning: str

hallucination_checker = Agent(
    name="Hallucination Checker",
    model=MODEL,
    model_settings=SETTINGS,
    instructions=(
        "You receive a triage result and the original clinical note. "
        "Check if the triage result mentions any diagnosis or medication "
        "NOT present in the original note. "
        "Set hallucination_detected=True if anything was fabricated."
    ),
    output_type=HallucinationCheck,
)

EMERGENCY_DIAG_KEYWORDS = [
    "myocardial infarction", "heart attack", "stroke", "sepsis",
    "pulmonary embol", "respiratory failure", "acute kidney injury",
    "intracranial hemorrhage", "aortic aneurysm rupture",
]

URGENT_DIAG_KEYWORDS = [
    "pneumonia", "dka", "diabetic ketoacidosis", "neutropenic fever",
    "appendicitis", "gi bleed", "acute coronary syndrome",
]

SPECIALTY_DIAG_KEYWORDS = [
    "cancer", "carcinoma", "neuroendocrine tumor", "myeloma",
    "leukemia", "lymphoma", "cholangiocarcinoma",
]

def parse_numeric(s):
    """Extract a number from a string like '100,000' or '1.4'."""
    if s is None:
        return None
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", str(s))
    if not m:
        return None
    return float(m.group(0).replace(",", ""))

def triage_from_extraction(extraction: dict) -> dict:
    """
    Deterministic triage label based on extraction fields.
    Returns dict with triage_level + red_flags + confidence.
    """
    diagnosis = (extraction.get("diagnosis") or "").lower()
    meds = [m.lower() for m in (extraction.get("medications") or [])]
    labs = extraction.get("lab_values") or {}
    red_flags = []

    # --- Keyword rules ---
    if any(k in diagnosis for k in EMERGENCY_DIAG_KEYWORDS):
        red_flags.append("diagnosis_keyword_emergency")
        level = "emergency"
    elif any(k in diagnosis for k in URGENT_DIAG_KEYWORDS):
        red_flags.append("diagnosis_keyword_urgent")
        level = "urgent"
    elif any(k in diagnosis for k in SPECIALTY_DIAG_KEYWORDS):
        red_flags.append("diagnosis_keyword_specialty")
        level = "specialty"
    else:
        level = "routine"

    # --- Lab red flags (very coarse) ---
    # WBC extremely high (example)
    wbc_val = None
    for k, v in labs.items():
        if str(k).strip().lower() in {"wbc", "white blood cell count"}:
            wbc_val = parse_numeric(v)
            break
    if wbc_val is not None and wbc_val >= 50_000:
        red_flags.append(f"wbc_extremely_high:{wbc_val}")
        level = "urgent" if level == "routine" else level

    # Creatinine high (very coarse)
    cr_val = None
    for k, v in labs.items():
        if str(k).strip().lower() in {"creatinine", "cr"}:
            cr_val = parse_numeric(v)
            break
    if cr_val is not None and cr_val >= 2.0:
        red_flags.append(f"creatinine_high:{cr_val}")
        level = "urgent" if level == "routine" else level

    # If extraction is empty → insufficient_info
    empty = (diagnosis.strip() == "" and len(meds) == 0 and len(labs) == 0)
    if empty:
        level = "insufficient_info"
        red_flags.append("no_explicit_clinical_facts_extracted")

    # Confidence: combine extraction confidence with how strong rules are
    base_conf = float(extraction.get("confidence") or 0.0)
    rule_bonus = 0.2 if level in {"emergency", "urgent", "specialty"} else 0.0
    triage_conf = max(0.0, min(1.0, base_conf + rule_bonus))

    return {
        "triage_level": level,
        "red_flags": red_flags,
        "triage_confidence": triage_conf,
    }

def triage_agent(record: dict) -> dict:
    """
    record should look like your Part 1 JSONL line:
    {cui, mention, note, extraction}
    """
    extraction = record.get("extraction") or {}
    triage = triage_from_extraction(extraction)

    diagnosis = extraction.get("diagnosis", "")
    meds = extraction.get("medications", [])
    labs = extraction.get("lab_values", {})
    note_preview = (record.get("note") or "")[:240]

    # Non-prescriptive next step text
    if triage["triage_level"] == "emergency":
        next_step = "High-risk pattern detected. Seek immediate evaluation (emergency services/ER) if this were a real case."
    elif triage["triage_level"] == "urgent":
        next_step = "Potentially urgent pattern detected. Prompt clinical evaluation would be appropriate in a real setting."
    elif triage["triage_level"] == "specialty":
        next_step = "Specialty follow-up indicated (e.g., oncology/hematology/cardiology), depending on context."
    elif triage["triage_level"] == "routine":
        next_step = "No high-risk triggers found in extracted text; routine follow-up would likely be considered in a real setting."
    else:
        next_step = "Insufficient explicit information in the snippet to triage reliably."

    return {
        "cui": record.get("cui", ""),
        "diagnosis": diagnosis,
        "triage_level": triage["triage_level"],
        "recommended_next_step": next_step,
        "red_flags": triage["red_flags"],
        "evidence": {
            "extracted": {"diagnosis": diagnosis, "medications": meds, "lab_values": labs},
            "note_preview": note_preview,
        },
        "confidence": triage["triage_confidence"],
        "disclaimer": "Educational prototype on synthetic/fragmented notes. Not medical advice; do not use for real clinical decisions.",
    }

@output_guardrail
async def hallucination_guardrail(
    ctx: RunContextWrapper, agent: Agent, output
) -> GuardrailFunctionOutput:
    context = ctx.context or {}
    original_note = context.get("original_note", "")
    check_input = f"Original note:\n{original_note}\n\nTriage result:\n{json.dumps(output)}"
    result = await Runner.run(hallucination_checker, check_input)
    check = result.final_output
    return GuardrailFunctionOutput(
        output_info={"reasoning": check.reasoning},
        tripwire_triggered=check.hallucination_detected,
    )

class TriageResult(BaseModel):
    cui: str
    diagnosis: str
    triage_level: str  # "emergency" | "urgent" | "specialty" | "routine" | "insufficient_info"
    recommended_next_step: str
    red_flags: list[str]
    confidence: float
    disclaimer: str

@function_tool
def semantic_search(query: str, top_k: int = 3) -> str:
    """Search the clinical notes corpus semantically and return the most relevant matches."""
    q_emb = embed_model.encode([query])
    sims = cosine_similarity(q_emb, np.asarray(embeddings))[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = corpus.iloc[idx]
        results.append({
            "cui": str(row["cui"]),
            "diagnosis": str(row["diagnosis"]),
            "score": round(float(sims[idx]), 3),
            "note_preview": str(row["note"])[:200],
        })
    return json.dumps(results, indent=2)

@function_tool
def triage_record(cui: str) -> str:
    """Run triage on a specific record by CUI and return triage level + next steps."""
    match = corpus[corpus["cui"] == cui]
    if match.empty:
        return json.dumps({"error": f"No record found for CUI: {cui}"})
    
    row = match.iloc[0].to_dict()
    result = triage_agent(row)
    return json.dumps(result, indent=2)

# --- Build the agent ---

medical_agent = Agent(
    name="MedicalTriageAgent",
    instructions="""You are a medical triage assistant working with synthetic clinical notes.
When given a query:
1. Use semantic_search to find relevant notes
2. Use triage_record on top results
3. Summarize: triage level, red flags, next steps
Always append the disclaimer that this is educational only.""",
    tools=[semantic_search, triage_record],
    model=MODEL,
    model_settings=SETTINGS,
    input_guardrails=[phi_guardrail],
    output_guardrails=[hallucination_guardrail],
)

# --- Run it ---

async def run_agent(query: str):
    print(f"\nQuery: {query}\n{'='*50}")
    result = await Runner.run(
        medical_agent, 
        query, 
        context={"original_note": query})
    print(result.final_output)

    output_record = {
        "query": query,
        "response": result.final_output,
    }

    json_path = "outputs/agent_outputs.json"
    existing = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)
    existing.append(output_record)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)

    # TXT (human-readable)
    with open("outputs/agent_outputs.txt", "a") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Response:\n{result.final_output}\n")
        f.write("="*50 + "\n\n")

    print(f"Saved to outputs/")

# Testing some example queries with my agent!

if __name__ == "__main__":
    asyncio.run(run_agent("patient with heart attack symptoms and high WBC"))


