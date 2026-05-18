import pandas as pd
import json
import os
import sys
import asyncio
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from pypdf import PdfReader

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

# Add config directory to path for actions import
sys.path.insert(0, str(CONFIG_DIR))

# Import actions FIRST
import actions

from nemoguardrails import LLMRails, RailsConfig


def _ensure_event_loop():
    """Ensure the current thread has an active asyncio event loop."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def init():
    """Initialise guardrails, OpenAI client, and RAG embeddings. Returns (rails, client, chunks, embeddings)."""
    load_dotenv(BASE_DIR / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key

    config = RailsConfig.from_path(str(CONFIG_DIR))
    _ensure_event_loop()
    rails = LLMRails(config)
    rails.register_action(actions.mask_pii, name="mask_pii")
    rails.register_action(actions.remove_sensitive_org_data, name="remove_sensitive_org_data")
    rails.register_action(actions.detect_prompt_injection, name="detect_prompt_injection")
    rails.register_action(actions.detect_toxicity, name="detect_toxicity")

    client = OpenAI(api_key=api_key)

    pdf_reader = PdfReader(str(DATA_DIR / "ITSM_Knowledge_Base.pdf"))
    doc = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    chunks = [doc[i:i+500] for i in range(0, len(doc), 500)]

    embeddings = []
    for chunk in chunks:
        resp = client.embeddings.create(input=chunk, model="text-embedding-3-small")
        embeddings.append(resp.data[0].embedding)

    return rails, client, chunks, embeddings


def load_tickets():
    """Load tickets from Excel and return a DataFrame with timestamps as strings."""
    df = pd.read_excel(str(DATA_DIR / "ITSM_Tickets.xlsx"))
    for record in df.to_dict(orient="records"):
        pass  # validation placeholder
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]):
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _extract_priority_label(text):
    """Extract and normalize a priority label from model output."""
    if not text:
        return None
    match = re.search(r"\bP\s*([1-4])\b", str(text), flags=re.IGNORECASE)
    if not match:
        return None
    level = match.group(1)
    labels = {
        "1": "P1 - Critical",
        "2": "P2 - High",
        "3": "P3 - Medium",
        "4": "P4 - Low",
    }
    return labels.get(level)


def prioritize_ticket(ticket_dict, rails):
    """Use LLM to classify ticket priority as one of P1/P2/P3/P4."""
    title = ticket_dict.get("Title", "")
    description = ticket_dict.get("Description", "")
    user_message = json.dumps({"Title": title, "Description": description}, ensure_ascii=False)

    system_message = (
        "You are an ITSM triage assistant. "
        "Classify ticket priority using business impact and urgency. "
        "Respond with exactly one label from this list only: "
        "P1 - Critical, P2 - High, P3 - Medium, P4 - Low."
    )
    response = rails.generate(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ])

    raw_text = response.get("content", "")
    parsed = _extract_priority_label(raw_text)
    if parsed is not None:
        return parsed

    # Safe default when output is malformed.
    return "P3 - Medium"


def analyse_ticket(ticket_dict, rails, client, chunks, embeddings):
    """Run RAG + guardrails analysis on a ticket dict and return the response string."""
    ticket_query = f"{ticket_dict.get('Title', '')}. {ticket_dict.get('Description', '')}"
    ticket_text = json.dumps([ticket_dict], indent=2, ensure_ascii=False)

    q_emb = client.embeddings.create(input=ticket_query, model="text-embedding-3-small").data[0].embedding
    sims = [cosine_similarity(q_emb, emb) for emb in embeddings]
    top_idx = np.argsort(sims)[-2:][::-1]
    top_chunks = [chunks[i] for i in top_idx]

    system_message = "Reference Knowledge:\n" + "\n".join(top_chunks)
    response = rails.generate(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": ticket_text},
    ])
    return response["content"]


if __name__ == "__main__":
    rails, client, chunks, embeddings = init()
    df = load_tickets()
    data = df.to_dict(orient="records")

    for ticket in data[:5]:
        result = analyse_ticket(ticket, rails, client, chunks, embeddings)
        print(result)