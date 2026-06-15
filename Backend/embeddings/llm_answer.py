#!/usr/bin/env python3
"""
LLM Answer Generation — Production-Grade Multi-Provider
Generates clean, well-structured answers from textbook excerpts.

Provider Priority:
  1. Groq          — llama-3.3-70b-versatile  (~500 tok/s, ultra-low latency)
  2. Gemini         — gemini-2.0-flash         (1M context, free tier)
  3. Together.ai    — Llama-3.3-70B-Instruct-Turbo
  4. OpenRouter     — mistral-7b-instruct:free

Features:
- Exponential backoff retry logic
- Structured output format with markdown
- Token-efficient prompt design
- Full provider fallback chain

Usage (import):
    from llm_answer import generate_answer
"""

import sys
import json
import re
import time
import logging
import requests
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------
GROQ_API_KEY      = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY    = os.getenv('GEMINI_API_KEY')
TOGETHER_API_KEY  = os.getenv('TOGETHER_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Production Grade
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are LearnLens, an expert academic tutor specialising in precise, thorough textbook explanations.
Answer the student's question using ONLY the textbook excerpts provided.

CORE RULES:
1. Base your answer EXCLUSIVELY on the provided excerpts — do NOT use external knowledge
2. Be EXHAUSTIVE and DETAILED — the student needs a complete understanding, not a summary
3. Use correct academic language and define every technical term on first use (bold it)
4. Synthesise information across ALL provided excerpts into a coherent, flowing answer
5. Explain the 'why' and 'how', not just the 'what'
6. If the excerpts don't fully answer the question, say so explicitly and answer from what IS available

REQUIRED OUTPUT STRUCTURE:
1. **Opening paragraph** — 2-3 sentences giving the direct, complete answer
2. `## ` section headers for each major sub-topic (use as many as needed)
3. **Bold** every key term the first time it appears
4. Bullet points only for enumeration/lists (not for narrative prose)
5. Include concrete examples from the excerpts wherever possible
6. End EVERY answer with a `### Key Takeaway` paragraph summarising the most important insight

CITATIONS: Cite every factual claim with [Source N] matching the excerpt number provided.
LENGTH: Write as much as the excerpts support — do NOT truncate or summarise prematurely."""


def build_prompt(chunks: List[str], user_query: str) -> str:
    """Build a standardized prompt from chunks and query (for single-turn APIs like Gemini)."""
    excerpts = '\n\n'.join([
        f"[Source {i+1}]\n{chunk[:2000]}"
        for i, chunk in enumerate(chunks)
    ])
    return f"""{SYSTEM_PROMPT}

---
TEXTBOOK EXCERPTS:
{excerpts}

---
STUDENT'S QUESTION: {user_query}

YOUR ANSWER:"""


# ---------------------------------------------------------------------------
# RETRY DECORATOR
# ---------------------------------------------------------------------------

def with_retry(fn, max_retries: int = 2, base_delay: float = 1.0):
    """Call fn with exponential backoff retry on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Timeout on attempt {attempt+1}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 429:
                logger.warning("Rate limited (429), retrying in 5s...")
                time.sleep(5)
                continue
            if status in (401, 403):
                raise  # Auth failures — don't retry
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


# ---------------------------------------------------------------------------
# ① GROQ — Primary Provider (Ultra-fast, ~500 tok/s)
# ---------------------------------------------------------------------------

def call_groq(chunks: List[str], user_query: str) -> str:
    """
    Call Groq API with llama-3.3-70b-versatile.
    Groq's LPU inference delivers ~500 tokens/second — fastest available.
    Models available: llama-3.3-70b-versatile, llama3-70b-8192, mixtral-8x7b-32768
    """
    if not GROQ_API_KEY or GROQ_API_KEY == 'your_groq_api_key_here':
        raise Exception("GROQ_API_KEY not configured")

    logger.info(f"[Groq] Calling llama-3.3-70b-versatile with {len(chunks)} chunks")

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }

    user_content = (
        "TEXTBOOK EXCERPTS:\n" +
        "\n\n".join([f"[Source {i+1}]\n{c[:2000]}" for i, c in enumerate(chunks)]) +
        f"\n\nSTUDENT'S QUESTION: {user_query}"
    )

    payload = {
        'model': 'llama-3.3-70b-versatile',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_content},
        ],
        'max_tokens': 4096,
        'temperature': 0.10,
        'top_p': 0.92,
        'stream': False,
    }

    def _call():
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers, json=payload, timeout=30  # Groq is fast — 30s is generous
        )
        resp.raise_for_status()
        return resp

    response = with_retry(_call)
    data = response.json()
    answer = data['choices'][0]['message']['content'].strip()
    tokens_used = data.get('usage', {}).get('completion_tokens', '?')
    logger.info(f"[Groq] ✓ {len(answer)} chars | {tokens_used} tokens")
    return clean_answer(answer)


# ---------------------------------------------------------------------------
# ② GEMINI — Fallback 1 (gemini-2.0-flash, 1M context)
# ---------------------------------------------------------------------------

def call_gemini(chunks: List[str], user_query: str) -> str:
    """Call Google Gemini 2.0 Flash — fast, high-quality, 1M context window."""
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not set")

    logger.info(f"[Gemini] Calling gemini-2.0-flash with {len(chunks)} chunks")
    prompt = build_prompt(chunks, user_query)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "topP": 0.92,
            "maxOutputTokens": 2048,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )

    def _call():
        resp = requests.post(url, json=payload, timeout=35)
        resp.raise_for_status()
        return resp

    response = with_retry(_call)
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise Exception("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise Exception("Gemini returned empty content")

    answer = parts[0].get("text", "").strip()
    if not answer:
        raise Exception("Gemini returned empty text")

    logger.info(f"[Gemini] ✓ {len(answer)} chars")
    return clean_answer(answer)


# ---------------------------------------------------------------------------
# ③ TOGETHER.AI — Fallback 2 (Llama-3.3-70B-Instruct-Turbo)
# ---------------------------------------------------------------------------

def call_together_ai(chunks: List[str], user_query: str) -> str:
    """Call Together.ai with Llama 3.3 70B Instruct Turbo."""
    if not TOGETHER_API_KEY:
        raise Exception("TOGETHER_API_KEY not set")

    logger.info(f"[Together.ai] Calling Llama-3.3-70B-Instruct-Turbo with {len(chunks)} chunks")

    headers = {
        'Authorization': f'Bearer {TOGETHER_API_KEY}',
        'Content-Type': 'application/json'
    }

    user_content = (
        "TEXTBOOK EXCERPTS:\n" +
        "\n\n".join([f"[Source {i+1}]\n{c[:2000]}" for i, c in enumerate(chunks)]) +
        f"\n\nQUESTION: {user_query}"
    )

    payload = {
        'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_content},
        ],
        'max_tokens': 4096,
        'temperature': 0.10,
        'top_p': 0.92,
        'repetition_penalty': 1.05,
        'stop': ['<|eot_id|>', '<|end_of_text|>']
    }

    def _call():
        resp = requests.post(
            'https://api.together.xyz/v1/chat/completions',
            headers=headers, json=payload, timeout=90
        )
        resp.raise_for_status()
        return resp

    response = with_retry(_call)
    data = response.json()
    answer = data['choices'][0]['message']['content'].strip()
    logger.info(f"[Together.ai] ✓ {len(answer)} chars")
    return clean_answer(answer)


# ---------------------------------------------------------------------------
# ④ OPENROUTER — Fallback 3 (Mistral 7B, free tier)
# ---------------------------------------------------------------------------

def call_openrouter(chunks: List[str], user_query: str) -> str:
    """Call OpenRouter with Mistral 7B Instruct (free tier)."""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY not set")

    logger.info(f"[OpenRouter] Calling mistral-7b-instruct with {len(chunks)} chunks")
    prompt = build_prompt(chunks, user_query)

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://learnlens.app',
        'X-Title': 'LearnLens Textbook Assistant'
    }

    payload = {
        'model': 'mistralai/mistral-7b-instruct:free',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 2048,
        'temperature': 0.15,
        'top_p': 0.92,
    }

    def _call():
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers, json=payload, timeout=90
        )
        resp.raise_for_status()
        return resp

    response = with_retry(_call)
    data = response.json()
    answer = data['choices'][0]['message']['content'].strip()
    logger.info(f"[OpenRouter] ✓ {len(answer)} chars")
    return clean_answer(answer)


# ---------------------------------------------------------------------------
# ANSWER CLEANING
# ---------------------------------------------------------------------------

def clean_answer(answer: str) -> str:
    """Clean and normalize LLM output for consistent markdown rendering."""
    answer = answer.strip()

    # Remove LLM preamble artifacts
    prefixes_to_remove = [
        "Answer:", "ANSWER:", "Response:", "RESPONSE:",
        "Here's the answer:", "Here is the answer:",
        "Based on the excerpts:", "According to the excerpts:",
        "YOUR ANSWER:", "Here's my answer:",
        "Based on the provided", "Based on the textbook",
        "According to the textbook",
    ]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Remove excessive blank lines (max 2 consecutive)
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    # Clean trailing whitespace per line
    lines = [line.rstrip() for line in answer.split('\n')]
    answer = '\n'.join(lines)

    # Fix spacing after sentence-ending punctuation
    answer = re.sub(r'([.!?])([A-Z])', r'\1 \2', answer)

    # Remove redundant spaces
    answer = re.sub(r' {2,}', ' ', answer)

    return answer.strip()


# ---------------------------------------------------------------------------
# MULTI-PROVIDER ORCHESTRATOR
# ---------------------------------------------------------------------------

def generate_answer(chunks: List[str], user_query: str) -> dict:
    """
    Generate LLM answer using priority chain:
      Groq → Gemini → Together.ai → OpenRouter

    Returns dict with keys:
      answer, api_used, model, error
    """
    chunks = chunks[:8]  # Pass up to 8 chunks for richer context

    providers = [
        ("groq",       "llama-3.3-70b-versatile",                call_groq),
        ("gemini",     "gemini-2.0-flash",                        call_gemini),
        ("together.ai","meta-llama/Llama-3.3-70B-Instruct-Turbo", call_together_ai),
        ("openrouter", "mistralai/mistral-7b-instruct",           call_openrouter),
    ]

    errors = []
    for api_name, model_name, call_fn in providers:
        try:
            logger.info(f"Attempting {api_name} ({model_name})...")
            answer = call_fn(chunks, user_query)
            if answer:
                logger.info(f"✓ {api_name} succeeded")
                return {
                    "answer":   answer,
                    "api_used": api_name,
                    "model":    model_name,
                    "error":    None,
                }
        except Exception as e:
            error_msg = f"{api_name} failed: {str(e)[:200]}"
            logger.warning(error_msg)
            errors.append(error_msg)

    return {
        "answer":   None,
        "api_used": None,
        "model":    None,
        "error":    f"All providers failed: {'; '.join(errors)}",
    }


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        result = {
            "error":   "Insufficient arguments",
            "message": "Usage: python llm_answer.py 'user_query' 'chunk1' 'chunk2' ...",
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)

    user_query = sys.argv[1].strip()
    chunks     = [c.strip() for c in sys.argv[2:] if c.strip()]

    if not user_query or not chunks:
        print(json.dumps({"error": "Empty query or chunks"}))
        sys.exit(1)

    result = generate_answer(chunks[:5], user_query)

    if result["answer"]:
        print(json.dumps({
            "answer":            result["answer"],
            "api_used":          result["api_used"],
            "model":             result["model"],
            "chunks_processed":  len(chunks),
            "status":            "success",
        }, indent=2))
    else:
        print(json.dumps({
            "error":   "All LLM APIs failed",
            "details": result["error"],
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()