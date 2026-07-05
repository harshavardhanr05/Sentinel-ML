"""
backend/llm/client.py
─────────────────────
Single adapter for all LLM calls in Sentinel-ML.
Reads LLM_PROVIDER from env (default: gemini) and routes accordingly.
Every agent calls get_llm_response() — never the SDK directly — so
switching providers is a one-line .env change, no code change.

Currently supports: gemini (default), ollama (local models)
"""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Gemini client (lazy-init so import is fast even if key is missing)
# ---------------------------------------------------------------------------

_gemini_client: Any = None


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        if not _GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example → .env and add your free key "
                "from https://aistudio.google.com"
            )
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=_GEMINI_API_KEY)
            _gemini_client = genai.GenerativeModel(_GEMINI_MODEL)
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            ) from e
    return _gemini_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[str, str] = {}


def get_llm_response(
    prompt: str,
    *,
    expect_json: bool = False,
    retry_on_json_fail: bool = True,
    temperature: float = 0.2,
) -> str:
    """
    Send a prompt to the configured LLM and return the text response.

    Args:
        prompt: The full prompt string.
        expect_json: If True, instructs the model to return ONLY valid JSON
                     and retries once with a stricter instruction on parse failure.
        retry_on_json_fail: Only relevant when expect_json=True. Default True.
        temperature: Generation temperature (lower = more deterministic).

    Returns:
        Raw string response from the model.

    Raises:
        RuntimeError: If the provider is unknown or the API call fails.
    """
    # ── AI API Limit Saver (Cache) ──
    # Hashes the exact prompt. If we've seen it before (e.g. loops), return cached.
    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    if prompt_hash in _LLM_CACHE:
        print(f"[Sentinel-ML] LLM Cache Hit! Skipping API request.")
        return _LLM_CACHE[prompt_hash]

    if _PROVIDER == "gemini":
        result = _call_gemini(prompt, expect_json=expect_json, retry_on_json_fail=retry_on_json_fail)
    elif _PROVIDER == "ollama":
        result = _call_ollama(prompt, expect_json=expect_json, retry_on_json_fail=retry_on_json_fail)
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER='{_PROVIDER}'. Set LLM_PROVIDER=gemini or ollama in .env."
        )

    _LLM_CACHE[prompt_hash] = result
    return result


def get_llm_json(prompt: str) -> dict[str, Any]:
    """
    Convenience wrapper: calls get_llm_response with expect_json=True and
    parses the result into a dict. Raises ValueError if the response is not
    valid JSON after retries.
    """
    raw = get_llm_response(prompt, expect_json=True)
    return _parse_json_safe(raw)


def get_llm_text(prompt: str) -> str:
    """
    Convenience wrapper: calls get_llm_response and returns the raw text.
    Use this when you need plain prose from the LLM (not JSON).
    """
    return get_llm_response(prompt, expect_json=False)


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


def _call_gemini(
    prompt: str,
    *,
    expect_json: bool = False,
    retry_on_json_fail: bool = True,
) -> str:
    client = _get_gemini_client()
    full_prompt = prompt
    if expect_json:
        full_prompt = (
            prompt
            + "\n\nIMPORTANT: Respond with ONLY valid JSON — no markdown fences, no prose, "
            "no explanation. The entire response must be parseable by json.loads()."
        )

    try:
        response = client.generate_content(full_prompt)
        text = response.text.strip()

        if expect_json and retry_on_json_fail:
            try:
                _parse_json_safe(text)  # validate
            except ValueError:
                # Retry with an even stricter prompt
                time.sleep(1)
                stricter = (
                    "Your previous response was not valid JSON. Respond with ONLY a JSON object. "
                    "No markdown, no backticks, no explanation.\n\n"
                    f"Original task:\n{prompt}"
                )
                retry_response = client.generate_content(stricter)
                text = retry_response.text.strip()

        return text

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}") from e


# ---------------------------------------------------------------------------
# Ollama implementation
# ---------------------------------------------------------------------------


def _call_ollama(
    prompt: str,
    *,
    expect_json: bool = False,
    retry_on_json_fail: bool = True,
) -> str:
    import urllib.request
    import urllib.error

    url = f"{_OLLAMA_HOST.rstrip('/')}/api/generate"
    
    full_prompt = prompt
    if expect_json:
        full_prompt += (
            "\n\nIMPORTANT: Respond with ONLY valid JSON. Do not include markdown formatting "
            "like ```json or any explanation text."
        )

    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    # Force JSON mode in Ollama if expected
    if expect_json:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            text = result.get("response", "").strip()

            if expect_json and retry_on_json_fail:
                try:
                    _parse_json_safe(text)
                except ValueError:
                    time.sleep(1)
                    stricter = (
                        "Your previous response was not valid JSON. Respond with ONLY a JSON object.\n\n"
                        f"Original task:\n{prompt}"
                    )
                    payload["prompt"] = stricter
                    data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req) as retry_resp:
                        retry_result = json.loads(retry_resp.read().decode("utf-8"))
                        text = retry_result.get("response", "").strip()

            return text
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama API call failed. Is Ollama running on {_OLLAMA_HOST}? Error: {e}") from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_safe(text: str) -> dict[str, Any]:
    """Parse JSON from model response, handling common wrapping (```json ... ```)."""
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response is not valid JSON.\nResponse: {text[:500]}\nError: {e}") from e
