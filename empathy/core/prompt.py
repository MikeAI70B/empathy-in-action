import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import openai  # type: ignore
except Exception:
    openai = None  # Fallback if not installed

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # Optional dependency


# Provider and model registry for experiments
PROVIDER_MODEL_REGISTRY: Dict[str, List[str]] = {
    "together": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        #"Qwen/Qwen3-235B-A22B-Thinking-2507",
        #"deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        #"moonshotai/Kimi-K2-Instruct",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022"
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ],
    "xai": [
        "grok-4-0709",
        "grok-3",
        "grok-3-mini",
    ],
    "openai": [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1"
        
    ],
}


def _retry_with_backoff(
    func,
    *args,
    attempts: int = 3,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 30.0,
    jitter_ratio: float = 0.2,
    **kwargs,
):
    """Retry a callable with exponential backoff and jitter.

    - attempts: total attempts including the first try
    - waits grow exponentially and are capped at max_delay_seconds
    - jitter adds/subtracts up to jitter_ratio of the delay to avoid thundering herds
    """
    last_exc: Optional[Exception] = None
    for attempt_index in range(attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # retry on any error surfaced by providers
            last_exc = exc
            is_last = attempt_index >= attempts - 1
            if is_last:
                break
            # exponential backoff with cap and jitter
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** attempt_index))
            try:
                import random
                import time
                jitter = delay * jitter_ratio
                sleep_for = max(0.0, delay + random.uniform(-jitter, jitter))
                print(f"Retrying in {sleep_for:.1f}s (attempt {attempt_index + 2}/{attempts}) due to: {exc}")
                time.sleep(sleep_for)
            except Exception:
                # If sleep/jitter fails for any reason, still proceed quickly
                pass
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_exc}")


def get_available_models() -> List[Tuple[str, str]]:
    """Return list of (provider, model) tuples available for experiments."""
    pairs: List[Tuple[str, str]] = []
    for provider, models in PROVIDER_MODEL_REGISTRY.items():
        for model in models:
            pairs.append((provider, model))
    return pairs


def _build_messages(query: str, system: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": query})
    return messages


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object embedded in free-form text.

    Strategies:
    1) Look for fenced code block ```json ... ```
    2) Look for fenced code block ``` ... ```
    3) Naively scan for the first balanced {...} object
    """
    import re

    # 1) ```json ... ```
    fence_json = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_json:
        candidate = fence_json.group(1).strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 2) generic ``` ... ```
    fence = re.search(r"```\s*([\s\S]*?)```", text)
    if fence:
        candidate = fence.group(1).strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3) naive balanced braces scan
    start_idx = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if start_idx is None:
                start_idx = i
            depth += 1
        elif ch == '}':
            if start_idx is not None:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i+1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        return None
    return None


def _call_openai(messages: List[Dict[str, str]], model: str, seed: int) -> str:
    if openai is None:
        raise RuntimeError("openai package not installed")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Heuristic is disabled; LLM is required.")
    openai.api_key = openai_api_key
    try:
        print("Calling OpenAI...")
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            seed=seed,
        )
        return response.choices[0].message.content  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"LLM call failed (openai): {e}")


def _call_together(messages: List[Dict[str, str]], model: str, seed: int) -> str:
    together_api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not together_api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set. Heuristic is disabled; LLM is required.")
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "seed": seed,#42
        "stream": False,
    }
    try:
        print("Calling Together...")
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        resp_json = resp.json()
        return resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM call failed (together): {e}")


def _call_anthropic(query: str, system: Optional[str], model: str, seed: int) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Heuristic is disabled; LLM is required.")
    try:
        print("Calling Anthropic...")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=1,
            #seed=42,
            system=system or None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query,
                        }
                    ],
                }
            ],
        )
        return message.content[0].text  # type: ignore[index]
    except Exception as e:
        raise RuntimeError(f"LLM call failed (anthropic): {e}")


def _call_google_gemini(query: str, system: Optional[str], model: str, seed: int) -> str:
    api_key = os.environ.get("GOOGLE_AI_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_AI_KEY is not set. Heuristic is disabled; LLM is required.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": query}
                ],
            }
        ],
        "generationConfig": {
            "seed": seed,
            "temperature": 1
        },
    }
    if system:
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [
                {"text": system}
            ],
        }
    params = {"key": api_key}
    try:
        print("Calling Google Gemini...")
        resp = requests.post(url, headers=headers, json=payload, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except Exception as e:
        raise RuntimeError(f"LLM call failed (google): {e}")


def _call_xai_grok(messages: List[Dict[str, str]], model: str, seed: int) -> str:
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set. Heuristic is disabled; LLM is required.")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "messages": messages,
        "model": model,
        "stream": False,
        "temperature": 1,
        "seed": seed,
    }
    try:
        print("Calling xAI Grok...")
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # x.ai format mirrors OpenAI style
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM call failed (xai): {e}")


def call_llm_with_prompt(query: str, system: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Call the configured LLM provider with the given prompt.

    Selection order:
    - explicit provider/model arguments
    - environment variables LLM_PROVIDER, LLM_MODEL
    - provider-specific default models
    """
    selected_provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    selected_model = model or os.environ.get("LLM_MODEL")
    # Seed selection: CLI/env override then default
    selected_seed: int = 1234
    if seed is not None:
        try:
            selected_seed = int(seed)
        except Exception:
            selected_seed = 1234
    else:
        env_seed = os.environ.get("LLM_SEED")
        if env_seed is not None:
            try:
                selected_seed = int(env_seed)
            except Exception:
                selected_seed = 1234

    # Default model per provider if not specified
    if not selected_model:
        if selected_provider == "together":
            selected_model = "openai/gpt-oss-20b"
        elif selected_provider == "anthropic":
            selected_model = "claude-3-7-sonnet-20250219"
        elif selected_provider == "google":
            selected_model = "gemini-2.5-flash"
        else:
            selected_model = "gpt-4.1"

    messages = _build_messages(query, system)

    if selected_provider == "together":
        ai_message = _retry_with_backoff(_call_together, messages, selected_model, selected_seed)
    elif selected_provider == "anthropic":
        ai_message = _retry_with_backoff(_call_anthropic, query, system, selected_model, selected_seed)
    elif selected_provider == "google":
        ai_message = _retry_with_backoff(_call_google_gemini, query, system, selected_model, selected_seed)
    elif selected_provider == "xai":
        ai_message = _retry_with_backoff(_call_xai_grok, messages, selected_model, selected_seed)
    else:
        ai_message = _retry_with_backoff(_call_openai, messages, selected_model, selected_seed)

    # Attempt to parse JSON; otherwise try to extract from text; else return raw content
    try:
        data = json.loads(ai_message)
        if isinstance(data, dict):
            return data
        return {"raw": ai_message}
    except Exception:
        extracted = _extract_json_from_text(ai_message)
        if isinstance(extracted, dict):
            return extracted
        return {"raw": ai_message}


def list_models() -> Dict[str, List[str]]:
    """Expose the registry for external callers and CLIs."""
    return dict(PROVIDER_MODEL_REGISTRY)
