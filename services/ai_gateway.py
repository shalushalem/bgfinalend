import json
import os
import re
from typing import Any, Dict, List, Tuple

import requests

from services import llm_service


def generate_text(
    prompt: str,
    *,
    options: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> str:
    return llm_service.generate_text(
        prompt=prompt,
        options=options,
        user_profile=user_profile,
        signals=signals,
    )


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    system_instruction: str = "",
    model: str | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> str:
    return llm_service.chat_completion(
        messages=messages,
        system_instruction=system_instruction,
        model=model or llm_service.DEFAULT_MODEL,
        user_profile=user_profile,
        signals=signals,
    )


def extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response")

    clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        return json.loads(clean)
    except Exception:
        pass

    obj_start = clean.find("{")
    obj_end = clean.rfind("}")
    arr_start = clean.find("[")
    arr_end = clean.rfind("]")

    candidates: List[str] = []
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(clean[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(clean[arr_start : arr_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("no valid JSON found in model response")


def parse_json_object(text: str) -> Dict[str, Any]:
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def parse_json_array(text: str) -> List[Any]:
    parsed = extract_json(text)
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON array")
    return parsed


def generate_json_object(
    prompt: str,
    *,
    options: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return parse_json_object(
        generate_text(
            prompt=prompt,
            options=options,
            user_profile=user_profile,
            signals=signals,
        )
    )


def chat_json_object(
    messages: List[Dict[str, Any]],
    *,
    system_instruction: str = "",
    model: str | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return parse_json_object(
        chat_completion(
            messages=messages,
            system_instruction=system_instruction,
            model=model,
            user_profile=user_profile,
            signals=signals,
        )
    )


def _vision_model_candidates() -> List[str]:
    preferred = str(os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:latest") or "").strip()
    fallback_raw = str(
        os.getenv(
            "OLLAMA_VISION_MODEL_FALLBACKS",
            "llama3.2-vision:latest,llama3.2-vision",
        )
        or ""
    ).strip()
    ordered: List[str] = []
    for model in [preferred, *[m.strip() for m in fallback_raw.split(",")]]:
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def _ollama_generate_url() -> str:
    base = str(os.getenv("OLLAMA_URL", "http://localhost:11434/api") or "").strip().rstrip("/")
    return f"{base}/generate" if base.endswith("/api") else f"{base}/api/generate"


def ollama_vision_json(
    *,
    prompt: str,
    image_base64: str,
    timeout_seconds: int | None = None,
) -> Tuple[Dict[str, Any], str]:
    timeout = int(timeout_seconds or int(os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "120")))
    payload = {
        "prompt": prompt,
        "images": [str(image_base64 or "").strip()],
        "stream": False,
        "format": "json",
    }

    last_error: Exception | None = None
    for model in _vision_model_candidates():
        try:
            response = requests.post(
                _ollama_generate_url(),
                json={**payload, "model": model},
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Ollama vision request failed model={model} status={response.status_code}"
                )
            raw = response.json().get("response", "{}")
            parsed = parse_json_object(raw)
            return parsed, model
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(str(last_error or "vision generation failed"))
