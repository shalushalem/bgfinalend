from typing import Any, Dict, List

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
