import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_runtime_env() -> None:
    prompt_dir = Path(__file__).resolve().parent
    rl_root = prompt_dir.parent
    env_path = rl_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        return
    load_dotenv(override=True)


def _system_prompt_for_subaction(subaction: str) -> str:
    if subaction == "ExplainNewFact":
        return """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and concise.
- Do not quote the visitor verbatim.
- Keep continuity with the conversation.
- Use only fact IDs provided in the prompt when fact IDs are requested.
- Maximum two sentences."""
    if subaction in ("RepeatFact", "ClarifyFact"):
        return """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and concise.
- Do not quote the visitor verbatim.
- For RepeatFact, use the exact fact ID provided in the prompt.
- For ClarifyFact, avoid adding new fact IDs.
- Maximum two sentences."""
    return """You are a natural, conversational museum guide.

IMPORTANT GUIDELINES:
- Be natural and concise.
- Do not quote the visitor verbatim.
- Keep continuity with the conversation.
- Maximum two sentences."""


def generate_with_openai(
    prompt: str,
    subaction: str,
    model: str = "gpt-5.4",
    api_key_env: str = "api_key",
    timeout_sec: Optional[float] = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Package 'openai' is required for RL OpenAI generation.") from exc

    _load_runtime_env()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing OpenAI API key in env var '{api_key_env}'")

    client = OpenAI(api_key=api_key, timeout=timeout_sec)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt_for_subaction(subaction)},
            {"role": "user", "content": prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def generate_judge_json(
    prompt: str,
    model: str = "gpt-5.4",
    api_key_env: str = "api_key",
    timeout_sec: Optional[float] = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Package 'openai' is required for RL judge generation.") from exc

    _load_runtime_env()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing OpenAI API key in env var '{api_key_env}'")

    client = OpenAI(api_key=api_key, timeout=timeout_sec)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON verifier. Return valid JSON only, no markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def generate_revision_text(
    prompt: str,
    model: str = "gpt-5.4",
    api_key_env: str = "api_key",
    timeout_sec: Optional[float] = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Package 'openai' is required for RL revision generation.") from exc

    _load_runtime_env()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing OpenAI API key in env var '{api_key_env}'")

    client = OpenAI(api_key=api_key, timeout=timeout_sec)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Revise the candidate text only. Output final response text only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()

