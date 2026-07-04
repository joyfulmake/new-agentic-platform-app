"""Thin client for a locally-hosted Ollama model.

No external LLM API is called anywhere in this module — every request goes
to OLLAMA_HOST, which defaults to a model running on localhost.
"""

import json
import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


class LLMError(RuntimeError):
    pass


def chat(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(
            f"Could not reach local Ollama at {OLLAMA_HOST}. Is `ollama serve` "
            f"running and has `ollama pull {OLLAMA_MODEL}` been run? ({exc})"
        ) from exc

    return resp.json()["message"]["content"]


def chat_json(prompt: str, system: str | None = None, retries: int = 2) -> dict:
    """Asks the model for a JSON object and parses it, retrying with a
    corrective follow-up prompt if the model returns malformed JSON."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return json.loads(content)
        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach local Ollama at {OLLAMA_HOST}. Is `ollama serve` "
                f"running and has `ollama pull {OLLAMA_MODEL}` been run? ({exc})"
            ) from exc
        except (json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY a valid JSON object.",
                }
            )

    raise LLMError(f"Model did not return valid JSON after retries: {last_error}")
