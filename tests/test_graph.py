"""Exercises the full LangGraph flow with the local-model calls and web
tools mocked out, so it runs deterministically with no Ollama and no
network access."""

from app import graph
from app.tools import web_search


def fake_chat_json(system: str = "", prompt: str = "", retries: int = 2) -> dict:
    if "classify" in system:
        return {"intent": "research"}
    if "what information is needed" in system:
        return {
            "intent": "research",
            "mandatory": [{"name": "scope", "description": "What angle to focus on"}],
            "optional": [{"name": "tone", "description": "Desired tone", "default": "neutral"}],
        }
    if "Design a short ordered plan" in system:
        return {
            "steps": [
                {"id": "s1", "tool": "web_search", "depends_on": [], "input": {"query": "topic"}},
                {"id": "s2", "tool": "synthesize", "depends_on": ["s1"], "input": {}},
            ]
        }
    raise AssertionError(f"unexpected chat_json call, system={system!r}")


def fake_chat(system: str = "", prompt: str = "") -> str:
    return "final synthesized answer"


def test_missing_mandatory_slot_pauses_for_clarification(monkeypatch):
    monkeypatch.setattr(graph.llm, "chat_json", fake_chat_json)

    state: graph.GraphState = {"topic": "quantum computing and cryptography", "slot_values": {}}
    result = graph.compiled_graph.invoke(state)

    assert result["status"] == "awaiting_input"
    assert result["intent"] == "research"
    assert [q["name"] for q in result["questions"]] == ["scope"]
    # optional slot got its default filled in already
    assert result["slot_values"]["tone"] == "neutral"


def test_full_run_once_mandatory_slots_are_filled(monkeypatch):
    monkeypatch.setattr(graph.llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(graph.llm, "chat", fake_chat)
    monkeypatch.setattr(
        web_search,
        "search",
        lambda query, limit=5: [{"title": "Example", "url": "https://example.com"}],
    )

    # Round 1: discovers the missing mandatory slot.
    state: graph.GraphState = {"topic": "quantum computing and cryptography", "slot_values": {}}
    paused = graph.compiled_graph.invoke(state)
    assert paused["status"] == "awaiting_input"

    # Round 2: user supplies the missing slot; carry the rest of the state forward.
    paused["slot_values"] = {**paused["slot_values"], "scope": "post-quantum algorithms"}
    final = graph.compiled_graph.invoke(paused)

    assert final["status"] == "done"
    assert final["result"] == "final synthesized answer"
    assert final["plan"]["steps"][0]["tool"] == "web_search"
