"""LangGraph orchestration for the intent-driven pipeline:

classify_intent -> generate_schema -> check_slots -> [plan -> execute -> synthesize]

The graph is re-invoked once per HTTP request (see main.py), carrying the
prior state forward. Each of the first three nodes is a no-op once its
output is already present in state, which is what lets the same compiled
graph be safely invoked again after the user answers clarifying questions
without a LangGraph checkpointer.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app import llm
from app.schemas import Plan, TopicSchema
from app.tools import web_fetch, web_search


class GraphState(TypedDict, total=False):
    topic: str
    intent: str
    mandatory_slots: list[dict]
    optional_slots: list[dict]
    slot_values: dict[str, str]
    status: str
    questions: list[dict]
    plan: dict
    step_outputs: dict[str, str]
    result: str


def classify_intent(state: GraphState) -> dict:
    if state.get("intent"):
        return {}
    data = llm.chat_json(
        system=(
            "You classify the high-level task category of a user's topic of "
            "interest. Reply with ONLY JSON: {\"intent\": \"<short_snake_case_label>\"}"
        ),
        prompt=f"Topic: {state['topic']}",
    )
    return {"intent": data.get("intent", "general")}


def generate_schema(state: GraphState) -> dict:
    if state.get("mandatory_slots") is not None:
        return {}
    data = llm.chat_json(
        system=(
            "Given a topic and its intent category, decide what information is "
            "needed to fully process it well. Reply with ONLY JSON matching: "
            '{"intent": str, "mandatory": [{"name": str, "description": str}], '
            '"optional": [{"name": str, "description": str, "default": str}]}. '
            "Keep names short snake_case identifiers. Mandatory slots are things "
            "that are truly required to do a good job; everything else is optional "
            "with a sensible default."
        ),
        prompt=f"Topic: {state['topic']}\nIntent: {state.get('intent', 'general')}",
    )
    schema = TopicSchema.model_validate(data)
    return {
        "mandatory_slots": [s.model_dump() for s in schema.mandatory],
        "optional_slots": [s.model_dump() for s in schema.optional],
    }


def check_slots(state: GraphState) -> dict:
    slot_values = dict(state.get("slot_values", {}))
    mandatory = state.get("mandatory_slots", [])
    optional = state.get("optional_slots", [])

    for slot in optional:
        if not slot_values.get(slot["name"]):
            slot_values[slot["name"]] = slot.get("default") or ""

    missing = [s for s in mandatory if not slot_values.get(s["name"])]
    if missing:
        return {"slot_values": slot_values, "status": "awaiting_input", "questions": missing}
    return {"slot_values": slot_values, "status": "ready", "questions": []}


def route_after_check_slots(state: GraphState) -> Literal["make_plan", "__end__"]:
    return "make_plan" if state.get("status") == "ready" else END


def make_plan(state: GraphState) -> dict:
    data = llm.chat_json(
        system=(
            "Design a short ordered plan to research and produce a great answer "
            "for the given topic, using only these tools: web_search, web_fetch, "
            "summarize, synthesize. Typical shape: one web_search step, a few "
            "web_fetch steps for the top results, a summarize step, then a final "
            "synthesize step. Reply with ONLY JSON matching: "
            '{"steps": [{"id": str, "tool": str, "depends_on": [str], "input": object}]}'
        ),
        prompt=(
            f"Topic: {state['topic']}\nIntent: {state.get('intent')}\n"
            f"Known details: {state.get('slot_values', {})}"
        ),
    )
    parsed = Plan.model_validate(data)
    return {"plan": parsed.model_dump()}


def _topo_order(steps: list[dict]) -> list[dict]:
    by_id = {s["id"]: s for s in steps}
    visited: set[str] = set()
    ordered: list[dict] = []

    def visit(step: dict) -> None:
        if step["id"] in visited:
            return
        visited.add(step["id"])
        for dep in step.get("depends_on", []):
            if dep in by_id:
                visit(by_id[dep])
        ordered.append(step)

    for s in steps:
        visit(s)
    return ordered


def execute(state: GraphState) -> dict:
    steps = _topo_order(state.get("plan", {}).get("steps", []))
    outputs: dict[str, str] = {}
    topic = state["topic"]
    slot_values = state.get("slot_values", {})

    for step in steps:
        tool = step["tool"]
        try:
            if tool == "web_search":
                query = step.get("input", {}).get("query", topic)
                results = web_search.search(query)
                outputs[step["id"]] = "\n".join(f"{r['title']}: {r['url']}" for r in results)
            elif tool == "web_fetch":
                url = step.get("input", {}).get("url")
                if not url:
                    dep_urls = _extract_urls(outputs, step.get("depends_on", []))
                    url = dep_urls[0] if dep_urls else None
                outputs[step["id"]] = web_fetch.fetch(url) if url else ""
            elif tool == "summarize":
                context = "\n\n".join(outputs.get(d, "") for d in step.get("depends_on", []))
                outputs[step["id"]] = llm.chat(
                    system="Summarize the following content, respecting these preferences: "
                    f"{slot_values}.",
                    prompt=context[:12000],
                )
            elif tool == "synthesize":
                context = "\n\n".join(outputs.get(d, "") for d in step.get("depends_on", []))
                outputs[step["id"]] = llm.chat(
                    system=(
                        "Compile a final, complete answer for the user's topic from "
                        f"the research notes below, honoring these preferences: {slot_values}."
                    ),
                    prompt=f"Topic: {topic}\n\nNotes:\n{context[:12000]}",
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user via step output
            outputs[step["id"]] = f"[step failed: {exc}]"

    return {"step_outputs": outputs}


def _extract_urls(outputs: dict[str, str], dep_ids: list[str]) -> list[str]:
    urls = []
    for dep in dep_ids:
        for line in outputs.get(dep, "").splitlines():
            if ": http" in line:
                urls.append(line.split(": ", 1)[1].strip())
    return urls


def synthesize(state: GraphState) -> dict:
    outputs = state.get("step_outputs", {})
    synth_steps = [s for s in state.get("plan", {}).get("steps", []) if s["tool"] == "synthesize"]
    if synth_steps:
        result = outputs.get(synth_steps[-1]["id"], "")
    else:
        result = "\n\n".join(outputs.values())
    return {"result": result, "status": "done"}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("generate_schema", generate_schema)
    graph.add_node("check_slots", check_slots)
    graph.add_node("make_plan", make_plan)
    graph.add_node("execute", execute)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "generate_schema")
    graph.add_edge("generate_schema", "check_slots")
    graph.add_conditional_edges(
        "check_slots", route_after_check_slots, {"make_plan": "make_plan", END: END}
    )
    graph.add_edge("make_plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


compiled_graph = build_graph()
