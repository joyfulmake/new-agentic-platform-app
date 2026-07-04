from typing import Any, Literal

from pydantic import BaseModel, create_model


class SlotSpec(BaseModel):
    name: str
    description: str
    default: str | None = None


class TopicSchema(BaseModel):
    """LLM-generated description of what info is needed to process a topic."""

    intent: str
    mandatory: list[SlotSpec] = []
    optional: list[SlotSpec] = []


def build_slot_model(schema: TopicSchema) -> type[BaseModel]:
    """Builds a Pydantic model at runtime so mandatory slots are required
    fields and optional slots have defaults, mirroring the dynamic schema
    the LLM produced for this specific topic."""
    fields: dict[str, Any] = {}
    for slot in schema.mandatory:
        fields[slot.name] = (str, ...)
    for slot in schema.optional:
        fields[slot.name] = (str | None, slot.default)
    return create_model("DynamicSlots", **fields)


class PlanStepSpec(BaseModel):
    id: str
    tool: Literal["web_search", "web_fetch", "summarize", "synthesize"]
    depends_on: list[str] = []
    input: dict[str, Any] = {}


class Plan(BaseModel):
    steps: list[PlanStepSpec]


class CreateSessionRequest(BaseModel):
    topic: str


class AnswerRequest(BaseModel):
    slot_values: dict[str, str]


class SessionResponse(BaseModel):
    session_id: str
    status: str
    intent: str | None = None
    questions: list[SlotSpec] | None = None
    slot_values: dict[str, Any] | None = None
    plan: dict | None = None
    result: str | None = None
    error: str | None = None
