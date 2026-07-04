import datetime
import uuid

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class AgentSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    topic: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="classifying")
    # status: classifying -> awaiting_input -> ready -> done -> failed
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full LangGraph state blob (intent, schema, slot lists, slot_values,
    # plan, step_outputs, questions) so a session can be resumed across
    # separate HTTP requests without needing a LangGraph checkpointer.
    graph_state: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    steps: Mapped[list["PlanStep"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class PlanStep(Base):
    __tablename__ = "plan_steps"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    step_index: Mapped[int] = mapped_column()
    tool: Mapped[str] = mapped_column(String)
    input_json: Mapped[dict] = mapped_column(JSON, default=dict)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    session: Mapped[AgentSession] = relationship(back_populates="steps")
