from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.graph import GraphState, compiled_graph
from app.llm import LLMError
from app.models import AgentSession
from app.schemas import AnswerRequest, CreateSessionRequest, SessionResponse

app = FastAPI(title="Intent-Driven Agentic Platform")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _to_response(record: AgentSession) -> SessionResponse:
    state = record.graph_state or {}
    return SessionResponse(
        session_id=record.id,
        status=record.status,
        intent=state.get("intent"),
        questions=state.get("questions") or None,
        slot_values=state.get("slot_values"),
        plan=state.get("plan"),
        result=record.result,
        error=record.error,
    )


def _run_graph(record: AgentSession, state: GraphState, db: Session) -> AgentSession:
    try:
        final_state = compiled_graph.invoke(state)
    except LLMError as exc:
        record.status = "failed"
        record.error = str(exc)
        db.commit()
        return record
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not swallowed silently
        record.status = "failed"
        record.error = f"Unexpected error: {exc}"
        db.commit()
        return record

    record.graph_state = final_state
    record.status = final_state.get("status", "failed")
    record.result = final_state.get("result")
    db.commit()
    db.refresh(record)
    return record


@app.post("/session", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, db: Session = Depends(get_db)) -> SessionResponse:
    record = AgentSession(topic=req.topic, status="classifying", graph_state={})
    db.add(record)
    db.commit()
    db.refresh(record)

    initial_state: GraphState = {"topic": req.topic, "slot_values": {}}
    record = _run_graph(record, initial_state, db)
    return _to_response(record)


@app.post("/session/{session_id}/answer", response_model=SessionResponse)
def answer_session(
    session_id: str, req: AnswerRequest, db: Session = Depends(get_db)
) -> SessionResponse:
    record = db.get(AgentSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    if record.status != "awaiting_input":
        raise HTTPException(status_code=400, detail=f"session is not awaiting input (status={record.status})")

    state: GraphState = dict(record.graph_state)
    state["slot_values"] = {**state.get("slot_values", {}), **req.slot_values}
    record = _run_graph(record, state, db)
    return _to_response(record)


@app.get("/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionResponse:
    record = db.get(AgentSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_response(record)
