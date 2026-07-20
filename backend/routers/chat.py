from fastapi import APIRouter

from schemas import ChatRequest, ChatResponse
from agent.graph import run_agent

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest):
    result = run_agent(body.question)
    return ChatResponse(
        answer=result.get("answer", ""),
        sql=result.get("sql") or None,
        columns=result.get("columns") or None,
        rows=result.get("rows") or None,
        error=result.get("error"),
    )
