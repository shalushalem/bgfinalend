from fastapi import APIRouter, Request
from brain.orchestrator import ahvi_orchestrator

router = APIRouter(prefix="/ai")

@router.post("/run")
def run_ai(payload: dict, request: Request):
    context = payload.get("context", {}) or {}
    if isinstance(context, dict):
        context.setdefault("request_id", str(getattr(request.state, "request_id", "") or ""))
    return ahvi_orchestrator.run(
        text=payload.get("message"),
        user_id=payload.get("userId"),
        context=context
    )
