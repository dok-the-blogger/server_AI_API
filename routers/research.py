"""Authenticated, bounded control of asynchronous research jobs."""
from __future__ import annotations

import asyncio
import json
import secrets

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from experiments.doknews_research.server_config import MAX_SNAPSHOT_BYTES, ResearchError


router = APIRouter()


def _controller(request: Request):
    token = getattr(request.app.state, "research_service_token", "")
    if not token:
        raise ResearchError("service_token_missing", "Research service authentication is not configured.")
    supplied = request.headers.get("authorization", "")
    if not secrets.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
        raise ResearchError("unauthorized", "Unauthorized", 401)
    controller = getattr(request.app.state, "research_controller", None)
    if controller is None:
        raise ResearchError("research_not_configured", "Research service is not configured.")
    return controller


def _error(exc: ResearchError):
    return JSONResponse(status_code=exc.status, content={"detail": {"code": exc.code, "message": str(exc)}})


@router.post("/research/jobs")
async def start_research_job(request: Request):
    try:
        controller = _controller(request)
        raw = bytearray()
        async for part in request.stream():
            if len(raw) + len(part) > MAX_SNAPSHOT_BYTES + 4096:
                raise ResearchError("snapshot_too_large", "Research request exceeds the size limit.", 413)
            raw.extend(part)
        try:
            def reject_constant(_):
                raise ValueError()
            body = json.loads(raw, parse_constant=reject_constant)
        except (ValueError, UnicodeError):
            raise ResearchError("invalid_json", "Research request must contain valid JSON.", 422) from None
        result = await asyncio.to_thread(controller.submit, body)
        return JSONResponse(status_code=202 if result["status"] == "queued" else 200, content=result)
    except ResearchError as exc:
        return _error(exc)
    except Exception:
        return _error(ResearchError("research_unavailable", "Research state is temporarily unavailable."))


@router.get("/research/jobs/{job_id}")
async def research_job_status(job_id: str, request: Request):
    try:
        return await asyncio.to_thread(_controller(request).status, job_id)
    except ResearchError as exc:
        return _error(exc)
    except Exception:
        return _error(ResearchError("research_unavailable", "Research state is temporarily unavailable."))


@router.get("/research/jobs/{job_id}/result")
async def research_job_result(job_id: str, request: Request):
    try:
        return await asyncio.to_thread(_controller(request).result, job_id)
    except ResearchError as exc:
        return _error(exc)
    except Exception:
        return _error(ResearchError("research_unavailable", "Research result is temporarily unavailable."))
