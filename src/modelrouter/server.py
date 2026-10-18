"""OpenAI-compatible FastAPI proxy. Point any OpenAI client's base_url at this server."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from . import __author__, __author_url__, __version__
from .config import Settings
from .factory import build_router
from .models import ModelError
from .router import Router, RouteResult
from .tracking import RequestLog
from .types import Tier

FORCE_MODELS: dict[str, Tier] = {"modelrouter-cheap": "cheap", "modelrouter-strong": "strong"}
AUTO_MODEL = "modelrouter-auto"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[Any] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = AUTO_MODEL
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


def _flatten_content(content: str | list[Any] | None) -> str:
    """Accept both plain-string content and OpenAI content-part lists."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
    return "\n".join(parts)


def _openai_body(result: RouteResult, request_id: str) -> dict[str, Any]:
    usage = result.usage
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.final.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.final.text},
                "finish_reason": result.final.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "modelrouter": {
            "route": result.final_tier,
            "initial_route": result.initial_tier,
            "p_hard": round(result.p_hard, 4),
            "confidence_threshold": result.confidence_threshold,
            "escalated": result.escalated,
            "escalation_reason": result.escalation_reason,
            "cost_usd": result.cost_usd,
            "baseline_strong_cost_usd": result.baseline_strong_cost_usd,
            "latency_ms": round(result.latency_s * 1000, 1),
        },
    }


async def _sse(body: dict[str, Any]) -> AsyncIterator[str]:
    """Replay a finished completion as an SSE stream.

    The router must see the whole answer before it can judge it, so true token streaming
    is not possible; clients that request `stream=true` still work.
    """
    base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }
    text = body["choices"][0]["message"]["content"]
    role = {
        **base,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role)}\n\n"
    chunk = {**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    yield f"data: {json.dumps(chunk)}\n\n"
    done = {
        **base,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": body["choices"][0]["finish_reason"]}
        ],
        "modelrouter": body["modelrouter"],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(
    settings: Settings | None = None,
    router: Router | None = None,
    log: RequestLog | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    router = router or build_router(settings)
    log = log or RequestLog(settings.db_path, log_queries=settings.log_queries)

    try:  # load the embedding model now so the first request isn't slow
        router.classifier.predict_proba_hard(["warm up"])
    except Exception:  # pragma: no cover - a broken classifier is surfaced per request
        pass

    app = FastAPI(
        title="ModelRouter",
        version=__version__,
        description=f"Cost-saving LLM routing proxy. Built by {__author__} ({__author_url__}).",
    )
    app.state.router = router
    app.state.log = log
    app.state.settings = settings

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "mock": settings.mock, "version": __version__}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        ids = [AUTO_MODEL, *FORCE_MODELS]
        return {
            "object": "list",
            "data": [{"id": i, "object": "model", "owned_by": "modelrouter"} for i in ids],
        }

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return log.summary()

    @app.get("/stats/recent")
    async def recent(limit: int = 50) -> list[dict[str, Any]]:
        return log.recent(min(max(limit, 1), 500))

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(req: ChatRequest, request: Request) -> Any:
        messages = [{"role": m.role, "content": _flatten_content(m.content)} for m in req.messages]
        if not messages:
            return _error(400, "messages must not be empty", "invalid_request_error")
        try:
            result = await router.handle(
                messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                force=FORCE_MODELS.get(req.model),
            )
        except ModelError as exc:
            return _error(502, str(exc), "upstream_error")
        log.record(result, mock=settings.mock)
        body = _openai_body(result, f"chatcmpl-{uuid.uuid4().hex[:24]}")
        headers = {
            "X-ModelRouter-Route": result.final_tier,
            "X-ModelRouter-Escalated": str(result.escalated).lower(),
            "X-ModelRouter-Cost-USD": f"{result.cost_usd:.8f}",
        }
        if req.stream:
            return StreamingResponse(_sse(body), media_type="text/event-stream", headers=headers)
        return JSONResponse(body, headers=headers)

    return app


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def app_factory() -> FastAPI:  # pragma: no cover - used by `uvicorn --factory`
    return create_app()
