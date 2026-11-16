from functools import lru_cache, partial
import json
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .core import answer, chat_pipeline, services
from .trace import Trace
from .weather import current_weather, weather_pipeline

load_dotenv()
app = FastAPI(title="AI Travel Assistant")
log = logging.getLogger(__name__)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    history: list[Message] = Field(default_factory=list, max_length=20)


class TraceRequest(ChatRequest):
    top_k: int = Field(default=4, ge=1, le=8)
    weather_location: str | None = Field(default=None, min_length=2, max_length=100, pattern=r"\S")


@lru_cache
def get_services():
    return services()


def runtime_mode():
    return os.getenv("TRAVEL_MODE", "demo").lower()


@app.get("/health")
def health():
    return {"status": "ok", "mode": runtime_mode()}


@app.get("/status")
def status():
    mode = runtime_mode()
    configured = all(os.getenv(k) for k in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"))
    result = {"mode": mode, "cloud_configured": configured}
    if mode == "demo":
        from .local import LocalStore
        result["documents"] = sorted({d.metadata.get("source", "unknown") for d in LocalStore().documents()})
    elif mode == "cloud" and configured:
        result["providers"] = {
            "index": os.environ["PINECONE_INDEX_NAME"],
            "namespace": os.getenv("PINECONE_NAMESPACE", "travel"),
            "embedding": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        }
    return result


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        store, model = get_services()
        return answer(request.question, [m.model_dump() for m in request.history], store, model)
    except Exception as exc:
        log.exception("Chat request failed")
        raise HTTPException(status_code=503, detail="Assistant unavailable; check server configuration and provider access.") from exc


def event_response(events):
    def lines():
        for event in events:
            yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/chat/trace")
def chat_trace(request: TraceRequest):
    def events():
        trace = Trace()
        try:
            store, model = yield from trace.step("connect", get_services, mode=runtime_mode())
            weather = partial(weather_pipeline, request.weather_location.strip()) if request.weather_location else None
            yield from chat_pipeline(request.question, [m.model_dump() for m in request.history],
                                     store, model, top_k=request.top_k, weather=weather, trace=trace)
        except Exception:
            log.exception("Chat pipeline failed")
            yield {"type": "error", "message": "The run could not finish. Inspect the failed step and check provider configuration."}

    return event_response(events())


@app.get("/", response_class=HTMLResponse)
def home():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


@app.get("/dashboard.js")
def dashboard_script():
    return FileResponse(Path(__file__).with_name("dashboard.js"), media_type="text/javascript")


@app.get("/dashboard.css")
def dashboard_styles():
    return FileResponse(Path(__file__).with_name("dashboard.css"), media_type="text/css")


@app.get("/weather")
def weather(location: str = Query(min_length=2, max_length=100)):
    try:
        return current_weather(location)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Weather service unavailable.") from exc


@app.get("/weather/trace")
def weather_trace(location: str = Query(min_length=2, max_length=100, pattern=r"\S")):
    def events():
        trace = Trace()
        yield trace.event("input", "done", city=location, mode="weather")
        for stage in ("query", "embedding", "retrieve", "context", "generate"):
            yield trace.event(stage, "skipped", reason="Weather-only lookup.")
        try:
            result = yield from weather_pipeline(location.strip(), trace)
            yield {"type": "result", **result}
        except ValueError:
            yield {"type": "error", "message": "Location not found. Try adding a country."}
        except Exception:
            log.exception("Weather pipeline failed")
            yield {"type": "error", "message": "Weather service unavailable. Try again shortly."}

    return event_response(events())
