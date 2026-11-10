from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pathlib import Path
import logging
import os
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from .core import answer, services
from .weather import current_weather

load_dotenv()
app = FastAPI(title="AI Travel Assistant")

class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=10000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r'\S')
    history: list[Message] = Field(default_factory=list, max_length=20)

@lru_cache
def get_services():
    return services()

@app.get('/health')
def health():
    return {'status': 'ok', 'mode': os.getenv('TRAVEL_MODE', 'demo')}

@app.get('/status')
def status():
    mode = os.getenv('TRAVEL_MODE', 'demo')
    result = {'mode': mode, 'cloud_configured': all(os.getenv(k) for k in ('OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME'))}
    if mode == 'demo':
        from .local import LocalStore
        result['documents'] = sorted({d.metadata.get('source', 'unknown') for d in LocalStore().documents()})
    return result

@app.post('/chat')
def chat(request: ChatRequest):
    try:
        store, model = get_services()
        return answer(request.question, [m.model_dump() for m in request.history], store, model)
    except Exception as exc:
        logging.getLogger(__name__).exception('Chat request failed')
        # Do not expose provider errors or credentials in HTTP responses.
        raise HTTPException(status_code=503, detail='Assistant unavailable; check server configuration and provider access.') from exc

@app.get('/', response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')

@app.get('/weather')
def weather(location: str = Query(min_length=2, max_length=100)):
    try:
        return current_weather(location)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Weather service unavailable.') from exc
