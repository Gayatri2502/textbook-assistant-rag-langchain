# AI Travel Assistant

A starting implementation of a conversational, document-grounded travel assistant using FastAPI, LangChain, OpenAI embeddings/chat, and Pinecone. It indexes your own travel documents and returns answer citations with the retrieved source text. Conversation history is supplied by the client and is not stored on the server.

## Setup

1. Use Python 3.10+ and create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set the OpenAI and Pinecone credentials and index name. Create a Pinecone dense index with dimension **1536** and cosine similarity for the default `text-embedding-3-small` model. Use the appropriate dimensions if changing embedding models.
4. Put `.txt`, `.md`, or text-based `.pdf` travel documents in `documents/`, then run `python -m travel_assistant.ingest documents`.
5. Start the API: `uvicorn travel_assistant.app:app --reload`. Open `http://127.0.0.1:8000/` for the chat interface, or `/docs` for interactive API documentation.

Example request:

```sh
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"question":"What should I know before visiting?", "history":[]}'
```

Follow-up requests can include prior messages as `{"role":"user","content":"..."}` or `{"role":"assistant","content":"..."}`. The response includes the answer and numbered source excerpts. Ingesting unchanged chunks is idempotent, but changed or deleted documents can leave stale chunks: clear the namespace and re-ingest when replacing documents.

## Scope and next steps

The weather panel fetches modelled current conditions on demand from [Open-Meteo](https://open-meteo.com/) and displays the provider timestamp. Document retrieval itself is not live; flight prices, schedules and entry rules are not connected to live providers. Open-Meteo free API usage is subject to its terms, including non-commercial usage restrictions. Authentication, persistent conversations, provider retries, ingestion synchronization, and deployment are future work. Model calls and Pinecone usage may incur charges. Do not index private documents into a shared namespace without access controls.
