# AI Travel Assistant · TravelAI

A runnable travel-document assistant with a browser interface, document citations, conversational follow-ups, document ingestion, and on-demand weather. It includes a local demo that starts without credentials and a cloud mode using LangChain, OpenAI, and Pinecone.

## Run locally

Use Python **3.10 or newer** (3.12 recommended):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn travel_assistant.app:app --host 127.0.0.1 --port 8001
```

Open **http://127.0.0.1:8001/**. API documentation is at `/docs`, health at `/health`, and the active mode/source list at `/status`.

The default `TRAVEL_MODE=demo` performs local keyword retrieval and returns document excerpts with citations. It does not run an LLM. Bundled documents contain a fictional Kyoto itinerary and booking policy, plus a packing checklist. They are explicitly marked as sample content. Try “What is the sample Kyoto itinerary?”, “What should I pack for rain?”, or “What is the cancellation policy?”.

## Live pipeline lab

Send a question to watch the pipeline in the dashboard. You can change the maximum document matches and add a city to fetch live weather as cited evidence for the same answer. Select a node or a row in the event timeline to see the operation, latency, and result. Use **Follow live** to return to the latest event.

- **Local document search** reports the real keyword matches and skips OpenAI and Pinecone. Browser notes run only in the tab and display their own trace.
- **Cloud retrieval** separates follow-up rewriting, embedding, Pinecone retrieval, context assembly, and OpenAI generation. The inspector displays native Pinecone scores and the first 24 query vector coordinates. It never prints API keys.
- **Weather** shows city resolution and the current conditions request separately. A weather failure remains visible and leaves document retrieval available.

## Index your own documents

Add UTF-8 `.txt`, `.md`, or text-based `.pdf` files to a directory, then run:

```sh
python -m travel_assistant.ingest documents --mode demo
```

This atomically replaces the local index (`.travel/index.json`). New chat requests read the updated index immediately. Remove this index to restore the bundled samples. Scanned PDF files require OCR before ingestion. The ingestion command fails explicitly if there are no readable documents.

## Cloud mode

Set `TRAVEL_MODE=cloud`, `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and optionally `PINECONE_NAMESPACE` in `.env`. Create a dense Pinecone index with dimension **1536** and cosine similarity for the default `text-embedding-3-small` embedding model. Change index dimensions appropriately if changing the embedding model.

```sh
python -m travel_assistant.ingest documents --mode cloud
python -m uvicorn travel_assistant.app:app --host 127.0.0.1 --port 8001
```

Cloud mode rewrites follow-up questions using conversation history, retrieves source chunks from Pinecone, and generates cited responses through OpenAI. Missing configuration or unavailable providers produce a clear error in the pipeline. The nonstreaming chat endpoint returns HTTP 503 for provider failures. A local ingestion manifest tracks the chunks belonging to each directory/index/namespace and removes superseded chunks on re-ingestion. Preserve `.travel/` for synchronization. Use separate namespaces for different document collections; identical relative filenames in the same namespace should not be ingested from different roots.

## API

```sh
curl -X POST http://127.0.0.1:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What should I pack for rain?","history":[]}'
```

`history` accepts up to 20 messages with `role` (`user` or `assistant`) and `content`. The response includes `answer`, `sources` (number, filename, and retrieved excerpt), and `mode`. History is held in the browser for the current page session; the server does not persist it.

For the live trace use `POST /chat/trace` with the same fields plus optional `top_k` (1–8) and `weather_location`. It streams newline-delimited JSON events, including stage start, completion, skip or failure, and a terminal result or error. Try:

```sh
curl -N http://127.0.0.1:8001/chat/trace \
  -H 'Content-Type: application/json' \
  -d '{"question":"What should I pack for rain?","top_k":2,"weather_location":"Kyoto, Japan"}'
```

`GET /weather/trace?location=Kyoto` streams a weather-only run.

`GET /weather?location=Kyoto` fetches current modelled conditions from [Open-Meteo](https://open-meteo.com/) with provider time and timezone. Internet access is required; weather requests work independently of document mode. The free API is subject to Open-Meteo's usage terms. Document answers do not have live flight, accommodation price, visa, or timetable feeds.

## Test

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions runs the same tests on Python 3.12 for pushes and pull requests. Tests cover local ingestion/replacement, unsupported questions, source citations, follow-ups, cloud orchestration with injected test doubles, API validation, and provider errors. Live OpenAI/Pinecone validation needs user credentials and an index; it is not part of offline tests.

This app binds to localhost by default. Add authentication, per-user namespaces, and request limits before exposing private documents publicly. Model and vector-store calls may incur charges; generated cloud answers require checking against their sources.
