from types import SimpleNamespace
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from travel_assistant import app as module
from travel_assistant.core import answer

class Store:
    def __init__(self): self.query = None
    def similarity_search(self, query, k):
        self.query = query
        return [Document(page_content='The museum closes Mondays.', metadata={'source': 'guide.txt'})]

class Model:
    def __init__(self): self.calls = []
    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content='museum opening days' if len(self.calls) == 1 else 'It closes Mondays [1].')

def test_followup_uses_rewritten_query_and_sources():
    store, model = Store(), Model()
    result = answer('When does it close?', [{'role': 'user', 'content': 'Tell me about the museum'}], store, model)
    assert store.query == 'museum opening days'
    assert result['sources'][0]['source'] == 'guide.txt'
    assert '[1]' in result['answer']

def test_chat_validation_and_health(monkeypatch):
    monkeypatch.setattr(module, 'get_services', lambda: (Store(), Model()))
    client = TestClient(module.app)
    assert client.get('/health').json()['status'] == 'ok'
    assert client.post('/chat', json={'question': ''}).status_code == 422
    assert client.post('/chat', json={'question': 'Museum?', 'history': []}).status_code == 200
    assert client.get('/').status_code == 200

def test_weather_endpoint(monkeypatch):
    monkeypatch.setattr(module, 'current_weather', lambda location: {'location': location})
    assert TestClient(module.app).get('/weather?location=Paris').json() == {'location': 'Paris'}


def test_actual_demo_route(monkeypatch, tmp_path):
    monkeypatch.setenv('TRAVEL_MODE', 'demo')
    monkeypatch.setenv('TRAVEL_LOCAL_INDEX', str(tmp_path / 'index.json'))
    module.get_services.cache_clear()
    client = TestClient(module.app)
    assert client.get('/status').json()['mode'] == 'demo'
    result = client.post('/chat', json={'question': 'What is the cancellation policy?'})
    assert result.status_code == 200
    assert 'seven days' in result.json()['answer']
    assert result.json()['sources']
    module.get_services.cache_clear()

def test_cloud_unconfigured_is_explicit(monkeypatch):
    monkeypatch.setenv('TRAVEL_MODE', 'cloud')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    module.get_services.cache_clear()
    response = TestClient(module.app).post('/chat', json={'question': 'Kyoto itinerary?'})
    assert response.status_code == 503
    assert 'Assistant unavailable' in response.json()['detail']
    module.get_services.cache_clear()


def stream_events(client, path, **kwargs):
    import json
    with client.stream("POST", path, **kwargs) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in response.iter_lines() if line]


def test_trace_demo_reports_real_retrieval_and_skips_cloud(monkeypatch, tmp_path):
    from travel_assistant.local import LocalStore
    monkeypatch.setattr(module, "get_services", lambda: (LocalStore(tmp_path / "empty.json"), None))
    events = stream_events(TestClient(module.app), "/chat/trace",
                           json={"question": "What should I pack for rain?", "top_k": 2})
    retrieval = next(e for e in events if e.get("stage") == "retrieve" and e.get("state") == "done")
    assert retrieval["detail"]["top_k"] == 2
    assert retrieval["detail"]["matches"][0]["source"] == "packing-checklist.md"
    assert "score" not in retrieval["detail"]["matches"][0]
    assert {e["stage"] for e in events if e.get("state") == "skipped"} >= {"embedding", "generate"}
    assert events[-1]["type"] == "result"
    assert "compact umbrella" in events[-1]["answer"]


def test_cloud_provider_calls_happen_between_stage_events():
    from travel_assistant.core import chat_pipeline
    calls = []

    class Embeddings:
        model = "test-embedding"

        def embed_query(self, query):
            calls.append(("embedding", query))
            return [0.2, -0.1, 0.5]

    class VectorStore:
        embeddings = Embeddings()

        def similarity_search_by_vector_with_score(self, vector, *, k):
            calls.append(("pinecone", vector, k))
            return [(Document(page_content="The museum closes Mondays.", metadata={"source": "guide.txt"}), 0.83)]

    class Chat:
        model_name = "test-chat"

        def invoke(self, messages):
            calls.append(("openai", messages))
            return SimpleNamespace(content="Closed Mondays [1].", usage_metadata={"input_tokens": 42, "output_tokens": 6})

    events = chat_pipeline("Museum hours?", [], VectorStore(), Chat(), top_k=2)
    seen = []
    for event in events:
        seen.append(event)
        if event.get("state") == "running":
            if event["stage"] == "embedding":
                assert calls == []
            elif event["stage"] == "retrieve":
                assert calls == [("embedding", "Museum hours?")]
            elif event["stage"] == "generate":
                assert calls[-1] == ("pinecone", [0.2, -0.1, 0.5], 2)
    embedding = next(e for e in seen if e.get("stage") == "embedding" and e.get("state") == "done")
    assert embedding["detail"]["dimensions"] == 3
    assert embedding["detail"]["sample"] == [0.2, -0.1, 0.5]
    retrieval = next(e for e in seen if e.get("stage") == "retrieve" and e.get("state") == "done")
    assert retrieval["detail"]["matches"][0]["score"] == 0.83
    assert "The museum closes Mondays." in calls[-1][1][1].content
    assert seen[-1]["answer"] == "Closed Mondays [1]."


def test_trace_weather_failure_is_visible_and_document_answer_continues(monkeypatch, tmp_path):
    from travel_assistant.local import LocalStore
    monkeypatch.setattr(module, "get_services", lambda: (LocalStore(tmp_path / "empty.json"), None))

    def fail_weather(city, trace):
        yield trace.event("geocode", "error", city=city)
        raise RuntimeError("Sensitive provider credentials should not escape")

    monkeypatch.setattr(module, "weather_pipeline", fail_weather)
    events = stream_events(TestClient(module.app), "/chat/trace",
                           json={"question": "What should I pack for rain?", "weather_location": "Kyoto"})
    assert {"type": "warning", "message": "Live weather was unavailable. Document search continued."} in events
    assert events[-1]["type"] == "result"
    assert "compact umbrella" in events[-1]["answer"]
    assert "could not be fetched" in events[-1]["answer"]
    assert "Sensitive provider" not in str(events)
