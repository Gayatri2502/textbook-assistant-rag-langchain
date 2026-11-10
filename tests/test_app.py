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
