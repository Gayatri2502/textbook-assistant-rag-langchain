from fastapi.testclient import TestClient

from travel_assistant.app import app, split_page


def test_split_preserves_exact_source_with_overlap():
    text = "One paragraph.\n\nA second paragraph.\n\nThe last one."
    spans = list(split_page(text, 25, 5, "recursive", ["\n\n", " "]))
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(right[0] == left[1] - 5 for left, right in zip(spans, spans[1:]))


def test_rejects_invalid_pdf_and_unknown_indexes():
    client = TestClient(app)
    response = client.post("/books", files={"file": ("bad.pdf", b"invalid", "application/pdf")})
    assert response.status_code == 422
    assert client.post("/books/missing/search", json={"question": "heart"}).status_code == 404


def test_experiment_route_serves_textbook_workspace():
    response = TestClient(app).get("/experiment?step=semantic-search")
    assert response.status_code == 200
    assert "Medical Textbook Playground" in response.text


def test_generated_claim_rejects_fabricated_quote_and_unverified_claim(monkeypatch):
    from travel_assistant import answers

    quote = "The sinoatrial node initiates the heartbeat."
    matches = [{"id": 987, "page": 987, "excerpt": quote, "score": 1}]
    monkeypatch.setenv("TEXTBOOK_ANSWER_MODEL", "test")
    responses = iter([
        {"claims": [
            {"id": 1, "text": "Invented claim", "quote": "Invented quote"},
            {"id": 1, "text": "Unsupported claim", "quote": quote},
            {"id": 1, "text": quote, "quote": quote},
        ]},
        {"supported": False},
        {"supported": True},
    ])
    monkeypatch.setattr(answers, "model_request", lambda *_: next(responses))
    claims = answers.answer_question("What does the sinoatrial node do?", matches, "generated")["claims"]
    assert [(claim["text"], claim["page"], claim["offset"]) for claim in claims] == [(quote, 987, 0)]
