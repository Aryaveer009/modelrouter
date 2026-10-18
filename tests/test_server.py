import json

import pytest
from fastapi.testclient import TestClient

from conftest import FixedClassifier
from modelrouter.config import Settings
from modelrouter.factory import build_router
from modelrouter.server import create_app
from modelrouter.tracking import RequestLog


@pytest.fixture
def client(tmp_path):
    settings = Settings.from_env(
        {"MODELROUTER_MOCK": "1", "MODELROUTER_DB": str(tmp_path / "s.db")}, use_dotenv=False
    )
    clf = FixedClassifier({"hard question": 0.95}, default=0.05)
    app = create_app(settings, build_router(settings, clf), RequestLog(settings.db_path))
    return TestClient(app)


def post(client, content, **extra):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "modelrouter-auto",
            "messages": [{"role": "user", "content": content}],
            **extra,
        },
    )


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "modelrouter-auto" in ids


def test_openai_shape_and_routing_metadata(client):
    r = post(client, "easy question")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert (
        body["usage"]["total_tokens"]
        == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["modelrouter"]["route"] == "cheap" and body["model"] == "mock-cheap"
    assert r.headers["x-modelrouter-route"] == "cheap"


def test_hard_question_routed_to_strong(client):
    body = post(client, "hard question").json()
    assert body["modelrouter"]["route"] == "strong" and body["model"] == "mock-strong"


def test_model_field_can_force_route(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "modelrouter-strong",
            "messages": [{"role": "user", "content": "easy question"}],
        },
    )
    assert r.json()["modelrouter"]["route"] == "strong"


def test_content_parts_are_accepted(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hard question"}]}]
        },
    )
    assert r.json()["modelrouter"]["route"] == "strong"


def test_streaming_returns_sse_chunks(client):
    r = post(client, "easy question", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[1][6:])
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["content"]


def test_requests_are_logged_and_exposed_in_stats(client):
    post(client, "easy question")
    post(client, "hard question")
    stats = client.get("/stats").json()
    assert stats["requests"] == 2 and stats["mock_requests"] == 2
    assert len(client.get("/stats/recent").json()) == 2


def test_empty_messages_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400 and "error" in r.json()
