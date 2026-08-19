import pytest
from fastapi.testclient import TestClient
from endpoints.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/ai_query")
    assert response.status_code == 200

def test_get_ai_query():
    response = client.get("/ai_query?prompt=Hola")
    assert response.status_code == 200
    assert "response" in response.json()

def test_post_ask_ai_default():
    response = client.post("/ask_ai", json={"prompt": "Hola"})
    assert response.status_code == 200
    assert "response" in response.json()

def test_post_ask_ai_ollama_existing():
    response = client.post("/ask_ai", json={
        "prompt": "Hola",
        "provider": "local",
        "model": "qwen2.5-coder:7b"
    })
    assert response.status_code == 200
    assert "response" in response.json()

def test_post_ask_ai_ollama_not_existing():
    response = client.post("/ask_ai", json={
        "prompt": "Hola",
        "provider": "local",
        "model": "modelo:inexistente"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["response"] is None
    assert "Ollama error" in data["error"]

def test_post_ask_ai_invalid_provider():
    response = client.post("/ask_ai", json={
        "prompt": "Hola",
        "provider": "xyz"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["response"] is None
    assert "Unsupported provider" in data["error"]

def test_post_ask_ai_groq_existing():
    response = client.post("/ask_ai", json={
        "prompt": "Hola",
        "provider": "groq",
        "model": "llama3-70b-8192"
    })
    assert response.status_code == 200
    data = response.json()
    # Si hay error, el test falla con mensaje
    assert data["error"] is None, f"Expected success but got error: {data['error']}"
    assert data["response"] is not None

def test_post_ask_ai_huggingface_existing():
    response = client.post("/ask_ai", json={
        "prompt": "Hola",
        "provider": "huggingface",
        "model": "google/flan-t5-base"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is None, f"Expected success but got error: {data['error']}"
    assert data["response"] is not None