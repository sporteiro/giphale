http://localhost:8000/ai_query?prompt=hola


curl -X POST http://127.0.0.1:8000/ask_ai \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explica qué es una API REST en una frase"}'


curl -X POST http://127.0.0.1:8000/ask_ai \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hola", "provider": "local", "model": "qwen2.5-coder:7b"}'