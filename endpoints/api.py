import os

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from schemas.ai_prompt import PromptRequest
from services.ai import AI

app = FastAPI()

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
rag_sources_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "rag_sources"
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/interface")
def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/rag_sources")
def get_rag_sources():
    if not os.path.exists(rag_sources_dir):
        return {"sources": []}

    sources = []
    for item in os.listdir(rag_sources_dir):
        item_path = os.path.join(rag_sources_dir, item)
        if os.path.isdir(item_path):
            sources.append(item)

    return {"sources": sources}


@app.get("/ai_query")
def get_info(request: PromptRequest = Depends()) -> dict:
    if not request.prompt:
        return {"response": "enter a prompt"}
    ai = AI()
    response = ai.ask_ai(request.prompt)
    return {"response": response}


@app.post("/ask_ai")
def post_ai_query(request: PromptRequest) -> dict:
    ai = AI()
    try:
        response = ai.ask_ai(
            request.prompt,
            request.provider,
            request.model,
            request.use_rag,
            request.rag_source,
        )
        return {"response": response, "error": None}
    except ValueError as e:
        return {"response": None, "error": str(e)}
    except RuntimeError as e:
        return {"response": None, "error": str(e)}
    except Exception as e:
        return {"response": None, "error": f"Unexpected error: {str(e)}"}
