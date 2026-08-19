from fastapi import FastAPI, Depends, HTTPException
from services.ai import AI
from schemas.ai_prompt import PromptRequest
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")



@app.get("/interface")
def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/ai_query")
def get_info(request: PromptRequest = Depends()) -> dict:
    if not request.prompt:
        return {'response': 'enter a prompt'}
    ai = AI()
    response = ai.ask_ai(request.prompt)
    return {'response': response}

@app.post("/ask_ai")
def post_ai_query(request: PromptRequest) -> dict:
    ai = AI()
    try:
        response = ai.ask_ai(request.prompt, request.provider, request.model)
        return {"response": response, "error": None}
    except ValueError as e:
        return {"response": None, "error": str(e)}
    except RuntimeError as e:
        return {"response": None, "error": str(e)}
    except Exception as e:
        return {"response": None, "error": f"Unexpected error: {str(e)}"}