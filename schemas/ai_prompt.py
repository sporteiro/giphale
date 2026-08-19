from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str | None = None
    provider: str | None = None
    model: str | None = None
    use_rag: bool = False
    rag_source: str | None = None
