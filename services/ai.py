import os
from dotenv import load_dotenv
import requests

load_dotenv()

class AI:
    def __init__(self):
        self.default_model = os.getenv('LLM_MODEL')
        self.openrouter_url = os.getenv('OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434/v1/chat/completions')
        self.groq_url = os.getenv('GROQ_URL', 'https://api.groq.com/openai/v1/chat/completions')
        self.groq_key = os.getenv('GROQ_API_KEY')
        self.hf_url = os.getenv('HF_URL', 'https://api-inference.huggingface.co/models')
        self.hf_key = os.getenv('HF_TOKEN')

    def ask_ai(self, prompt: str, provider: str = None, model: str = None) -> str:
        provider = provider or 'openrouter'
        model = model or self.default_model

        if provider == 'openrouter':
            return self._ask_openrouter(prompt, model)
        elif provider == 'local':
            if not model:
                raise ValueError("Model is required for local provider")
            return self._ask_ollama(prompt, model)
        elif provider == 'groq':
            return self._ask_groq(prompt, model)
        elif provider == 'huggingface':
            return self._ask_huggingface(prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_ai_api(self, url: str, headers: dict, payload: dict, error_prefix: str = "API error") -> str:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise RuntimeError(f"{error_prefix}: Resource not found")
            raise RuntimeError(f"{error_prefix}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{error_prefix}: {str(e)}")

    def _ask_openrouter(self, prompt: str, model: str) -> str:
        if not self.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        return self._call_ai_api(self.openrouter_url, headers, payload, "OpenRouter error")

    def _ask_ollama(self, prompt: str, model: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        return self._call_ai_api(self.ollama_url, headers, payload, "Ollama error")

    def _ask_groq(self, prompt: str, model: str) -> str:
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        return self._call_ai_api(self.groq_url, headers, payload, "Groq error")

    def _ask_huggingface(self, prompt: str, model: str) -> str:
        if not self.hf_key:
            raise ValueError("HF_TOKEN not set")
        url = f"{self.hf_url}/{model}"  # Hugging Face inference API uses model in URL
        headers = {
            "Authorization": f"Bearer {self.hf_key}",
            "Content-Type": "application/json"
        }
        # Hugging Face expects a different payload structure
        payload = {
            "inputs": prompt,
            "parameters": {"return_full_text": False}
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # HF returns list of dicts, we extract the generated text
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"]
            else:
                raise RuntimeError("Unexpected response format from Hugging Face")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Hugging Face error: {str(e)}")