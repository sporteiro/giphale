import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIProviderConfig:
    url: str
    api_key: str = None
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class AIProvider(ABC):
    def __init__(self, config: AIProviderConfig):
        self.config = config
    
    @abstractmethod
    def build_payload(self, prompt: str, model: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def extract_response(self, response_data: Dict[str, Any]) -> str:
        pass
    
    def call_api(self, prompt: str, model: str) -> str:
        headers = self.config.headers.copy()
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = self.build_payload(prompt, model)
        
        logger.info(f"Calling API: {self.config.url}")
        logger.debug(f"Payload: {payload}")
        
        try:
            response = requests.post(
                self.config.url,
                headers=headers,
                json=payload,
                timeout=30
            )
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            result = self.extract_response(response.json())
            logger.info(f"API call successful")
            return result
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}, status: {response.status_code}")
            if response.status_code == 404:
                raise RuntimeError("Resource not found")
            raise RuntimeError(f"API error: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise RuntimeError(f"Network error: {str(e)}")


class OpenAICompatibleProvider(AIProvider):
    def build_payload(self, prompt: str, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
    
    def extract_response(self, response_data: Dict[str, Any]) -> str:
        return response_data["choices"][0]["message"]["content"]


class OllamaProvider(AIProvider):
    def build_payload(self, prompt: str, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    
    def extract_response(self, response_data: Dict[str, Any]) -> str:
        if "message" in response_data:
            return response_data["message"]["content"]
        elif "response" in response_data:
            return response_data["response"]
        else:
            raise RuntimeError("Unexpected response format from Ollama")


class HuggingFaceProvider(AIProvider):
    def build_payload(self, prompt: str, model: str) -> Dict[str, Any]:
        return {
            "inputs": prompt,
            "parameters": {"return_full_text": False}
        }
    
    def extract_response(self, response_data: Dict[str, Any]) -> str:
        if isinstance(response_data, list) and len(response_data) > 0:
            if "generated_text" in response_data[0]:
                return response_data[0]["generated_text"]
        raise RuntimeError("Unexpected response format from Hugging Face")


class AIService:
    def __init__(self):
        self.default_model = os.getenv('LLM_MODEL')
        self.providers = self._initialize_providers()
    
    def _initialize_providers(self) -> Dict[str, AIProvider]:
        return {
            'openrouter': OpenAICompatibleProvider(
                AIProviderConfig(
                    url=os.getenv('OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions'),
                    api_key=os.getenv('OPENROUTER_API_KEY'),
                    headers={"Content-Type": "application/json"}
                )
            ),
            'local': OllamaProvider(
                AIProviderConfig(
                    url=os.getenv('OLLAMA_URL', 'http://localhost:11434/api/chat'),
                    headers={"Content-Type": "application/json"}
                )
            ),
            'groq': OpenAICompatibleProvider(
                AIProviderConfig(
                    url=os.getenv('GROQ_URL', 'https://api.groq.com/openai/v1/chat/completions'),
                    api_key=os.getenv('GROQ_API_KEY'),
                    headers={"Content-Type": "application/json"}
                )
            ),
            'huggingface': HuggingFaceProvider(
                AIProviderConfig(
                    url=os.getenv('HF_URL', 'https://api-inference.huggingface.co/models'),
                    api_key=os.getenv('HF_TOKEN'),
                    headers={"Content-Type": "application/json"}
                )
            )
        }
    
    def ask_ai(self, prompt: str, provider: str = None, model: str = None) -> str:
        provider = provider or 'openrouter'
        model = model or self.default_model
        
        logger.info(f"Using provider: {provider}, model: {model}")
        
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        if provider == 'local' and not model:
            raise ValueError("Model is required for local provider")
        
        ai_provider = self.providers[provider]
        
        if provider == 'huggingface':
            ai_provider.config.url = f"{ai_provider.config.url}/{model}"
        
        return ai_provider.call_api(prompt, model)


class AI:
    def __init__(self):
        self.service = AIService()
    
    def ask_ai(self, prompt: str, provider: str = None, model: str = None) -> str:
        return self.service.ask_ai(prompt, provider, model)