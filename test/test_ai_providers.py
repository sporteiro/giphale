import os
import pytest
from unittest.mock import Mock, patch
from services.ai import (
    AIProviderConfig,
    OpenAICompatibleProvider,
    OllamaProvider,
    HuggingFaceProvider,
    AIService,
    AI
)


class TestOpenAICompatibleProvider:
    def test_build_payload(self):
        config = AIProviderConfig(url="http://test.com")
        provider = OpenAICompatibleProvider(config)
        payload = provider.build_payload("test prompt", "gpt-3")
        
        assert payload == {
            "model": "gpt-3",
            "messages": [{"role": "user", "content": "test prompt"}]
        }
    
    def test_extract_response(self):
        config = AIProviderConfig(url="http://test.com")
        provider = OpenAICompatibleProvider(config)
        response_data = {
            "choices": [{"message": {"content": "test response"}}]
        }
        
        result = provider.extract_response(response_data)
        assert result == "test response"


class TestOllamaProvider:
    def test_build_payload(self):
        config = AIProviderConfig(url="http://localhost:11434/api/chat")
        provider = OllamaProvider(config)
        payload = provider.build_payload("test prompt", "llama2")
        
        assert payload == {
            "model": "llama2",
            "messages": [{"role": "user", "content": "test prompt"}],
            "stream": False
        }
    
    def test_extract_response_message_format(self):
        config = AIProviderConfig(url="http://localhost:11434/api/chat")
        provider = OllamaProvider(config)
        response_data = {
            "message": {"content": "ollama response"}
        }
        
        result = provider.extract_response(response_data)
        assert result == "ollama response"
    
    def test_extract_response_response_format(self):
        config = AIProviderConfig(url="http://localhost:11434/api/chat")
        provider = OllamaProvider(config)
        response_data = {
            "response": "ollama response"
        }
        
        result = provider.extract_response(response_data)
        assert result == "ollama response"
    
    def test_extract_response_invalid_format(self):
        config = AIProviderConfig(url="http://localhost:11434/api/chat")
        provider = OllamaProvider(config)
        response_data = {"invalid": "format"}
        
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            provider.extract_response(response_data)


class TestHuggingFaceProvider:
    def test_build_payload(self):
        config = AIProviderConfig(url="https://api-inference.huggingface.co/models")
        provider = HuggingFaceProvider(config)
        payload = provider.build_payload("test prompt", "gpt2")
        
        assert payload == {
            "inputs": "test prompt",
            "parameters": {"return_full_text": False}
        }
    
    def test_extract_response(self):
        config = AIProviderConfig(url="https://api-inference.huggingface.co/models")
        provider = HuggingFaceProvider(config)
        response_data = [{"generated_text": "hf response"}]
        
        result = provider.extract_response(response_data)
        assert result == "hf response"
    
    def test_extract_response_invalid_format(self):
        config = AIProviderConfig(url="https://api-inference.huggingface.co/models")
        provider = HuggingFaceProvider(config)
        response_data = {"invalid": "format"}
        
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            provider.extract_response(response_data)


class TestAIService:
    @patch.dict(os.environ, {
        'LLM_MODEL': 'default-model',
        'OPENROUTER_API_KEY': 'test-key',
        'GROQ_API_KEY': 'test-key',
        'HF_TOKEN': 'test-token'
    })
    def test_initialization(self):
        service = AIService()
        
        assert 'openrouter' in service.providers
        assert 'local' in service.providers
        assert 'groq' in service.providers
        assert 'huggingface' in service.providers
    
    @patch.dict(os.environ, {
        'LLM_MODEL': 'default-model',
        'OPENROUTER_API_KEY': 'test-key'
    })
    def test_unsupported_provider(self):
        service = AIService()
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            service.ask_ai("test", provider="invalid")
    
    @patch.dict(os.environ, {
        'LLM_MODEL': 'default-model',
        'OPENROUTER_API_KEY': 'test-key'
    })
    def test_local_provider_requires_model(self):
        service = AIService()
        
        with pytest.raises(ValueError, match="Model is required for local provider"):
            service.ask_ai("test", provider="local")


class TestAI:
    @patch.dict(os.environ, {
        'LLM_MODEL': 'default-model',
        'OPENROUTER_API_KEY': 'test-key'
    })
    def test_ai_facade(self):
        ai = AI()
        
        assert hasattr(ai, 'service')
        assert isinstance(ai.service, AIService)