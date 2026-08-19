import os
import pytest
from services.ai import AI


@pytest.mark.integration
class TestOllamaIntegration:
    def test_ollama_real_connection(self):
        ai = AI()
        
        try:
            result = ai.ask_ai(
                "Say hello in one word",
                provider="local",
                model="llama2"
            )
            
            assert result is not None
            assert len(result) > 0
            print(f"Ollama response: {result}")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
    
    def test_ollama_with_different_model(self):
        ai = AI()
        
        try:
            result = ai.ask_ai(
                "What is 2+2?",
                provider="local",
                model="qwen2.5-coder:7b"
            )
            
            assert result is not None
            assert len(result) > 0
            print(f"Ollama Qwen response: {result}")
        except Exception as e:
            pytest.skip(f"Ollama Qwen model not available: {e}")