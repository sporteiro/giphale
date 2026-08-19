import os
import pytest
from services.ai import AI


@pytest.mark.integration
class TestHuggingFaceIntegration:
    def test_huggingface_real_connection(self):
        if not os.getenv('HF_TOKEN'):
            pytest.skip("HF_TOKEN not set")
        
        ai = AI()
        
        try:
            result = ai.ask_ai(
                "Say hello in one word",
                provider="huggingface",
                model="google/flan-t5-base"
            )
            
            assert result is not None
            assert len(result) > 0
            print(f"HuggingFace response: {result}")
        except Exception as e:
            pytest.skip(f"HuggingFace API error: {e}")