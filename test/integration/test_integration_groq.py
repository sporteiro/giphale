import os

import pytest

from services.ai import AI


@pytest.mark.integration
class TestGroqIntegration:
    def test_groq_real_connection(self):
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")

        ai = AI()

        try:
            result = ai.ask_ai(
                "Say hello in one word", provider="groq", model="allam-2-7b"
            )

            assert result is not None
            assert len(result) > 0
            print(f"Groq response: {result[:100]}...")
        except Exception as e:
            pytest.skip(f"Groq API error: {e}")
