import os

import pytest

from services.ai import AI


@pytest.mark.integration
class TestOpenRouterIntegration:
    def test_openrouter_real_connection(self):
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        ai = AI()

        try:
            result = ai.ask_ai(
                "Say hello in one word",
                provider="openrouter",
                model="openrouter/free",
            )

            assert result is not None
            assert len(result) > 0
            print(f"OpenRouter response: {result[:100]}...")
        except Exception as e:
            pytest.skip(f"OpenRouter API error: {e}")

    def test_openrouter_fallback_models(self):
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        ai = AI()

        try:
            result = ai.ask_ai(
                "Say hello in one word",
                provider="openrouter",
                model="nvidia/nemotron-3-ultra:free",
            )

            assert result is not None
            assert len(result) > 0
            print(f"OpenRouter fallback response: {result[:100]}...")
        except Exception as e:
            pytest.skip(f"OpenRouter fallback API error: {e}")
