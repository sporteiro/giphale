import os
import pytest
from unittest.mock import patch, Mock
from services.ai import AI


class TestOpenRouterModels:
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENROUTER_URL': 'https://openrouter.ai/api/v1/chat/completions'
    })
    @patch('services.ai.requests.post')
    def test_openrouter_connection_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response from OpenRouter"}}]
        }
        mock_post.return_value = mock_response
        
        ai = AI()
        result = ai.ask_ai("test prompt", provider="openrouter", model="gpt-3")
        
        assert result == "Test response from OpenRouter"
        mock_post.assert_called_once()
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENROUTER_URL': 'https://openrouter.ai/api/v1/chat/completions'
    })
    @patch('services.ai.requests.post')
    def test_openrouter_missing_api_key(self, mock_post):
        with patch.dict(os.environ, {}, clear=True):
            ai = AI()
            
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
                ai.ask_ai("test prompt", provider="openrouter")
    
    @patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENROUTER_URL': 'https://openrouter.ai/api/v1/chat/completions'
    })
    @patch('services.ai.requests.post')
    def test_openrouter_network_error(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        
        ai = AI()
        
        with pytest.raises(RuntimeError, match="Network error"):
            ai.ask_ai("test prompt", provider="openrouter")