import os
import pytest
from unittest.mock import patch, Mock
from services.ai import AI


class TestGroqModels:
    @patch.dict(os.environ, {
        'GROQ_API_KEY': 'test-key',
        'GROQ_URL': 'https://api.groq.com/openai/v1/chat/completions'
    })
    @patch('services.ai.requests.post')
    def test_groq_connection_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response from Groq"}}]
        }
        mock_post.return_value = mock_response
        
        ai = AI()
        result = ai.ask_ai("test prompt", provider="groq", model="llama3-70b-8192")
        
        assert result == "Test response from Groq"
        mock_post.assert_called_once()
    
    @patch.dict(os.environ, {
        'GROQ_API_KEY': 'test-key',
        'GROQ_URL': 'https://api.groq.com/openai/v1/chat/completions'
    })
    @patch('services.ai.requests.post')
    def test_groq_missing_api_key(self, mock_post):
        with patch.dict(os.environ, {}, clear=True):
            ai = AI()
            
            with pytest.raises(ValueError, match="GROQ_API_KEY not set"):
                ai.ask_ai("test prompt", provider="groq")
    
    @patch.dict(os.environ, {
        'GROQ_API_KEY': 'test-key',
        'GROQ_URL': 'https://api.groq.com/openai/v1/chat/completions'
    })
    @patch('services.ai.requests.post')
    def test_groq_network_error(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        
        ai = AI()
        
        with pytest.raises(RuntimeError, match="Network error"):
            ai.ask_ai("test prompt", provider="groq")