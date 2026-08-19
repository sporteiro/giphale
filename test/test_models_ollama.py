import pytest
from unittest.mock import patch, Mock
from services.ai import AI


class TestOllamaModels:
    @patch.dict(os.environ, {
        'OLLAMA_URL': 'http://localhost:11434/api/chat'
    })
    @patch('services.ai.requests.post')
    def test_ollama_connection_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Test response from Ollama"}
        }
        mock_post.return_value = mock_response
        
        ai = AI()
        result = ai.ask_ai("test prompt", provider="local", model="llama2")
        
        assert result == "Test response from Ollama"
        mock_post.assert_called_once()
    
    @patch.dict(os.environ, {
        'OLLAMA_URL': 'http://localhost:11434/api/chat'
    })
    @patch('services.ai.requests.post')
    def test_ollama_model_not_found(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_post.return_value = mock_response
        
        ai = AI()
        
        with pytest.raises(RuntimeError, match="Resource not found"):
            ai.ask_ai("test prompt", provider="local", model="nonexistent")
    
    @patch.dict(os.environ, {
        'OLLAMA_URL': 'http://localhost:11434/api/chat'
    })
    @patch('services.ai.requests.post')
    def test_ollama_network_error(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        
        ai = AI()
        
        with pytest.raises(RuntimeError, match="Network error"):
            ai.ask_ai("test prompt", provider="local", model="llama2")
    
    @patch.dict(os.environ, {
        'OLLAMA_URL': 'http://localhost:11434/api/chat'
    })
    @patch('services.ai.requests.post')
    def test_ollama_alternative_response_format(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Alternative Ollama response"
        }
        mock_post.return_value = mock_response
        
        ai = AI()
        result = ai.ask_ai("test prompt", provider="local", model="llama2")
        
        assert result == "Alternative Ollama response"