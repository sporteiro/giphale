import os
import pytest
from unittest.mock import patch, Mock
from services.ai import AI


class TestHuggingFaceModels:
    @patch.dict(os.environ, {
        'HF_TOKEN': 'test-token',
        'HF_URL': 'https://api-inference.huggingface.co/models'
    })
    @patch('services.ai.requests.post')
    def test_huggingface_connection_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"generated_text": "Test response from HuggingFace"}
        ]
        mock_post.return_value = mock_response
        
        ai = AI()
        result = ai.ask_ai("test prompt", provider="huggingface", model="google/flan-t5-base")
        
        assert result == "Test response from HuggingFace"
        mock_post.assert_called_once()
    
    @patch.dict(os.environ, {
        'HF_TOKEN': 'test-token',
        'HF_URL': 'https://api-inference.huggingface.co/models'
    })
    @patch('services.ai.requests.post')
    def test_huggingface_missing_token(self, mock_post):
        with patch.dict(os.environ, {}, clear=True):
            ai = AI()
            
            with pytest.raises(ValueError, match="HF_TOKEN not set"):
                ai.ask_ai("test prompt", provider="huggingface")
    
    @patch.dict(os.environ, {
        'HF_TOKEN': 'test-token',
        'HF_URL': 'https://api-inference.huggingface.co/models'
    })
    @patch('services.ai.requests.post')
    def test_huggingface_network_error(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        
        ai = AI()
        
        with pytest.raises(RuntimeError, match="Network error"):
            ai.ask_ai("test prompt", provider="huggingface")
    
    @patch.dict(os.environ, {
        'HF_TOKEN': 'test-token',
        'HF_URL': 'https://api-inference.huggingface.co/models'
    })
    @patch('services.ai.requests.post')
    def test_huggingface_invalid_response_format(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "format"}
        mock_post.return_value = mock_response
        
        ai = AI()
        
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            ai.ask_ai("test prompt", provider="huggingface")