"""Unit tests for BaseModelAdapter abstract class and related models."""

import pytest
from pydantic import ValidationError

from src.adapters.base_adapter import (
    BaseModelAdapter,
    InferenceRequest,
    InferenceResponse,
)


class TestInferenceRequest:
    """Test InferenceRequest model validation."""
    
    def test_valid_request(self):
        """Test creating a valid inference request."""
        request = InferenceRequest(
            prompt="What is the capital of France?",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )
        assert request.prompt == "What is the capital of France?"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop_sequences is None
    
    def test_request_with_defaults(self):
        """Test that default values are applied correctly."""
        request = InferenceRequest(prompt="Test prompt")
        assert request.max_tokens == 1024
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop_sequences is None
    
    def test_request_with_stop_sequences(self):
        """Test request with stop sequences."""
        request = InferenceRequest(
            prompt="Test",
            stop_sequences=["END", "\n\n"]
        )
        assert request.stop_sequences == ["END", "\n\n"]
    
    def test_invalid_max_tokens(self):
        """Test that max_tokens must be positive."""
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="Test", max_tokens=0)
        
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="Test", max_tokens=-1)
    
    def test_invalid_temperature(self):
        """Test that temperature must be in [0, 1]."""
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="Test", temperature=-0.1)
        
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="Test", temperature=1.1)
    
    def test_invalid_top_p(self):
        """Test that top_p must be in [0, 1]."""
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="Test", top_p=-0.1)
        
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="Test", top_p=1.1)
    
    def test_boundary_values(self):
        """Test boundary values for temperature and top_p."""
        request = InferenceRequest(
            prompt="Test",
            temperature=0.0,
            top_p=1.0
        )
        assert request.temperature == 0.0
        assert request.top_p == 1.0
        
        request = InferenceRequest(
            prompt="Test",
            temperature=1.0,
            top_p=0.0
        )
        assert request.temperature == 1.0
        assert request.top_p == 0.0


class TestInferenceResponse:
    """Test InferenceResponse model."""
    
    def test_valid_response(self):
        """Test creating a valid inference response."""
        response = InferenceResponse(
            text="Paris is the capital of France.",
            input_tokens=10,
            output_tokens=8,
            latency_ms=250.5,
            model_id="claude-3-sonnet",
            finish_reason="stop"
        )
        assert response.text == "Paris is the capital of France."
        assert response.input_tokens == 10
        assert response.output_tokens == 8
        assert response.latency_ms == 250.5
        assert response.model_id == "claude-3-sonnet"
        assert response.finish_reason == "stop"
    
    def test_response_with_length_finish(self):
        """Test response that finished due to length limit."""
        response = InferenceResponse(
            text="Truncated response...",
            input_tokens=5,
            output_tokens=100,
            latency_ms=500.0,
            model_id="test-model",
            finish_reason="length"
        )
        assert response.finish_reason == "length"
    
    def test_response_serialization(self):
        """Test that response can be serialized to dict."""
        response = InferenceResponse(
            text="Test",
            input_tokens=1,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test",
            finish_reason="stop"
        )
        data = response.model_dump()
        assert data["text"] == "Test"
        assert data["input_tokens"] == 1
        assert data["output_tokens"] == 1


class TestBaseModelAdapter:
    """Test BaseModelAdapter abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseModelAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelAdapter()
    
    def test_subclass_must_implement_all_methods(self):
        """Test that subclasses must implement all abstract methods."""
        
        class IncompleteAdapter(BaseModelAdapter):
            """Incomplete adapter missing some methods."""
            pass
        
        with pytest.raises(TypeError):
            IncompleteAdapter()
    
    def test_complete_subclass_can_be_instantiated(self):
        """Test that a complete implementation can be instantiated."""
        
        class CompleteAdapter(BaseModelAdapter):
            """Complete adapter implementing all methods."""
            
            async def invoke(self, request: InferenceRequest) -> InferenceResponse:
                return InferenceResponse(
                    text="test",
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=100.0,
                    model_id="test",
                    finish_reason="stop"
                )
            
            async def invoke_stream(self, request: InferenceRequest):
                yield "test"
            
            async def list_models(self):
                return []
            
            async def get_model_info(self, model_id: str):
                return None
            
            async def validate_connection(self) -> bool:
                return True
            
            def supports_streaming(self) -> bool:
                return True
            
            def supports_fine_tuning(self, model_id: str) -> bool:
                return False
        
        adapter = CompleteAdapter()
        assert adapter is not None
        assert isinstance(adapter, BaseModelAdapter)
    
    def test_adapter_interface_structure(self):
        """Test that adapter interface has correct method signatures."""
        
        class TestAdapter(BaseModelAdapter):
            """Test adapter for interface validation."""
            
            async def invoke(self, request: InferenceRequest) -> InferenceResponse:
                return InferenceResponse(
                    text=f"Response to: {request.prompt}",
                    input_tokens=len(request.prompt.split()),
                    output_tokens=5,
                    latency_ms=100.0,
                    model_id="test-model",
                    finish_reason="stop"
                )
            
            async def invoke_stream(self, request: InferenceRequest):
                for word in ["Hello", " ", "World"]:
                    yield word
            
            async def list_models(self):
                return []
            
            async def get_model_info(self, model_id: str):
                return None
            
            async def validate_connection(self) -> bool:
                return True
            
            def supports_streaming(self) -> bool:
                return True
            
            def supports_fine_tuning(self, model_id: str) -> bool:
                return model_id == "fine-tunable-model"
        
        adapter = TestAdapter()
        
        # Test that adapter has all required methods
        assert hasattr(adapter, 'invoke')
        assert hasattr(adapter, 'invoke_stream')
        assert hasattr(adapter, 'list_models')
        assert hasattr(adapter, 'get_model_info')
        assert hasattr(adapter, 'validate_connection')
        assert hasattr(adapter, 'supports_streaming')
        assert hasattr(adapter, 'supports_fine_tuning')
        
        # Test synchronous methods
        assert adapter.supports_streaming() is True
        assert adapter.supports_fine_tuning("fine-tunable-model") is True
        assert adapter.supports_fine_tuning("other-model") is False
