import pytest
import tensorflow as tf
import numpy as np
from src.compression.compressor import ModelCompressor

def create_simple_model():
    """Create a simple model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model

class TestModelCompressor:
    @pytest.fixture
    def model(self):
        return create_simple_model()
    
    @pytest.fixture
    def compressor(self, model):
        return ModelCompressor(model)
    
    def test_initialization(self, compressor):
        assert compressor.metrics is not None
        assert len(compressor.original_weights) > 0
    
    def test_compression(self, compressor):
        pruning_config = {
            'conv2d': {'ratio': 0.5, 'method': 'magnitude'},
            'conv2d_1': {'ratio': 0.3, 'method': 'structured'}
        }
        
        quantization_config = {
            'conv2d': {'bits': 8, 'method': 'minmax'},
            'conv2d_1': {'bits': 6, 'method': 'symmetric'}
        }
        
        compressor.compress_model(pruning_config, quantization_config)
        metrics = compressor.metrics.to_dict()
        
        assert metrics['compressed_size_mb'] < metrics['original_size_mb']
        assert metrics['compression_ratio'] > 1.0
        