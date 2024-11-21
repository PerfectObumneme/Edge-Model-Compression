import pytest
import numpy as np
import tensorflow as tf
from src.compression.quantization import apply_quantization_to_layer

class TestQuantization:
    @pytest.fixture
    def model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(10)
        ])
    
    def test_minmax_quantization(self, model):
        config = {'method': 'minmax', 'bits': 8}
        original_weights = [w.numpy() for w in model.layers[0].weights]
        
        apply_quantization_to_layer(model, 'conv2d', config)
        
        quantized_weights = model.layers[0].get_weights()
        
        # Check if values are quantized (should have fewer unique values)
        for orig, quant in zip(original_weights, quantized_weights):
            assert len(np.unique(quant)) < len(np.unique(orig))
    
    def test_symmetric_quantization(self, model):
        config = {'method': 'symmetric', 'bits': 8}
        original_weights = [w.numpy() for w in model.layers[0].weights]
        
        apply_quantization_to_layer(model, 'conv2d', config)
        
        quantized_weights = model.layers[0].get_weights()
        
        # Check symmetry around zero
        for w in quantized_weights:
            unique_vals = np.unique(np.abs(w))
            assert len(unique_vals) <= 2**(config['bits']-1)