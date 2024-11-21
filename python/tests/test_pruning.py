import pytest
import numpy as np
import tensorflow as tf
from src.compression.pruning import apply_pruning_to_layer

class TestPruning:
    @pytest.fixture
    def model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(10)
        ])
    
    def test_magnitude_pruning(self, model):
        config = {'method': 'magnitude', 'ratio': 0.5}
        original_params = np.sum([np.prod(w.shape) for w in model.layers[0].weights])
        
        apply_pruning_to_layer(model, 'conv2d', config)
        
        pruned_weights = model.layers[0].get_weights()
        zeros = np.sum([np.sum(w == 0) for w in pruned_weights])
        total = np.sum([np.prod(w.shape) for w in pruned_weights])
        
        assert zeros / total >= 0.4  # Allow for some tolerance

    def test_structured_pruning(self, model):
        config = {'method': 'structured', 'ratio': 0.5}
        original_shape = model.layers[0].weights[0].shape
        
        apply_pruning_to_layer(model, 'conv2d', config)
        
        new_shape = model.layers[0].weights[0].shape
        assert new_shape[-1] < original_shape[-1]