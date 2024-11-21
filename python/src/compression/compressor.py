from typing import Dict, List, Tuple, Optional, Callable, Any
import tensorflow as tf
import numpy as np
import logging
from .pruning import apply_pruning_to_layer
from .quantization import apply_quantization_to_layer
from src.evaluation.metrics import CompressionMetrics

class ModelCompressor:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.metrics = CompressionMetrics()
        self.logger = logging.getLogger(__name__)
        self._store_original_weights()
        
    def _store_original_weights(self):
        """Store original weights for reference and metrics."""
        self.original_weights = {}
        total_params = 0
        for layer in self.model.layers:
            if len(layer.weights) > 0:
                self.original_weights[layer.name] = [w.numpy() for w in layer.weights]
                layer_params = np.sum([np.prod(w.shape) for w in layer.weights])
                total_params += layer_params
                self.metrics.add_layer_metrics(layer.name, original_params=layer_params)
        
        self.metrics.set_original_size(total_params * 4)  # Assuming float32

    def compress_model(self, pruning_config: Dict[str, Dict], quantization_config: Dict[str, Dict]) -> None:
        """Apply compression according to configurations."""
        # Apply pruning first
        for layer_name, config in pruning_config.items():
            apply_pruning_to_layer(self.model, layer_name, config)
            
        # Then apply quantization
        for layer_name, config in quantization_config.items():
            apply_quantization_to_layer(self.model, layer_name, config)
        
        # Update final metrics
        self._update_compression_metrics()

    def _update_compression_metrics(self):
        """Update compression metrics after model modification."""
        compressed_size = 0
        for layer in self.model.layers:
            if len(layer.weights) > 0:
                layer_params = np.sum([np.prod(w.shape) for w in layer.weights])
                compressed_size += layer_params
                self.metrics.update_layer_metrics(layer.name, compressed_params=layer_params)
        
        self.metrics.set_compressed_size(compressed_size * 4)

    def convert_to_tflite(self, representative_dataset: Optional[Callable] = None, optimization_level: str = 'LATENCY') -> bytes:
        """Convert to TFLite with optimization options."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if optimization_level == 'LATENCY':
            converter.optimizations = {tf.lite.Optimize.OPTIMIZE_FOR_LATENCY}
        elif optimization_level == 'SIZE':
            converter.optimizations = {tf.lite.Optimize.OPTIMIZE_FOR_SIZE}
        
        if representative_dataset:
            converter.representative_dataset = representative_dataset # type: ignore
            converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        return converter.convert()