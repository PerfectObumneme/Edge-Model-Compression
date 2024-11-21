from typing import List
import numpy as np
import tensorflow as tf
from typing import Dict, Any

def apply_quantization_to_layer(model: tf.keras.Model,
    layer_name: str,
    config: Dict[str, Any]) -> None:
    """Apply quantization to a specific layer."""
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()
    bits = config.get('bits', 8)
    method = config.get('method', 'minmax')
    if method == 'minmax':
        quantized_weights = _minmax_quantization(weights, bits)
    elif method == 'symmetric':
        quantized_weights = _symmetric_quantization(weights, bits)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    layer.set_weights(quantized_weights)
    
def _minmax_quantization(weights: List[np.ndarray], bits: int) -> List[np.ndarray]:
    """Apply min-max quantization."""
    quantized_weights = []
    for w in weights:
        w_min, w_max = w.min(), w.max()
        scale = (2**bits - 1) / (w_max - w_min)
        w_q = np.round((w - w_min) * scale) / scale + w_min
        quantized_weights.append(w_q)
    return quantized_weights

def _symmetric_quantization(weights: List[np.ndarray], bits: int) -> List[np.ndarray]:
    """Apply symmetric quantization."""
    quantized_weights = []
    for w in weights:
        w_max = max(abs(w.min()), abs(w.max()))
        scale = (2**(bits-1) - 1) / w_max
        w_q = np.round(w * scale) / scale
        quantized_weights.append(w_q)
    return quantized_weights