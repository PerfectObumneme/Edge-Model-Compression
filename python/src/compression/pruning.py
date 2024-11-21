from typing import List
import numpy as np
import tensorflow as tf
from typing import Dict, Any

def apply_pruning_to_layer(model: tf.keras.Model, layer_name: str, config: Dict[str, Any]) -> None:
    """Apply pruning to a specific layer."""
    layer   = model.get_layer(layer_name)
    weights = layer.get_weights()
    method  = config.get('method', 'magnitude')
    ratio   = config.get('ratio', 0.5)
    
    if method == 'magnitude':
        pruned_weights = _magnitude_pruning(weights, ratio)
    elif method == 'structured':
        pruned_weights = _structured_pruning(weights, ratio)
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    layer.set_weights(pruned_weights)

def _magnitude_pruning(weights: List[np.ndarray], ratio: float) -> List[np.ndarray]:
    """Apply magnitude-based pruning."""
    pruned_weights = []
    for w in weights:
        threshold = np.percentile(np.abs(w), ratio * 100)
        mask = np.abs(w) > threshold
        pruned_weights.append(w * mask)
    return pruned_weights

def _structured_pruning(weights: List[np.ndarray], ratio: float) -> List[np.ndarray]:
    """Apply structured (channel) pruning."""
    pruned_weights = []
    for w in weights:
        if len(w.shape) == 4:  # Conv layer
            channel_l2_norm = np.sqrt(np.sum(w ** 2, axis=(0, 1, 2)))
            num_channels = len(channel_l2_norm)
            channels_to_keep = int(num_channels * (1 - ratio))
            threshold = np.sort(channel_l2_norm)[num_channels - channels_to_keep]
            channel_mask = channel_l2_norm >= threshold
            pruned_weights.append(w[:, :, :, channel_mask])
        else:
            pruned_weights.append(w)  # No pruning for non-conv layers
    return pruned_weights