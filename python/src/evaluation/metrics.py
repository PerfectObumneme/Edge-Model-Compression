from typing import Dict

class CompressionMetrics:
    def __init__(self):
        self.original_size = 0
        self.compressed_size = 0
        self.layer_metrics = {}
        
    def add_layer_metrics(self, layer_name: str, original_params: int):
        """Initialize metrics for a layer."""
        self.layer_metrics[layer_name] = {
            'original_params': original_params,
            'compressed_params': original_params
        }

    def update_layer_metrics(self, layer_name: str, compressed_params: int):
        """Update metrics for a layer after compression."""
        self.layer_metrics[layer_name]['compressed_params'] = compressed_params

    def set_original_size(self, size: int):
        """Set total original model size in bytes."""
        self.original_size = size

    def set_compressed_size(self, size: int):
        """Set total compressed model size in bytes."""
        self.compressed_size = size

    def get_compression_ratio(self) -> float:
        """Calculate overall compression ratio."""
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            'original_size_mb': self.original_size / (1024 * 1024),
            'compressed_size_mb': self.compressed_size / (1024 * 1024),
            'compression_ratio': self.get_compression_ratio(),
            'layer_metrics': self.layer_metrics
        }
