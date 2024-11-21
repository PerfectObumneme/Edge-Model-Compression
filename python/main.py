import argparse
import json
import logging
from pathlib import Path
import tensorflow as tf
from src.compression.compressor import ModelCompressor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('compression.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def load_pretrained_model(model_name='resnet50', input_shape=(224, 224, 3)):
    """Load a pretrained model from Keras applications."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading pretrained {model_name}")
    
    model_options = {
        # Standard models
        'resnet50': tf.keras.applications.ResNet50,
        'resnet101': tf.keras.applications.ResNet101,
        'mobilenet': tf.keras.applications.MobileNet,
        'mobilenetv2': tf.keras.applications.MobileNetV2,
        'efficientnetb0': tf.keras.applications.EfficientNetB0,
        'efficientnetb1': tf.keras.applications.EfficientNetB1,
        'vgg16': tf.keras.applications.VGG16,
        'vgg19': tf.keras.applications.VGG19,
    }
    
    if model_name.lower() not in model_options:
        raise ValueError(f"Unsupported model: {model_name}. "
                       f"Available models: {list(model_options.keys())}")
    
    model_class = model_options[model_name.lower()]
    
    # Load model with pretrained weights
    model = model_class(
        include_top=True,  # Include classification layers
        weights='imagenet',  # Use pretrained ImageNet weights
        input_shape=input_shape,
        classes=1000  # ImageNet classes
    )
    
    logger.info(f"Loaded {model_name} successfully")
    logger.info(f"Model summary:")
    logger.info(f"- Input shape: {input_shape}")
    logger.info(f"- Number of layers: {len(model.layers)}")
    logger.info(f"- Total parameters: {model.count_params():,}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Model Compression CLI')
    parser.add_argument('--model_name', type=str, default='resnet50',
                      choices=['resnet50', 'resnet101', 'mobilenet', 
                              'mobilenetv2', 'efficientnetb0', 'efficientnetb1',
                              'vgg16', 'vgg19'],
                      help='Pretrained model to use')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save compressed model')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.model_name}")
    model = load_pretrained_model(args.model_name)
    
    # Initialize compressor
    compressor = ModelCompressor(model)
    
    # Apply compression
    logger.info("Applying compression...")
    compressor.compress_model(
        pruning_config=config['pruning_config'],
        quantization_config=config['quantization_config']
    )
    
    # Convert and save model
    logger.info("Converting to TFLite...")
    tflite_model = compressor.convert_to_tflite(
        optimization_level=config.get('optimization_level', 'LATENCY')
    )
    
    # Save compressed model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Save metrics
    metrics_path = output_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump(compressor.metrics.to_dict(), f, indent=2)
    
    logger.info(f"Compressed model saved to {output_path}")
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()