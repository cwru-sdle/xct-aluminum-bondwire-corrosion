import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from pathlib import Path
import segmentation_models as sm

from models import Unet
from config import ModelConfig
from dataloader import prediction_dataset

def save_prediction(prediction: np.ndarray, filename: str, output_dir: Path):
    prediction = (prediction * 255).astype(np.uint8)
    img = Image.fromarray(prediction.squeeze(), mode='L')
    img.save(output_dir / filename)

def main():
    # set up environment
    config = ModelConfig()
    sm.set_framework('tf.keras')
    tf.random.set_seed(config.random_seed)

    # prepare data
    preprocess_input = sm.get_preprocessing(config.backbone) if config.backbone else None
    img_paths = list(config.img_dir.glob('*timestep_070*'))
    ds = prediction_dataset(
        img_paths, 
        config.batch_size,
        config.img_shape,
        preprocess_input
    )

    # define model
    if config.backbone:
        unet_model = sm.Unet(
            config.backbone, 
            encoder_weights=config.encoder_weights, 
            input_shape=config.img_shape, 
            classes=1,
            activation='sigmoid'
        )
    else:
        unet_model = Unet(
            input_shape=config.img_shape, 
            classes=1, 
            activation='sigmoid'
        ).build()
    
    # compile model
    unet_model.compile()
    unet_model.load_weights(config.save_model_path)

    for images, basenames in ds:
        predictions = unet_model.predict(images)
        
        for pred, basename in zip(predictions, basenames.numpy()):
            filename = basename.decode('utf-8')
            save_prediction(pred, filename, config.prediction_dir)
    print(f'All predictions saved to {config.prediction_dir}')

if __name__ == '__main__':
    main()
