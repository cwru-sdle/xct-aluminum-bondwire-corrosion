import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from skimage.io import imsave
import segmentation_models as sm

from models import Unet
from config import ModelConfig
from dataloader import prediction_dataset

def save_prediction(prediction: np.ndarray, filename: str, output_dir: Path):
    prediction = (prediction * 255).astype(np.uint8)
    imsave(output_dir / filename, prediction, check_contrast=False)

def main():
    # set up environment
    config = ModelConfig()
    sm.set_framework('tf.keras')

    # prepare data
    preprocess_input = sm.get_preprocessing(config.backbone) if config.encoder_weights else None
    img_paths = list(config.img_dir.glob('*'))
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

    for images, paths in ds:
        predictions = unet_model.predict(images)
        
        for pred, path in zip(predictions, paths.numpy()):
            filename = os.path.basename(path.decode('utf-8'))
            save_prediction(pred, filename, config.prediction_dir)

    print(f'All predictions saved to {config.prediction_dir}')

if __name__ == '__main__':
    main()
