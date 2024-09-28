import os
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from glob import glob

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import load_model

@dataclass
class PredictionConfig:
    input_dir: str = '../data/images'
    output_dir: str = '../data/predictions'
    model_path: str = '../models/unet_100epochs_seresnext101_aug.keras'
    backbone: str = 'seresnext101'
    batch_size: int = 8

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f'Error setting GPU memory growth: {e}')

def read_image(path: str, preprocess_func=None) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if preprocess_func is not None:
        return preprocess_func(image)
    return image

def save_prediction(prediction: tf.Tensor, output_path: str):
    prediction = tf.where(prediction > 0.5, 1.0, 0.0)
    prediction = tf.squeeze(prediction)
    prediction = tf.cast(prediction * 255, tf.uint8)
    encoded_mask = tf.io.encode_png(prediction)
    tf.io.write_file(output_path, encoded_mask)

def prepare_dataset(image_paths: List[str], preprocess_func, batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: read_image(x, preprocess_func), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def main(config: PredictionConfig):
    set_gpu_memory_growth()
    sm.set_framework('tf.keras')

    # load model
    custom_objects = {
        'binary_focal_jaccard_loss': sm.losses.binary_focal_jaccard_loss,
        'precision': sm.metrics.precision,
        'recall': sm.metrics.recall,
        'iou_score': sm.metrics.iou_score
    }
    model = load_model(config.model_path, custom_objects=custom_objects)

    # prepare dataset
    preprocess_input = sm.get_preprocessing(config.backbone)
    image_paths = glob(os.path.join(config.input_dir, '*.png'))
    dataset = prepare_dataset(image_paths, preprocess_input, config.batch_size)

    # generate predictions
    for batch, paths in zip(dataset, tf.data.Dataset.from_tensor_slices(image_paths).batch(config.batch_size)):
        predictions = model.predict(batch)
        
        for pred, path in zip(predictions, paths):
            output_path = os.path.join(config.output_dir, f'pred_{os.path.basename(path.numpy().decode())}')
            save_prediction(pred, output_path)
            print(f'Prediction saved for {os.path.basename(path.numpy().decode())}')

if __name__ == '__main__':
    config = PredictionConfig()
    main(config)