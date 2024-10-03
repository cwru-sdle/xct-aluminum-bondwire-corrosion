import os
import numpy as np
from glob import glob
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import load_model

from dataloader import read_image
from config import ModelConfig

def prepare_dataset(image_paths: List[str], 
                    batch_size: int, 
                    preprocess_func: Callable = None
                    ) -> tf.data.Dataset:

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: read_image(x, preprocess_func), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def save_prediction(prediction: tf.Tensor, output_path: str):
    prediction = tf.where(prediction > 0.5, 1.0, 0.0)
    prediction = tf.squeeze(prediction)
    prediction = tf.cast(prediction * 255, tf.uint8)
    encoded_mask = tf.io.encode_png(prediction)
    tf.io.write_file(output_path, encoded_mask)

def main():
    # set up environment
    config = ModelConfig()
    sm.set_framework('tf.keras')

    # load model
    custom_objects = {
        'binary_focal_jaccard_loss': sm.losses.binary_focal_jaccard_loss,
        'precision': sm.metrics.precision,
        'recall': sm.metrics.recall,
        'iou_score': sm.metrics.iou_score
    }
    model = load_model(config.save_model_path, custom_objects=custom_objects)

    # prepare dataset
    sm.get_preprocessing(config.backbone) if config.backbone else None
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
    main()