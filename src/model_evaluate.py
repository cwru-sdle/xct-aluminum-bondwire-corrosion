import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

from models import Unet
from config import ModelConfig
from dataloader import prepare_datasets
from utils import print_metrics, save_metrics

def main():
    # set up environment
    config = ModelConfig()
    sm.set_framework('tf.keras')
    tf.random.set_seed(config.random_seed)

    # prepare data
    preprocess_input = sm.get_preprocessing(config.backbone) if config.encoder_weights else None
    df = pd.read_csv(config.split_path)
    train_ds, val_ds, test_ds = prepare_datasets(
        df, 
        config.batch_size,
        config.img_shape,
        preprocess_input,
        augment_flag=False, 
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
    unet_model.compile(
        loss=sm.losses.binary_focal_jaccard_loss,
        metrics=['accuracy', 
                 sm.metrics.Precision(), 
                 sm.metrics.Recall(), 
                 sm.metrics.FScore(beta=1), 
                 sm.metrics.IOUScore()])

    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1-score', 'iou']

    # load pretrained weights
    if config.save_model_path.exists():
        print(f'Loading weights from {config.save_model_path}')
        unet_model.load_weights(config.save_model_path)
    else:
        print(f'No weights found at {config.save_model_path}. Using randomly initialized weights.')

    # evaluate on validation set
    print('Evaluating on validation set:')
    val_results = unet_model.evaluate(val_ds, verbose=1)
    print_metrics(val_results, metrics)

    # evaluate on test set
    print('\nEvaluating on test set:')
    test_results = unet_model.evaluate(test_ds, verbose=1)
    print_metrics(test_results, metrics)

    # save metrics to CSV
    save_metrics(config.metrics_path,
                 config.backbone,
                 config.encoder_weights,
                 val_results, 
                 test_results, 
                 metrics)

if __name__ == '__main__':
    main()
