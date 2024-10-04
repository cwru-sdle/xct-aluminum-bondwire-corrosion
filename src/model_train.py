import os
import pandas as pd
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from models import Unet
from config import ModelConfig
from dataloader import load_data, prepare_datasets

def main():
    # set up environment
    config = ModelConfig()
    sm.set_framework('tf.keras')
    tf.random.set_seed(config.random_seed)

    # prepare data
    preprocess_input = sm.get_preprocessing(config.backbone) if config.backbone else None
    df = pd.read_csv(config.split_path)
    train_ds, val_ds, test_ds = prepare_datasets(
        df, 
        config.batch_size, 
        preprocess_input, 
        augment_flag=config.use_augmentation, 
    )

    # define model
    if config.backbone:
        unet_model = sm.Unet(
            config.backbone, 
            encoder_weights=config.encoder_weights, 
            input_shape=config.img_size + (config.num_channels,), 
            classes=1,
            activation='sigmoid'
        )
    else:
        unet_model = Unet(
            input_shape=config.img_size + (config.num_channels,), 
            classes=1, 
            activation='sigmoid'
        ).build()
    
    # compile model
    unet_model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=sm.losses.binary_focal_jaccard_loss,
        metrics=['accuracy', sm.metrics.precision, sm.metrics.recall, sm.metrics.iou_score],
    )

    callbacks = [
        ModelCheckpoint(filepath=config.save_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=1),
    ]

    # train model
    model_history = unet_model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # evaluate model
    test_loss, test_accuracy, test_precision, test_recall, test_iou = unet_model.evaluate(test_ds)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test IoU: {test_iou:.4f}')

if __name__ == '__main__':
    main()
