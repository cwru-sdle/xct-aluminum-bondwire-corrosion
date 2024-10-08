import os
import pandas as pd
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from models import Unet
from config import ModelConfig
from dataloader import prepare_datasets

def main():
    # set up environment
    config = ModelConfig()
    sm.set_framework('tf.keras')
    tf.random.set_seed(config.random_seed)

    # prepare data
    preprocess_input = sm.get_preprocessing(config.backbone) if config.backbone and config.encoder_weights else None
    df = pd.read_csv(config.split_path)
    train_ds, val_ds, test_ds = prepare_datasets(
        df, 
        config.batch_size,
        config.img_shape,
        preprocess_input, 
        augment_flag=config.use_augmentation, 
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
        optimizer=Adam(learning_rate=config.learning_rate, clipnorm=1),
        loss=sm.losses.binary_focal_dice_loss,
        metrics=['accuracy', 
                 sm.metrics.Precision(per_image=True), 
                 sm.metrics.Recall(per_image=True), 
                 sm.metrics.FScore(beta=1, per_image=True), 
                 sm.metrics.IOUScore(per_image=True)])

    callbacks = [
        CSVLogger(
            filename=config.save_log_path
        ),
        ModelCheckpoint(
            filepath=config.save_model_path, 
            monitor='val_loss', 
            save_weights_only=True, 
            save_best_only=True, 
            mode='min', 
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.1, 
            patience=15, 
            min_lr=1e-6, 
            verbose=1
        ),
    ]

    # train model
    model_history = unet_model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

if __name__ == '__main__':
    main()
