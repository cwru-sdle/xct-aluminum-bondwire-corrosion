import os
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataloader import load_data, prepare_datasets

@dataclass
class Config:
    # Data paths
    img_dir: str = '../data/processed/images'
    mask_dir: str = '../data/processed/masks'
    model_path: str = '../models'

    # Training parameters
    batch_size: int = 2
    epochs: int = 100
    img_size: Tuple[int, int] = (768, 768)
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05

    # Model parameters
    backbone: str = 'seresnext101'
    learning_rate: float = 0.001

    # Other settings
    random_seed: int = 24
    use_augmentation: bool = True

    def __post_init__(self):
        self.model_name = f'unet_{self.epochs}epochs_{self.backbone}_aug.keras'
        self.save_model_path = os.path.join(self.model_path, self.model_name)
        
        # Ensure splits sum to 1
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, "Split proportions must sum to 1"

def set_gpu_memory_growth():
    """Configure GPU to grow memory allocation as needed."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")

def main(config: Config):
    # Set up environment
    set_gpu_memory_growth()
    sm.set_framework('tf.keras')
    tf.random.set_seed(config.random_seed)

    # Ensure model directory exists
    os.makedirs(config.model_path, exist_ok=True)

    # Prepare data
    preprocess_input = sm.get_preprocessing(config.backbone)
    img_paths, mask_paths = load_data(config.img_dir, config.mask_dir)
    train_ds, val_ds, test_ds = prepare_datasets(
        img_paths, mask_paths, config.batch_size, preprocess_input, 
        augment_flag=config.use_augmentation, 
        train_split=config.train_split, 
        val_split=config.val_split, 
        test_split=config.test_split
    )

    # Define model
    unet_model = sm.Unet(
        config.backbone, 
        encoder_weights='imagenet', 
        input_shape=config.img_size + (3,), 
        classes=1, 
        activation='sigmoid'
    )

    # Compile model
    unet_model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=sm.losses.binary_focal_jaccard_loss,
        metrics=['accuracy', sm.metrics.precision, sm.metrics.recall, sm.metrics.iou_score],
    )

    # Define callbacks
    callbacks = [
        ModelCheckpoint(filepath=config.save_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1),
    ]

    # Train model
    model_history = unet_model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy, test_precision, test_recall, test_iou = unet_model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test IoU: {test_iou:.4f}")

if __name__ == "__main__":
    config = Config()
    main(config)