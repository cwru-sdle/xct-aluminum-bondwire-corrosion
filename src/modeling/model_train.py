# %%
''' import libraries and set parameters '''

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from dataloader import load_data, prepare_datasets

# user defined parameters
IMG_DIR = './data/processed/images'
MASK_DIR = './data/processed/masks'
MODEL_PATH = './models/'

BATCH_SIZE = 1
EPOCHS = 100
IMG_SIZE = [768, 768]

TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

BACKBONE = 'seresnext101'
SAVE_MODEL_NAME = f'{MODEL_PATH}unet_{EPOCH}epochs_{BACKBONE}_aug_p.keras'

sm.set_framework('tf.keras')
preprocess_input = sm.get_preprocessing(BACKBONE)
gpu = tf.config.list_physical_devices('GPU')[0]
tf.random.set_seed(24)

# %%
''' instantiate datasets '''

img_paths, mask_paths = load_data(IMG_DIR, MASK_DIR)
train_ds, val_ds, test_ds = prepare_datasets(img_paths, mask_paths, BATCH_SIZE, augment_flag=True)

# %%
''' instantiate model '''

unet_model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')

unet_model.compile(
    'Adam',
    loss=sm.losses.binary_focal_jaccard_loss,
    metrics=['accuracy', sm.metrics.precision, sm.metrics.recall, sm.metrics.iou_score],
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVE_MODEL_NAME,
    monitor='val_loss',  
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# %%
''' train model '''

model_history = unet_model.fit(train_ds,
                            epochs=50,
                            validation_data=val_ds,
                            verbose=1,
                            callbacks=[checkpoint_callback])
