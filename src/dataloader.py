import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import segmentation_models as sm
from typing import List, Callable, Optional, Tuple

def read_image(path: str) -> tf.Tensor:
    """
    Read and preprocess an image.

    Args:
        path (str): Path to the image file.
        preprocess_func (Callable, optional): Function to preprocess the image.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def read_mask(path: str) -> tf.Tensor:
    """
    Read and preprocess a mask.

    Args:
        path (str): Path to the mask file.

    Returns:
        tf.Tensor: Preprocessed mask tensor.
    """
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    return tf.where(mask > 0.5, 1.0, 0.0)

def preprocess_wrapper(img, mask, preprocess_func, input_shape):
    """
    Wrapper for segmentation_models preprocessing functions.

    Args:
        img (tf.Tensor): Image tensor
        mask (tf.Tensor): Mask tensor
        preprocess_func (callable): Preprocessing function to apply
        input_shape (tuple): Expected shape of the input image (height, width, channels)

    Returns:
        tuple: Preprocessed image tensor and original mask tensor
    """
    img = tf.py_function(func=lambda x: preprocess_func(x.numpy()), inp=[img], Tout=tf.float32)
    img.set_shape(input_shape)
    mask.set_shape(input_shape[0:2] + (1,))
    return img, mask

@tf.function
def augment(input_image: tf.Tensor, input_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply data augmentation to image and mask.

    Args:
        input_image (tf.Tensor): Input image tensor.
        input_mask (tf.Tensor): Input mask tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Augmented image and mask tensors.
    """
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)
    input_image = tf.image.random_contrast(input_image, lower=0.8, upper=1.2)
    return input_image, input_mask

def prepare_datasets(
    df: pd.DataFrame,
    batch_size: int,
    input_shape: Tuple[int, int, int],
    preprocess_func: Callable = None,
    augment_flag: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'img_path', 'mask_path', and 'split' columns.
        batch_size (int): Size of batches to create.
        preprocess_func (Callable, optional): Function to preprocess images.
        augment_flag (bool): Whether to apply data augmentation.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
            Training, validation, and test datasets.
    """

    def create_dataset(split):
        split_df = df[df['split'] == split]
        dataset = tf.data.Dataset.from_tensor_slices((split_df['img_path'], split_df['mask_path']))
        dataset = dataset.map(
            lambda img, mask: (read_image(img), read_mask(mask)), 
            num_parallel_calls=tf.data.AUTOTUNE)
        if preprocess_func:
            dataset = dataset.map(
                lambda img, mask: preprocess_wrapper(img, mask, preprocess_func, input_shape),
                num_parallel_calls=tf.data.AUTOTUNE)
        if split == 'train' and augment_flag:
            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    train_ds = create_dataset('train')
    val_ds = create_dataset('val')
    test_ds = create_dataset('test')

    print(f'Training set size: {len(df[df["split"] == "train"])}')
    print(f'Validation set size: {len(df[df["split"] == "val"])}')
    print(f'Test set size: {len(df[df["split"] == "test"])}')

    return train_ds, val_ds, test_ds

def preprocess_wrapper(img: tf.Tensor, basename: str, preprocess_func: Callable, input_shape: Tuple[int, int, int]) -> Tuple[tf.Tensor, str]:
    """
    Wrapper for segmentation_models preprocessing functions.

    Args:
        img (tf.Tensor): Image tensor
        basename (str): Basename of the image file
        preprocess_func (callable): Preprocessing function to apply
        input_shape (tuple): Expected shape of the input image (height, width, channels)

    Returns:
        tuple: Preprocessed image tensor and basename
    """
    img = tf.image.resize(img, input_shape[:2])
    img = tf.py_function(func=lambda x: preprocess_func(x.numpy()), inp=[img], Tout=tf.float32)
    img.set_shape(input_shape)
    return img, basename

def prediction_dataset(
    img_paths: List[Path],
    batch_size: int,
    input_shape: Tuple[int, int, int],
    preprocess_func: Callable = None,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset for prediction, including image basenames.

    Args:
        img_paths (List[Path]): List of paths to image files.
        batch_size (int): Size of batches to create.
        input_shape (Tuple[int, int, int]): Shape of input images (height, width, channels).
        preprocess_func (Callable, optional): Function to preprocess images.

    Returns:
        tf.data.Dataset: Dataset containing tuples of (preprocessed image, image basename).
    """
    basenames = [path.name for path in img_paths]
    img_paths = [path.as_posix() for path in img_paths]
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, basenames))

    dataset = dataset.map(
        lambda img_path, basename: (read_image(img_path), basename), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if preprocess_func:
        dataset = dataset.map(
            lambda img, basename: preprocess_wrapper(img, basename, preprocess_func, input_shape),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset