import os
import pandas as pd
from glob import glob
import tensorflow as tf
import segmentation_models as sm
from typing import List, Callable, Optional, Tuple

def load_data(img_dir: str, mask_dir: str, file_pattern: Optional[str]='*.png') -> Tuple[List[str], List[str]]:
    """
    Load image and mask file paths from the specified directories.

    Args:
        img_dir (str): Directory containing the image files.
        mask_dir (str): Directory containing the mask files.
        file_pattern (str): Regex file pattern to filter files.

    Returns:
        Tuple[List[str], List[str]]: Lists of image and mask file paths.
    """
    img_paths = glob(os.path.join(img_dir, file_pattern))
    mask_paths = glob(os.path.join(mask_dir, file_pattern))

    df_images = pd.DataFrame({'image_path': img_paths, 'base_name': [os.path.basename(f) for f in img_paths]})
    df_masks = pd.DataFrame({'mask_path': mask_paths, 'base_name': [os.path.basename(f) for f in mask_paths]})

    df = pd.merge(df_images, df_masks, on='base_name', how='inner')

    return df['image_path'].tolist(), df['mask_path'].tolist()

def read_image(path: str, preprocess_func: Callable = None) -> tf.Tensor:
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
    if preprocess_func is not None:
        return preprocess_func(image)
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
    preprocess_func: Callable = None,
    augment_flag: bool = False,
    shuffle: bool = True,
    seed: int = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'img_path', 'mask_path', and 'split' columns.
        batch_size (int): Size of batches to create.
        preprocess_func (Callable, optional): Function to preprocess images.
        augment_flag (bool): Whether to apply data augmentation.
        shuffle (bool): Whether to shuffle the dataset.
        seed (int, optional): Random seed for shuffling.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
            Training, validation, and test datasets.
    """

    def create_dataset(split: str) -> tf.data.Dataset:
        split_df = df[df['split'] == split]
        dataset = tf.data.Dataset.from_tensor_slices((split_df['img_path'], split_df['mask_path']))
        dataset = dataset.map(lambda img, mask: (read_image(img, preprocess_func), read_mask(mask)), 
                              num_parallel_calls=tf.data.AUTOTUNE)
        if split == 'train' and augment_flag:
            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle and split == 'train':
            dataset = dataset.shuffle(buffer_size=len(split_df))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = create_dataset('train')
    val_ds = create_dataset('val')
    test_ds = create_dataset('test')

    print(f'Training set size: {len(df[df["split"] == "train"])}')
    print(f'Validation set size: {len(df[df["split"] == "val"])}')
    print(f'Test set size: {len(df[df["split"] == "test"])}')

    return train_ds, val_ds, test_ds