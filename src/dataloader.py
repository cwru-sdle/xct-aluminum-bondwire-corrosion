import os
import pandas as pd
from glob import glob
import tensorflow as tf
import segmentation_models as sm
from typing import List, Callable, Tuple

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
    image_paths: List[str], 
    mask_paths: List[str], 
    batch_size: int,
    preprocess_func: Callable = None,
    augment_flag: bool = False,
    train_split: float = 0.7, 
    val_split: float = 0.15, 
    test_split: float = 0.15, 
    shuffle: bool = True,
    seed: int = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create and split a TensorFlow dataset with proper handling of augmentation.

    Args:
        image_paths (List[str]): List of image file paths.
        mask_paths (List[str]): List of mask file paths.
        batch_size (int): Size of batches to create.
        preprocess_func (Callable, optional): Function to preprocess images.
        augment_flag (bool): Whether to apply data augmentation.
        train_split (float): Proportion of data to use for training.
        val_split (float): Proportion of data to use for validation.
        test_split (float): Proportion of data to use for testing.
        shuffle (bool): Whether to shuffle the dataset.
        seed (int, optional): Random seed for shuffling.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
            Training, validation, and test datasets.
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, 'Split proportions must sum to 1'
    
    # create and preprocess intital dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, mask: (read_image(img, preprocess_func), read_mask(mask)), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # conduct train val test splits
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train_ds, temp_ds = tf.keras.utils.split_dataset(
        dataset, 
        left_size=train_size,
        right_size=None,
        shuffle=shuffle,
        seed=seed
    )
    
    val_ds, test_ds = tf.keras.utils.split_dataset(
        temp_ds,
        left_size=val_size,
        right_size=None,
        shuffle=False,  # no second shuffle
        seed=seed
    )
    
    # apply augmentation to train split
    if augment_flag:
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # batch and prefetch
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f'Total dataset size: {dataset_size}')
    print(f'Training set size: {tf.data.experimental.cardinality(train_ds).numpy() * batch_size}')
    print(f'Validation set size: {tf.data.experimental.cardinality(val_ds).numpy() * batch_size}')
    print(f'Test set size: {tf.data.experimental.cardinality(test_ds).numpy() * batch_size}')
    
    return train_ds, val_ds, test_ds