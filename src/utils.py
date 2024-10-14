import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

def print_metrics(results: List[float], metric_names: List[str]) -> None:
    """
    Print metrics with their corresponding names.

    Args:
        results (List[float]): A list of metric values.
        metric_names (List[str]): A list of metric names corresponding to the values.

    Returns:
        None
    """
    for name, value in zip(metric_names, results):
        print(f'{name}: {value:.4f}')

def save_metrics(
    metrics_path: Path,
    backbone: str,
    encoder_weights: str,
    val_results: List[float],
    test_results: List[float],
    metric_names: List[str]
) -> None:
    """
    Save model metrics to a CSV file.

    This function saves the validation and test metrics along with model configuration
    details to a CSV file. If the file already exists, it appends the new results.

    Args:
        metrics_path (Path): Path to the CSV file where metrics will be saved.
        backbone (str): Name of the backbone/encoder used in the model.
        encoder_weights (str): Type of encoder weights used (e.g., 'imagenet', 'none').
        val_results (List[float]): List of validation metric values.
        test_results (List[float]): List of test metric values.
        metric_names (List[str]): List of metric names corresponding to the values.

    Returns:
        None

    Raises:
        ValueError: If the lengths of val_results, test_results, and metric_names are not equal.
        IOError: If there's an error writing to the CSV file.
    """

    if not (len(val_results) == len(test_results) == len(metric_names)):
        raise ValueError("The lengths of val_results, test_results, and metric_names must be equal.")

    results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backbone': backbone if backbone else 'none',
        'encoder_weights': encoder_weights if encoder_weights and backbone else 'none',
    }
    
    for name, val, test in zip(metric_names, val_results, test_results):
        results[f'val_{name}'] = val
        results[f'test_{name}'] = test
    
    df = pd.DataFrame([results])
    
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        df_existing = pd.read_csv(metrics_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    
    try:
        df.to_csv(metrics_path, index=False)
        print(f'Metrics saved to {metrics_path}')
    except IOError as e:
        print(f'Error writing to CSV file: {e}')
        raise

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

def _preprocess_wrapper(img: tf.Tensor, basename: str, preprocess_func: Callable, input_shape: Tuple[int, int, int]) -> Tuple[tf.Tensor, str]:
    """ Modified wrapper for segmentation_models preprocessing functions. """
    img = tf.image.resize(img, input_shape[:2])
    img = tf.py_function(func=lambda x: preprocess_func(x.numpy()), inp=[img], Tout=tf.float32)
    img.set_shape(input_shape)
    return img, basename