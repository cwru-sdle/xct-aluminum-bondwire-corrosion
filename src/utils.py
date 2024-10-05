import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union

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

def save_metrics_to_csv(config, 
                        val_results: List[float], 
                        test_results: List[float], 
                        metric_names: List[str]) -> None:
    """
    Save model metrics to a CSV file.

    This function saves the validation and test metrics along with model configuration
    details to a CSV file. If the file already exists, it appends the new results.

    Args:
        config (ModelConfig): An object containing model configuration details.
        val_results (List[float]): A list of validation metric values.
        test_results (List[float]): A list of test metric values.
        metric_names (List[str]): A list of metric names corresponding to the values.

    Returns:
        None

    Raises:
        FileNotFoundError: If the CSV file cannot be created or accessed.
    """
    results: Dict[str, Union[str, float]] = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'encoder': config.backbone,
        'encoder_weights': config.encoder_weights,
    }
    
    for name, val, test in zip(metric_names, val_results, test_results):
        results[f'val_{name}'] = val
        results[f'test_{name}'] = test
    
    df = pd.DataFrame([results])
    
    csv_path = Path('model_metrics.csv')
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    print(f'Metrics saved to {csv_path}')