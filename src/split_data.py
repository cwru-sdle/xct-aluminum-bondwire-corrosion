import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from config import ModelConfig

def match_files(img_dir: Path, mask_dir: Path, file_pattern: str = '*.png') -> pd.DataFrame:
    """
    Match image and mask files based on their base names.

    Args:
        img_dir (Path): Directory containing image files.
        mask_dir (Path): Directory containing mask files.
        file_pattern (str): Pattern to match files. Defaults to '*.png'.

    Returns:
        pd.DataFrame: DataFrame containing matched image and mask file paths.
    """
    img_paths = list(img_dir.glob(file_pattern))
    mask_paths = list(mask_dir.glob(file_pattern))

    df_images = pd.DataFrame({'img_path': img_paths, 'base_name': [f.stem for f in img_paths]})
    df_masks = pd.DataFrame({'mask_path': mask_paths, 'base_name': [f.stem for f in mask_paths]})

    df = pd.merge(df_images, df_masks, on='base_name', how='inner')
    return df.drop(columns=['base_name'])

def extract_info(filepath: Path) -> Tuple[int, int]:
    """
    Extract timestep and slice information from a filepath.

    Args:
        filepath (Path): Path to the file.

    Returns:
        Tuple[int, int]: Timestep and slice numbers. Returns (None, None) if not found.
    """
    match = re.search(r'timestep_(\d+)_slice_(\d+)', filepath.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def perform_random_split(df: pd.DataFrame, val_ratio: float, test_ratio: float = 0.0) -> pd.DataFrame:
    """
    Perform random splits on the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        val_ratio (float): Ratio of validation set (0 to 1).
        test_ratio (float): Ratio of test set (0 to 1). Defaults to 0.0.

    Returns:
        pd.DataFrame: DataFrame with an additional 'split' column.

    Raises:
        ValueError: If sum of val_ratio and test_ratio is >= 1.
    """
    if val_ratio + test_ratio >= 1:
        raise ValueError("Sum of val_ratio and test_ratio must be less than 1")
    
    df = df.copy()
    indices = np.arange(len(df))
    train_indices, val_test_indices = train_test_split(indices, test_size=(val_ratio + test_ratio))
    val_indices, test_indices = train_test_split(val_test_indices, test_size=(test_ratio / (val_ratio + test_ratio)))
    
    df['split'] = 'train'
    df.loc[val_indices, 'split'] = 'val'
    df.loc[test_indices, 'split'] = 'test'
    return df

def perform_manual_split(df: pd.DataFrame, val_timesteps: List[int], test_timesteps: List[int] = None) -> pd.DataFrame:
    """
    Perform manual splits on the dataframe based on timesteps.

    Args:
        df (pd.DataFrame): Input dataframe.
        val_timesteps (List[int]): List of timesteps for validation set.
        test_timesteps (List[int], optional): List of timesteps for test set. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with an additional 'split' column.
    """
    df = df.copy()
    df['split'] = 'train'
    df.loc[df['timestep'].isin(val_timesteps), 'split'] = 'val'
    if test_timesteps:
        df.loc[df['timestep'].isin(test_timesteps), 'split'] = 'test'
    return df

def perform_hybrid_split(df: pd.DataFrame, val_ratio: float, test_timesteps: List[int]) -> pd.DataFrame:
    """
    Perform a hybrid split where train/val are random and test is manual based on timesteps.

    Args:
        df (pd.DataFrame): Input dataframe.
        val_ratio (float): Ratio of validation set (0 to 1).
        test_timesteps (List[int]): List of timesteps for test set.

    Returns:
        pd.DataFrame: DataFrame with an additional 'split' column.
    """
    df = df.copy()
    df['split'] = 'train'
    
    test_mask = df['timestep'].isin(test_timesteps)
    df.loc[test_mask, 'split'] = 'test'
    
    train_val_indices = df[df['split'] == 'train'].index
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio,
        random_state=24
    )
    
    df.loc[val_indices, 'split'] = 'val'
    return df

def main():
    config = ModelConfig()

    img_dir = config.img_dir
    mask_dir = config.mask_dir

    # match files and extract info
    df = match_files(img_dir, mask_dir)
    df['timestep'], df['slice'] = zip(*df['img_path'].map(extract_info))

    # perform data split
    if config.split_type == 'random':
        df = perform_random_split(df, config.val_ratio, config.test_ratio)
    elif config.split_type == 'hybrid':
        df = perform_hybrid_split(df, config.val_ratio, config.test_timesteps)
    else:
        df = perform_manual_split(df, config.val_timesteps, config.test_timesteps)

    output_path = config.split_path
    df.to_csv(output_path, index=False)
    print(f'Processed data saved to {output_path}')

if __name__ == '__main__':
    main()