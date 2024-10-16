import os
from pathlib import Path
from typing import List, Dict, Any

import sparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.measure import label, regionprops_table

from config import ModelConfig

def construct_volume(img_paths: List[Path]) -> np.ndarray:
    """
    Construct a 3D volume from a list of image paths.

    Args:
        img_paths (List[Path]): List of paths to image files.

    Returns:
        np.ndarray: 3D numpy array representing the constructed volume.
    """
    volume = []
    for img_path in img_paths:
        img_arr = imread(img_path)
        volume.append(img_arr)

    return np.stack(volume, axis=0)

def analyze_volume(volume: np.ndarray) -> pd.DataFrame:
    """
    Analyze a 3D volume and compute region properties.

    Args:
        volume (np.ndarray): 3D numpy array representing the volume.

    Returns:
        pd.DataFrame: DataFrame containing region properties.
    """
    vol_label = label(np.where(volume != 0, True, False))
    props = regionprops_table(vol_label, 
                              properties=['label', 
                                          'area', 
                                          'bbox', 
                                          'centroid'])
    return pd.DataFrame(props)

def process_timestep(timestep: str, img_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
    """
    Process a single timestep: construct volume, analyze it, and save results.

    Args:
        timestep (str): The timestep being processed.
        img_paths (List[Path]): List of image paths for this timestep.
        output_dir (Path): Directory to save output files.

    Returns:
        Dict[str, Any]: Dictionary containing processing results.
    """
    volume = construct_volume(img_paths)
    df_stats = analyze_volume(volume)
    
    # save volume as sparse matrix
    sparse_matrix = sparse.COO(volume)
    vol_output_path = output_dir / f'timestep_{timestep}_sparse_volume.npz'
    sparse.save_npz(vol_output_path, sparse_matrix)
    
    # save statistics
    stats_output_path = output_dir / f'timestep_{timestep}_stats.csv'
    df_stats.to_csv(stats_output_path, index=False)

def main():
    config = ModelConfig()
    
    # get all image paths and create DataFrame
    img_paths = list(config.prediction_dir.glob('*'))
    df = pd.DataFrame({'img_path': img_paths})
    df['basename'] = df['img_path'].apply(lambda x: x.stem)
    df['timestep'] = df['basename'].str.split('_').str[1]
    df['slice'] = df['basename'].str.split('_').str[3]
    df = df.sort_values(['timestep', 'slice'])

    # process each timestep
    results = []
    timesteps = df['timestep'].unique()
    for timestep in tqdm(timesteps, desc="Processing timesteps"):
        img_paths = df[df['timestep'] == timestep]['img_path'].tolist()
        process_timestep(timestep, img_paths, config.volume_dir)

if __name__ == "__main__":
    main()