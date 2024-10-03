import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from skimage.draw import ellipse
from skimage.io import imread, imsave
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import find_contours, EllipseModel
from skimage.filters import threshold_multiotsu, threshold_otsu

from config import DataConfig

def match_image_mask_paths(img_dir: str, mask_dir: str, file_ext: str='.png') -> Tuple[List[str], List[str]]:
    """
    Match image and mask paths based on their basenames.

    Args:
    img_dir (str): Directory containing the images.
    mask_dir (str): Directory containing the masks.
    file_ext (str): File extension to search for.

    Returns:
    Tuple[List[str], List[str]]: Two lists containing the matched image and mask paths.

    """
    img_files = [f for f in img_dir.iterdir() if f.name.endswith(file_ext)]
    mask_files = [f for f in mask_dir.iterdir() if f.name.endswith(file_ext)]

    # create dictionaries with basenames as keys
    img_dict = {f.name: f for f in img_files}
    mask_dict = {f.name: f for f in mask_files}

    # find common basenames
    common_basenames = set(img_dict.keys()) & set(mask_dict.keys())

    matched_img_paths = [img_dict[basename] for basename in common_basenames]
    matched_mask_paths = [mask_dict[basename] for basename in common_basenames]

    print(f'Total images: {len(img_files)}')
    print(f'Total masks: {len(mask_files)}')
    print(f'Matched pairs: {len(matched_img_paths)}')

    return matched_img_paths, matched_mask_paths

def read_img(img_path: str) -> np.ndarray:
    """
    Read an image file and perform basic assertions.

    Args:
    img_path (str): Path to the image file.

    Returns:
    np.ndarray: The image array.

    Raises:
    ValueError: If the image file cannot be read, has no dimensions, or if pixel values are out of range.
    """
    img_arr = imread(img_path, as_gray=True)
    
    if img_arr is None or img_arr.size == 0:
        raise ValueError(f'Failed to read image or has no dimensions: {img_path.name}')
    
    actual_min, actual_max = np.min(img_arr), np.max(img_arr)
    if actual_min < 0 or actual_max > 255:
        raise ValueError(f'Image values out of expected range [0, 255]. '
                         f'Actual range: [{actual_min}, {actual_max}]. '
                         f'Image: {img_path.name}')
 
    return img_arr

def read_mask(mask_path: str) -> np.ndarray:
    """
    Read a mask file and perform basic assertions.

    Args:
    mask_path (str): Path to the mask file.

    Returns:
    np.ndarray: The mask array.

    Raises:
    ValueError: If the mask file cannot be read, has no dimensions, or if values other than 0 and 255.
    """
    mask_arr = imread(mask_path, as_gray=True)
    
    if mask_arr is None or mask_arr.size == 0:
        raise ValueError(f'Failed to read mask: {mask_path.name}')
    
    unique_values = np.unique(mask_arr)
    if not np.array_equal(unique_values, np.array([0, 255])):
        raise ValueError(f'Mask contains values other than 0 and 255. '
                         f'Unique values found: {unique_values}. '
                         f'Mask: {mask_path.name}')
    
    return mask_arr

def remove_background(img_arr: np.ndarray) -> np.ndarray:
    """
    Remove image background.

    Otsu's method to compute thresholds and apply the higher threshold 
    to separate the foreground (material) from the background.

    Args:
    img_arr (np.ndarray): Input image array.

    Returns:
    np.ndarray: Thresholded image array.
    """
    thresholds = threshold_multiotsu(img_arr)
    return np.where(img_arr > thresholds[1], img_arr, 0)

def ellipse_filter(img: np.ndarray, increase_factor: float = 1.0) -> np.ndarray:
    """
    Fit an ellipse to the material to remove threshold artifacts.

    Args:
    img (np.ndarray): Input image array.
    increase_factor (float): Factor to increase the ellipse size. Default is 1.05.

    Returns:
    np.ndarray: Image with artifacts outside the fitted ellipse removed.
    """
    thresh = threshold_otsu(img)
    binary = img > thresh

    # find contours and select the largest one
    contours = find_contours(binary, 0.8)
    if not contours:
        return img
    contour = max(contours, key=len)

    # fit ellipse to the contour
    ellipse_model = EllipseModel()
    if not ellipse_model.estimate(contour):
        return img
    
    xc, yc, a, b, theta = ellipse_model.params
    
    # create ellipse mask
    rr, cc = ellipse(int(xc), int(yc), 
                     int(b * increase_factor), 
                     int(a * increase_factor), 
                     rotation=-theta)
    
    # apply ellipse mask
    rr = np.clip(rr, 0, img.shape[0] - 1)
    cc = np.clip(cc, 0, img.shape[1] - 1)
    ellipse_mask = np.zeros_like(img, dtype=bool)
    ellipse_mask[rr, cc] = True
    
    return np.where(ellipse_mask, img, 0)

def find_material_center(img_arr: np.ndarray) -> Tuple[int, int]:
    """
    Find the center of the material.

    Args:
    thresholded_img (np.ndarray): Thresholded image array.

    Returns:
    Tuple[int, int]: (y, x) coordinates of the material center.
    """
    threshold = threshold_otsu(img_arr)
    thresholded_img = np.where(img_arr > threshold, 1, 0)
    y_coords, x_coords = np.nonzero(thresholded_img)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return thresholded_img.shape[0] // 2, thresholded_img.shape[1] // 2
    return int(np.mean(y_coords)), int(np.mean(x_coords))

def center_crop(img_arr: np.ndarray, crop_dim: Tuple[int, int], center: Tuple[int, int]) -> np.ndarray:
    """
    Center crop the input array to the specified dimensions based on the given center.

    Args:
    img_arr (np.ndarray): Input array to be cropped.
    crop_dim (Tuple[int, int]): Desired dimensions (width, height) after cropping.
    center (Tuple[int, int]): (y, x) coordinates of the center point for cropping.

    Returns:
    np.ndarray: Cropped array.
    """
    height, width = img_arr.shape[:2]
    post_crop_height, post_crop_width = crop_dim
    center_y, center_x = center

    # calculate crop coordinates
    crop_y_start = max(0, center_y - post_crop_height // 2)
    crop_x_start = max(0, center_x - post_crop_width // 2)
    crop_y_end = min(height, crop_y_start + post_crop_height)
    crop_x_end = min(width, crop_x_start + post_crop_width)

    # adjust if crop goes out of bounds
    if crop_y_end - crop_y_start < post_crop_height:
        diff = post_crop_height - (crop_y_end - crop_y_start)
        crop_y_start = max(0, crop_y_start - diff)
        crop_y_end = min(height, crop_y_end + diff)
    if crop_x_end - crop_x_start < post_crop_width:
        diff = post_crop_width - (crop_x_end - crop_x_start)
        crop_x_start = max(0, crop_x_start - diff)
        crop_x_end = min(width, crop_x_end + diff)

    return img_arr[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

def process_pair(img_path: str, mask_path: str, crop_dim: Tuple[int, int], output_dir: str) -> None:
    """
    Process a single image-mask pair and save the results.

    Args:
    img_path (str): Path to the input image file.
    mask_path (str): Path to the input mask file.
    crop_dim (Tuple[int, int]): Desired dimensions (width, height) for cropping.
    output_dir (str): Directory to save processed images and masks.
    """
    try:
        # process and save image
        img = read_img(img_path)
        img_material = remove_background(img)
        img_filtered = ellipse_filter(img_material)
        material_center = find_material_center(img_filtered)
        img_cropped = center_crop(img_filtered, crop_dim, material_center)
        img_output_path = output_dir / 'images' / img_path.name
        imsave(img_output_path, img_cropped.astype(np.uint8))
        
        # process and save mask
        mask = read_mask(mask_path)
        mask_cropped = center_crop(mask, crop_dim, material_center)
        mask_output_path = output_dir / 'masks' / mask_path.name
        imsave(mask_output_path, mask_cropped.astype(np.uint8), check_contrast=False)
    
    except Exception as e:
        print(f'Error processing {img_path.name}: {str(e)}')

def main():
    config = DataConfig()
    
    # ensure output directories exist
    (config.output_dir / 'images').mkdir(exist_ok=True)
    (config.output_dir / 'masks').mkdir(exist_ok=True)

    # get all matching image and mask paths
    img_dir = config.download_dir / 'images'
    mask_dir = config.download_dir / 'masks'
    img_paths, mask_paths = match_image_mask_paths(img_dir, mask_dir)
    
    # process using thread pool (since IO bound task)
    crop_dim = config.crop_dim
    num_workers = config.num_workers if config.num_workers is not None else os.cpu_count()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(process_pair, img_paths, mask_paths, 
                         [crop_dim]*len(img_paths), [config.output_dir]*len(img_paths)),
            total=len(img_paths),
            desc='Processing images and masks'
        ))

    print('Processing complete!')
    
if __name__ == '__main__':
    main()