import os
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from skimage.io import imread, imsave
from skimage.filters import threshold_multiotsu
from concurrent.futures import ThreadPoolExecutor

def read_img(img_path: str) -> np.ndarray:
    """
    Read an image file and perform basic quality checks.

    Args:
    img_path (str): Path to the image file.

    Returns:
    np.ndarray: The image array.

    Raises:
    ValueError: If the image file cannot be read, has no dimensions, or if pixel values are out of range.
    """
    img_arr = imread(img_path, as_gray=True)
    
    if img_arr is None or img_arr.size == 0:
        raise ValueError(f'Failed to read image or has no dimensions: {os.path.basename(img_path)}')
    
    actual_min, actual_max = np.min(img_arr), np.max(img_arr)
    if actual_min < 0 or actual_max > 255:
        raise ValueError(f'Image values out of expected range [0, 255]. '
                         f'Actual range: [{actual_min}, {actual_max}]. '
                         f'Image: {os.path.basename(img_path)}')
 
    return img_arr

def read_mask(mask_path: str) -> np.ndarray:
    """
    Read a mask file and perform basic quality checks.

    Args:
    mask_path (str): Path to the mask file.

    Returns:
    np.ndarray: The mask array.

    Raises:
    ValueError: If the mask file cannot be read, has no dimensions, or if pixel values are out of range.
    """
    mask_arr = imread(mask_path, as_gray=True)
    
    if mask_arr is None or mask_arr.size == 0:
        raise ValueError(f'Failed to read mask: {os.path.basename(mask_path)}')
    
    actual_min, actual_max = np.min(mask_arr), np.max(mask_arr)
    if actual_min < 0 or actual_max > 1:
        raise ValueError(f'Mask values out of expected range [0, 1]. '
                         f'Actual range: [{actual_min}, {actual_max}]. '
                         f'Mask: {os.path.basename(mask_path)}')
    
    return mask_arr

def threshold(img_arr: np.ndarray) -> np.ndarray:
    """
    Apply multi-Otsu thresholding to the image.

    Otsu's method to compute thresholds and apply the higher threshold 
    to separate the foreground (material) from the background.

    Args:
    img_arr (np.ndarray): Input image array.

    Returns:
    np.ndarray: Thresholded image array.
    """
    thresholds = threshold_multiotsu(img_arr)
    return np.where(img_arr > thresholds[1], img_arr, 0)

def center_crop(arr: np.ndarray, crop_dim: Tuple[int, int]) -> np.ndarray:
    """
    Center crop the input array to the specified dimensions.

    If the input array is smaller than the crop dimensions, 
    it will be returned without modification.

    Args:
    arr (np.ndarray): Input array to be cropped.
    crop_dim (Tuple[int, int]): Desired dimensions (width, height) after cropping.

    Returns:
    np.ndarray: Cropped array.
    """
    height, width = arr.shape[:2]
    post_crop_width, post_crop_height = crop_dim

    # calculate crop coordinates
    crop_x_start = max(0, (width - post_crop_width) // 2)
    crop_y_start = max(0, (height - post_crop_height) // 2)
    crop_x_end = min(width, crop_x_start + post_crop_width)
    crop_y_end = min(height, crop_y_start + post_crop_height)

    return arr[crop_y_start:crop_y_end, crop_x_start:crop_x_end]    

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
        img_thresholded = threshold(img)
        img_cropped = center_crop(img_thresholded, crop_dim)
        img_output_path = os.path.join(output_dir, 'images', os.path.basename(img_path))
        imsave(img_output_path, img_processed.astype(np.uint8))
        
        # process and save mask
        mask = read_mask(mask_path)
        mask_cropped = center_crop(mask, crop_dim)
        mask_output_path = os.path.join(output_dir, 'masks', os.path.basename(mask_path))
        imsave(mask_output_path, mask_processed.astype(np.uint8))
    
    except Exception as e:
        print(f'Error processing {os.path.basename(img_path)}: {str(e)}')

def main():
    parser = argparse.ArgumentParser(description='Process image and mask pairs.')
    parser.add_argument('--img_dir', type=str, default='./data/raw/images', help='Directory containing input images')
    parser.add_argument('--mask_dir', type=str, default='./data/raw/masks', help='Directory containing input masks')
    parser.add_argument('--output_dir', type=str, default='./data/processed', help='Directory to save processed images and masks')
    parser.add_argument('--crop_width', type=int, default=768, help='Width of cropped image')
    parser.add_argument('--crop_height', type=int, default=768, help='Height of cropped image')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes for parallel processing')
    
    args = parser.parse_args()
    
    # ensure output directories exist
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)

    # get all image and mask paths
    img_paths = sorted([os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.endswith('.jpg')])
    mask_paths = sorted([os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir) if f.endswith('.jpg')])
    assert len(img_paths) == len(mask_paths), 'Mismatch in number of images and masks'

    # process using thread pool (since IO bound task)
    crop_dim = (args.crop_width, args.crop_height)
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(process_pair, img_paths, mask_paths, 
                         [crop_dim]*len(img_paths), [args.output_dir]*len(img_paths)),
            total=len(img_paths),
            desc='Processing images and masks'
        ))

    print('Processing complete!')

if __name__ == '__main__':
    main()