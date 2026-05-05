import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import glob

# Configuration
DATA_ROOT = '/general/omrosa/FFHQ'
MAPPING = {
    'reals_256': 'canny_reals',
    'fakes_256': 'canny_fakes'
}

def auto_canny(image, sigma=0.33):
    """
    Computes optimal Canny thresholds based on the image median.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def process_single_image(paths):
    in_path, out_path = paths
    
    if os.path.exists(out_path):
        return

    # Load as grayscale
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # --- RE-ADDED GAUSSIAN BLUR ---
    # Kernel size (3,3) helps suppress noise before gradient calculation
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply Canny to the blurred image
    canny_map = auto_canny(blurred)
    
    # Ensure the subfolder (like 00000/) exists in the destination
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canny_map)

if __name__ == '__main__':
    for src_folder_name, dst_folder_name in MAPPING.items():
        src_dir = os.path.join(DATA_ROOT, src_folder_name)
        dst_dir = os.path.join(DATA_ROOT, dst_folder_name)
        
        print(f"\n--- Processing {src_folder_name} ---")
        
        # Recursive glob to handle the nested structure (00000/, 00001/, etc.)
        all_images = glob.glob(os.path.join(src_dir, '**/*.png'), recursive=True)
        
        task_list = []
        for img_path in all_images:
            rel_path = os.path.relpath(img_path, src_dir)
            task_list.append((img_path, os.path.join(dst_dir, rel_path)))

        print(f"Found {len(task_list)} images. Generating Edge Maps (with Blur)...")
        
        # Parallel processing using 16 cores
        with Pool(16) as p:
            list(tqdm(p.imap_unordered(process_single_image, task_list), total=len(task_list)))

    print("\nCanny edge map generation with Gaussian Blur complete.")
