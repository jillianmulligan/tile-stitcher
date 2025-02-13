import numpy as np
from multiview_stitcher import msi_utils, weights
from multiview_stitcher import spatial_image_utils as si_utils
from dask.diagnostics import ProgressBar
from multiview_stitcher import registration
import os
import cv2
from multiprocessing import freeze_support
from multiview_stitcher import fusion
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import argparse
import csv
import fcntl
import time

filepath = r"Z:\Weekly\Jillian Mulligan\mosaic_test\\"
savepath = filepath
if not os.path.exists(savepath):
    os.makedirs(savepath)
# Get list of image files in directory
image_files = [f for f in os.listdir(filepath) if f.endswith(('.tif', '.tiff'))]

x_trim = 60
x_overlap = 213
y_overlap = 252

# Read first image to get dimensions
first_image = cv2.imread(os.path.join(filepath, image_files[0]), cv2.IMREAD_UNCHANGED)
y_size, x_size = first_image.shape[:2]

csv_root = 'tile_shifts'

fuse_method = 'feather'

def parse_files(image_folder):
    """
    Parse and organize image files into a 2D array where:
    - Each row contains images with the same tile ID
    - Each column represents a specific tile placement
    
    Example filename: XB550-0411_25-02-03_164821_0-1-1_InLens.tif
    """
    # Get all tif files
    image_names = [f for f in os.listdir(image_folder) if f.endswith(('.tif', '.tiff'))]
    
    # Create dictionary to group files by their tile ID
    tile_groups = {}
    for img in image_names:
        # Split the filename to extract tile ID and placement
        parts = img.split('_')
        tile_id = parts[2]  # '164821'
        tile_placement = parts[3]  # '0-1-1'
        
        if tile_id not in tile_groups:
            tile_groups[tile_id] = []
        tile_groups[tile_id].append((tile_placement, img))
    
    # Create the result array
    result = []
    for tile_id in sorted(tile_groups.keys()):
        # Sort images in this group by tile placement
        files_in_group = sorted(tile_groups[tile_id], key=lambda x: x[0])
        # Extract just the filenames in the sorted order
        row = [f[1] for f in files_in_group]
        result.append(row)
    
    return result

def read_images(image_files):
    """Read the provided list of image files."""
    # Read as 16-bit using -1 flag to maintain original bit depth
    images = [cv2.imread(os.path.join(filepath, file), cv2.IMREAD_UNCHANGED)[:,x_trim:] for file in image_files]
    return images

tile_translations = [
    {'x': 0, 'y': -1*(y_size-y_overlap)},
    {'x': x_size-x_overlap-x_trim, 'y': -1*(y_size-y_overlap)},
    {'x': 0, 'y': 0},
    {'x': x_size-x_overlap-x_trim, 'y': 0},
]

spacing = {'x': 1, 'y': 1}

channels = ["InLens"]

def custom_min_fusion(transformed_views, weights=None, **kwargs):
    """
    Custom fusion function that implements minimum projection with handling for partial overlaps.
    
    Args:
        transformed_views: List of pre-transformed view chunks
        weights: Optional weights for blending (not used in min projection)
        kwargs: Additional arguments passed from fusion.fuse
    
    Returns:
        Array containing the minimum projection
    """
    # Stack all views along a new axis
    stacked_views = np.stack(transformed_views, axis=0)
    
    # Create a mask for non-zero values
    valid_mask = (stacked_views > 0)
    
    # Replace invalid values with maximum possible value for comparison
    max_val = np.finfo(stacked_views.dtype).max
    masked_views = np.where(valid_mask, stacked_views, max_val)
    
    # Take minimum along the views axis
    min_projection = np.min(masked_views, axis=0)
    
    # Create final mask where any valid value existed
    final_mask = np.any(valid_mask, axis=0)
    
    # Zero out areas where no valid data existed
    result = np.where(final_mask, min_projection, 0)
    
    return result

def blend_images(tile_arrays, transform_csv, x_trim):
    """
    Blend multiple image tiles using weighted feathering.
    
    Args:
        tile_arrays: List of input image arrays
        transform_csv: List of transformation values from registration
        tile_translations: List of initial tile positions
        x_trim: Amount to trim from x dimension
        
    Returns:
        merged_image: Final blended image as numpy array
    """
    # Calculate shifted tile positions
    tile_translations_shifted = [
        {'x': 0 + transform_csv[1], 'y': -1*(y_size-y_overlap) + transform_csv[0]},
        {'x': x_size-x_overlap-x_trim + transform_csv[3], 'y': -1*(y_size-y_overlap) + transform_csv[2]},
        {'x': 0 + transform_csv[5], 'y': 0 + transform_csv[4]},
        {'x': x_size-x_overlap-x_trim + transform_csv[7], 'y': 0 + transform_csv[6]},
    ]

    # Calculate canvas dimensions
    min_x = min(t['x'] for t in tile_translations_shifted)
    max_x = max(t['x'] + tile_arrays[0].shape[1] for t in tile_translations_shifted)
    min_y = min(t['y'] for t in tile_translations_shifted)
    max_y = max(t['y'] + tile_arrays[0].shape[0] for t in tile_translations_shifted)
    
    total_width = int(max_x - min_x)
    total_height = int(max_y - min_y)

    # Create canvas and place images
    canvas = np.zeros((total_height, total_width, 4), dtype=np.uint16)
    for i in range(4):
        y_start = int(tile_translations_shifted[i]['y'] - min_y)
        y_end = y_start + tile_arrays[i].shape[0]
        x_start = int(tile_translations_shifted[i]['x'] - min_x)
        x_end = x_start + tile_arrays[i].shape[1]
        canvas[y_start:y_end, x_start:x_end, i] = tile_arrays[i]

    # Create and normalize weight maps
    weights = np.zeros_like(canvas, dtype=np.float32)
    for i in range(4):
        mask = (canvas[:,:,i] > 0).astype(np.float32)
        dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
        weights[:,:,i] = dist

    weight_sum = np.sum(weights, axis=2, keepdims=True)
    weight_sum[weight_sum == 0] = 1
    weights = weights / weight_sum

    # Blend images
    merged_image = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint16)
    for i in range(4):
        merged_image += (canvas[:,:,i] * weights[:,:,i]).astype(np.uint16)
    
    return merged_image

def create_msim(tile_array, tile_translation, spacing, channels):
    """
    Create a single msim from a tile array
    """
    if len(tile_array.shape) == 3:
        tile_array = cv2.cvtColor(tile_array, cv2.COLOR_BGR2GRAY)
    
    sim = si_utils.get_sim_from_array(
        tile_array,
        dims=["y", "x"],
        scale=spacing,
        translation=tile_translation,
        transform_key="stage_metadata",
        c_coords=channels,
    )
    return msi_utils.get_msim_from_sim(sim, scale_factors=[])

def stitch_tile_set(tile_arrays, tile_translations, spacing, channels, method = 'feather'):
    """
    Stitch a single set of tiles
    """
    msims = [
        create_msim(tile_array, tile_translation, spacing, channels)
        for tile_array, tile_translation in zip(tile_arrays, tile_translations)
    ]

    params = registration.register(
        msims,
        reg_channel="InLens",
        transform_key="stage_metadata",
        new_transform_key="translation_registered",
    )

    if method == 'min':
        fused_sim = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim) for msim in msims],
            transform_key="translation_registered",
            fusion_func=custom_min_fusion
        )
        merged = fused_sim.data[0,0,:,:].compute()
    elif method == 'feather':
        transform_csv, _ = read_transforms(msims)
        merged = blend_images(tile_arrays, transform_csv, tile_translations, x_trim)

    return merged


def process_single_tileset(tile_files, filepath, savepath, tile_translations, spacing, channels, x_trim):
    """Process a single tile set"""
    # Read images
    tile_arrays = [cv2.imread(os.path.join(filepath, file), cv2.IMREAD_UNCHANGED)[:,x_trim:] for file in tile_files]
    
    # Stitch tiles
    fused_array = stitch_tile_set(tile_arrays, tile_translations, spacing, channels)
    
    # Save result
    base_name = '_'.join(tile_files[0].split('_')[:3])
    save_filename = os.path.join(savepath, f"{base_name}_stitched.tif")
    cv2.imwrite(save_filename, fused_array)
    
    return base_name

def read_transforms(translation_data_msims):
    transform_list = []
    for msim in translation_data_msims:
        transforms = np.array(msi_utils.get_transform_from_msim(msim, transform_key='translation_registered')[0])
        transform_list.append(transforms)

    transform_array = np.stack(transform_list, axis=0)
    transform_csv = [transform_array[0,0,2], transform_array[0,1,2], transform_array[1,0,2], transform_array[1,1,2], transform_array[2,0,2], transform_array[2,1,2], transform_array[3,0,2], transform_array[3,1,2]]
    
    return transform_csv,transform_array

def write_csv_with_lock(transform_csv, tileset, csv_path=f'{csv_root}.csv'):
    """Write to CSV with file locking to handle parallel processes."""
    transform_csv = [float(x) for x in transform_csv]
    while True:
        try:
            with open(csv_path, 'a', newline='') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                # Check if file is empty to write headers
                f.seek(0, 2)  # Seek to end
                if f.tell() == 0:
                    writer = csv.writer(f)
                    writer.writerow(['tileset','x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
                
                # Write the data
                writer = csv.writer(f)
                writer.writerow([tileset,transform_csv])
                
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                break
        except IOError:
            # If lock acquisition fails, wait and retry
            time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description='Stitch a single tile set')
    parser.add_argument('--filepath', type=str, required=True, help='Input directory containing tiles')
    parser.add_argument('--savepath', type=str, required=True, help='Output directory for stitched images')
    parser.add_argument('--tile-index', type=int, required=True, help='Index of tile set to process')
    parser.add_argument('--x-trim', type=int, default=60, help='Number of pixels to trim from x axis')
    parser.add_argument('--count', action='store_true', help='Count number of tile sets and exit')
    args = parser.parse_args()

    # If count flag is set, just print the number of tile sets and exit
    if args.count:
        fileNames = parse_files(args.filepath)
        print(len(fileNames))  # Just print the number, no extra text
        return

    # Define constants
    tile_translations = [
        {'x': 0, 'y': -1*(2625-252)},
        {'x': 2625-213-args.x_trim, 'y': -1*(2625-252)},
        {'x': 0, 'y': 0},
        {'x': 2625-213-args.x_trim, 'y': 0},
    ]
    spacing = {'x': 1, 'y': 1}
    channels = ["InLens"]

    # Ensure save directory exists
    os.makedirs(args.savepath, exist_ok=True)

    # Get tile sets
    fileNames = parse_files(args.filepath)
    if args.tile_index >= len(fileNames):
        print(f"Error: Tile index {args.tile_index} is out of range. Only {len(fileNames)} tile sets found.")
        return

    # Process the specified tile set
    tile_files = fileNames[args.tile_index]
    try:
        base_name = process_single_tileset(
            tile_files=tile_files,
            filepath=args.filepath,
            savepath=args.savepath,
            tile_translations=tile_translations,
            spacing=spacing,
            channels=channels,
            x_trim=args.x_trim
        )
        print(f"Successfully processed tile set {args.tile_index}: {base_name}")
    except Exception as e:
        print(f"Error processing tile set {args.tile_index}: {str(e)}")
        raise  # Re-raise the exception to ensure the job fails properly

if __name__ == '__main__':
    main()