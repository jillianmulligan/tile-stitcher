import numpy as np
from multiview_stitcher import msi_utils, weights
from multiview_stitcher import spatial_image_utils as si_utils
from dask.diagnostics import ProgressBar
from multiview_stitcher import registration
import os
import cv2
import glob
from multiprocessing import freeze_support
from multiview_stitcher import fusion
import matplotlib.pyplot as plt
import csv


filepath = r"Z:\Weekly\Jillian Mulligan\mosaic_test\\"
# Get list of image files in directory
image_files = [f for f in os.listdir(filepath) if f.endswith(('.tif', '.tiff'))]
csv_root = "test"

x_trim = 60

def read_images(image_folder):
    """Read all images in the folder, assuming they are named sequentially."""
    file_paths = sorted(glob.glob(os.path.join(image_folder, '*.tif')))
    # Read images maintaining original bit depth with IMREAD_UNCHANGED
    images = [cv2.imread(file, cv2.IMREAD_UNCHANGED)[:,x_trim:] for file in file_paths]
    # Apply histogram equalization to each image
    #images = [cv2.equalizeHist(img) for img in images]
    return images

tile_translations = [
    {'x': 0, 'y': -1*(2625-252)},
    {'x': 2625-213-x_trim, 'y': -1*(2625-252)},
    {'x': 0, 'y': 0},
    {'x': 2625-213-x_trim, 'y': 0},
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
    
    # Replace zeros with the maximum possible value for the data type
    max_val = np.iinfo(np.uint8).max if stacked_views.dtype == np.uint8 else np.finfo(stacked_views.dtype).max
    masked_views = np.where(valid_mask, stacked_views, max_val)
    
    # Take minimum along the views axis
    min_projection = np.min(masked_views, axis=0)
    
    # Create final mask where any valid value existed
    final_mask = np.any(valid_mask, axis=0)
    
    # Zero out areas where no valid data existed
    result = np.where(final_mask, min_projection, 0)
    
    # Ensure output maintains the same data type as input
    result = result.astype(stacked_views.dtype)
    
    return result

def read_transforms(translation_data_msims):
    transform_list = []
    for msim in translation_data_msims:
        transforms = np.array(msi_utils.get_transform_from_msim(msim, transform_key='translation_registered')[0])
        transform_list.append(transforms)

    transform_array = np.stack(transform_list, axis=0)
    transform_csv = [transform_array[0,0,2], transform_array[0,1,2], transform_array[1,0,2], transform_array[1,1,2], transform_array[2,0,2], transform_array[2,1,2], transform_array[3,0,2], transform_array[3,1,2]]
    
    return transform_csv,transform_array

def write_csv_with_lock(transform_csv, csv_path=f'{csv_root}.csv'):
    import fcntl
    import time
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
                    writer.writerow(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
                
                # Write the data
                writer = csv.writer(f)
                writer.writerow(transform_csv)
                
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                break
        except IOError:
            # If lock acquisition fails, wait and retry
            time.sleep(0.1)

def blend_images(tile_arrays, transform_csv, tile_translations, x_trim):
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
        {'x': 0 + transform_csv[1], 'y': -1*(2625-252) + transform_csv[0]},
        {'x': 2625-213-x_trim + transform_csv[3], 'y': -1*(2625-252) + transform_csv[2]},
        {'x': 0 + transform_csv[5], 'y': 0 + transform_csv[4]},
        {'x': 2625-213-x_trim + transform_csv[7], 'y': 0 + transform_csv[6]},
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

def main():
    tile_arrays = read_images(filepath)
    
    msims = []
    for tile_array, tile_translation in zip(tile_arrays, tile_translations):
        # Ensure array is 2D (height x width)
        if len(tile_array.shape) == 3:
            # Take first channel to maintain bit depth
            tile_array = tile_array[:, :, 0]
        
        sim = si_utils.get_sim_from_array(
            tile_array,
            dims=["y", "x"],
            scale=spacing,
            translation=tile_translation,
            transform_key="stage_metadata",
            c_coords=channels,
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

    with ProgressBar():
        params = registration.register(
            msims,
            reg_channel="InLens",
            transform_key="stage_metadata",
            new_transform_key="translation_registered",
        )
    
    # from multiview_stitcher import vis_utils
    # # Plot the tile configuration after registration
    # vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False)

    fused_sim = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key="translation_registered",
        fusion_func=custom_min_fusion,
        output_chunksize=512
    )

    #get fused array as a dask array
    fused_sim.data

    # Get the fused array as numpy array
    fused_array = fused_sim.data.compute()[0,0,:,:]

    # Ensure 16-bit depth to match input images
    fused_array = fused_array.astype(np.uint16)
    
    # Save as 16-bit TIFF
    cv2.imwrite('fused_image.tif', fused_array)

    transform_csv, _ = read_transforms(msims)
    merged_image = blend_images(tile_arrays, transform_csv, tile_translations, x_trim)
    cv2.imwrite('merged_image.tif', merged_image)
    #_, transform_array = read_transforms(msims)
    # print(f"Transform array shape: {transforms.shape}")
    # print("\nTransforms for each image:")
    # for i in range(len(msims)):
    #     print(f"Image {i + 1} transform:")
    #     print(transforms[i])
    
    # transform_csv = [float(x) for x in transform_csv]  # Convert np.float64 to Python float

    # Check if file exists
    # file_exists = os.path.isfile(f'{csv_root}.csv')
    
    # # Open in append mode
    # with open(f'{csv_root}.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     # Write headers only if file is new
    #     if not file_exists:
    #         writer.writerow(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    #     writer.writerow(transform_csv)

    #for parallel processing on Unix systems
    #write_csv_with_lock(f'{csv_root}.csv', transform_csv)
    
    # # Display the fused image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(fused_array, cmap='gray')
    # plt.axis('off')
    # plt.title('Fused Image')
    # plt.show()

if __name__ == '__main__':
    freeze_support()  # Required for Windows
    main()
