import numpy as np
from tqdm import tqdm
from utils import upsample_image
import cv2

def naive_align(raw_images):
    """Find Average across all images"""
    return np.mean(raw_images, axis=0)

def gaussian_downsample(image, downsample_factor):
    """Downsample the given image using Gaussian pyramid"""
    downsampled_image = image
    two_d_kernel = cv2.getGaussianKernel(3, 0.1)

    for _ in range(downsample_factor//2):
        height, width = downsampled_image.shape[:2]
        downsampled_image = cv2.filter2D(downsampled_image, -1, two_d_kernel)
        downsampled_image = cv2.resize(downsampled_image, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
    return downsampled_image

def generate_pyramid(images, pyramid_levels=4, downsample_factor=4):
    """Generate Gaussian pyramid for the given image"""
    pyramid_images = []
    for i in range(0, pyramid_levels):
        pyramid_level = []
        for j in range(images.shape[0]):
            if i == 0:
                pyramid_level.append(images[j])
            else:
                downsampled_image = gaussian_downsample(pyramid_images[-1][j], downsample_factor)
                pyramid_level.append(downsampled_image)
        pyramid_images.append(pyramid_level)
    return pyramid_images

def align_level_images(level, pyramid_level_images, motion_matrix, reference_image_index, block_size=16, search_window=4):
    level_reference_image = pyramid_level_images[reference_image_index]
    x_max, y_max = level_reference_image.shape

    for i in tqdm(range(motion_matrix.shape[0])):
        image_offset = motion_matrix[i]
        
        for x in range(0, x_max, block_size):
            for y in range(0, y_max, block_size):
                # Initialize the best match and the best match error
                best_match = np.zeros(3)
                best_match_error = np.inf

                end_x = min(x + block_size, x_max)
                end_y = min(y + block_size, y_max)

                reference_block = level_reference_image[x:end_x, y:end_y]
                offset = image_offset[x, y]

                for x_offset in range(-search_window + offset[0], search_window  + offset[0]):
                    for y_offset in range(-search_window + offset[1], search_window + offset[1]):

                        x_start = max(0, x + x_offset)
                        x_end = min(x + x_offset + block_size, x_max)

                        y_start = max(0, y + y_offset)
                        y_end = min(y + y_offset + block_size, y_max)

                        target_block = pyramid_level_images[i][x_start:x_end, y_start:y_end]

                        if reference_block.shape != target_block.shape:
                            continue

                        # L2 norm
                        if level < 2:
                            # [TODO]: Implement optimized L2 norm
                            block_error = np.sum((reference_block - target_block)**2)
                        else:
                            block_error = np.sum(np.abs(reference_block - target_block))

                        if block_error < best_match_error:
                            best_match_error = block_error
                            best_match = np.array([x_offset, y_offset, 0])

                motion_matrix[i][x:end_x, y:end_y, :] = best_match
    return motion_matrix

def burst_align(reference_image_index, raw_images):
    """Pyramid block matching algorithm"""
    pyramid_levels = 4
    downsample_factor = 4

    pyramid_images = generate_pyramid(raw_images, pyramid_levels, downsample_factor)
    # Arrange the pyramid images in descending order
    pyramid_images = pyramid_images[::-1]
    pyramid_shapes = [image[0].shape for image in pyramid_images]
    print("pyramid shapes: ", pyramid_shapes)

    # Initialize the motion vectors to zeros initially
    motion_matrix = np.zeros((raw_images.shape[0], pyramid_shapes[0][0], pyramid_shapes[0][1],  3), dtype=np.int32)

    for level in range(pyramid_levels):
        if level != 0:
            motion_matrix = upsample_image(motion_matrix, pyramid_shapes[level][0], pyramid_shapes[level][1]) * downsample_factor

        motion_matrix = align_level_images(level, pyramid_images[level], motion_matrix, reference_image_index, block_size=8 if level < 2 else 16, search_window=4)

        # Update the motion vectors
    return motion_matrix

def align(motion_matrix, raw_images, block_size=8):
    print(motion_matrix.shape, raw_images.shape)
    aligned_burst_patches = []
    for i in tqdm(range(raw_images.shape[0])):
        x_max, y_max = motion_matrix[i].shape[:2]
        aligned_image_patches = []

        for x in range(0, x_max, block_size):
            aligned_row_patches = []
            for y in range(0, y_max, block_size):
                best_match = motion_matrix[i][x, y]
                start_x = min(x + best_match[0], x_max)
                end_x = min(start_x + block_size, x_max)
                start_y = min(y + best_match[1], y_max)
                end_y = min(start_y + block_size, y_max)

                aligned_row_patches.append(raw_images[i][start_x:end_x, start_y:end_y])
            aligned_image_patches.append(aligned_row_patches)
        aligned_burst_patches.append(aligned_image_patches)

    return aligned_burst_patches