from utils import upsample_image
from align import *
from merge import *
import numpy as np
import time

def align_and_merge_channel(raw_images, ref_image_index):
    # A coarse-to-fine, pyramid-based block matching that creates a pyramid representation of every input frame and performs a limited window search to find the most similar tile

    # Downsample the raw images to speed up alignment
    downsampled_raw_images = np.array([gaussian_downsample(raw_image, 2) for raw_image in raw_images])
    print("downsampled raw images with shape: ", downsampled_raw_images[0].shape, "previous shape: ", raw_images[0].shape)

    # Generate alignment matrix using pyramid block matching
    start = time.time()
    motion_matrix = burst_align(ref_image_index, downsampled_raw_images)
    print("Time taken for burst alignment: ", time.time() - start)

    # Upsample the motion matrix to the original image size
    motion_matrix = upsample_image(motion_matrix, raw_images.shape[1], raw_images.shape[2]) * 2

    start = time.time()
    aligned_burst_patches = align(motion_matrix, raw_images)
    print("Time taken for alignment: ", time.time() - start)
    
    # temporal and spatial denoising
    start = time.time()
    final_merged_frame = merge_images(aligned_burst_patches, ref_image_index)
    print("Time taken for merge: ", time.time() - start)

    final_merged_bayer = merge_patches(final_merged_frame)

    return final_merged_bayer

def parallel_align_and_merge_channel(raw_images, ref_image_index):
    # A coarse-to-fine, pyramid-based block matching that creates a pyramid representation of every input frame and performs a limited window search to find the most similar tile

    # Downsample the raw images to speed up alignment
    downsampled_raw_images = np.array([gaussian_downsample(raw_image, 2) for raw_image in raw_images])
    print("downsampled raw images with shape: ", downsampled_raw_images[0].shape, "previous shape: ", raw_images[0].shape)

    # Generate alignment matrix using pyramid block matching
    start = time.time()
    motion_matrix = burst_align(ref_image_index, downsampled_raw_images, parallel=True)
    print("Time taken for parallel alignment: ", time.time() - start)

    # Upsample the motion matrix to the original image size
    motion_matrix = upsample_image(motion_matrix, raw_images.shape[1], raw_images.shape[2]) * 2

    start = time.time()
    aligned_burst_patches = parallel_align(motion_matrix, raw_images)
    print("Time taken for parallel alignment: ", time.time() - start)

    # temporal and spatial denoising
    start = time.time()
    final_merged_frame = merge_images(aligned_burst_patches, ref_image_index)
    print("Time taken for parallel merge: ", time.time() - start)

    final_merged_bayer = merge_patches(final_merged_frame)

    return final_merged_bayer