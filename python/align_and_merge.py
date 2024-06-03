from utils import upsample_image
from align import burst_align, align, upsample_image, gaussian_downsample
from merge import merge_images, merge_patches
import numpy as np

def align_and_merge_channel(raw_images, ref_image_index):
    # A coarse-to-fine, pyramid-based block matching that creates a pyramid representation of every input frame and performs a limited window search to find the most similar tile

    # Downsample the raw images to speed up alignment
    downsampled_raw_images = np.array([gaussian_downsample(raw_image, 2) for raw_image in raw_images])
    print("downsampled raw images with shape: ", downsampled_raw_images[0].shape, "previous shape: ", raw_images[0].shape)

    # Generate alignment matrix using pyramid block matching
    motion_matrix = burst_align(ref_image_index, downsampled_raw_images)

    # Upsample the motion matrix to the original image size
    motion_matrix = upsample_image(motion_matrix, raw_images.shape[1], raw_images.shape[2]) * 2

    aligned_burst_patches = align(motion_matrix, raw_images)

    # temporal and spatial denoising
    final_merged_frame = merge_images(aligned_burst_patches, ref_image_index)

    final_merged_bayer = merge_patches(final_merged_frame)

    return final_merged_bayer