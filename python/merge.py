import numpy as np
import torch as th
from tqdm import tqdm
import concurrent.futures
from numba import njit, prange

# Reference: From IPOL article: https://www.ipol.im/pub/art/2021/336/article_lr.pdf
def patch_shift(patchSize, patchSize_y, noiseVariance, spatialFactor):
	'''Spatially denoise a set of 2D Frequency-domain patches'''
	rowDistances = (np.arange(patchSize) - patchSize / 2).reshape(-1, 1).repeat(patchSize_y, 1)
	columnDistances = rowDistances.T
	distancePatch = np.sqrt(rowDistances**2 + columnDistances**2)
	distPatchShift = np.fft.ifftshift(distancePatch, axes=(-2, -1))
	noiseScaling = patchSize**2 * 1 / 4**2
	noiseScaling *= spatialFactor
	return distPatchShift * noiseScaling * noiseVariance

def merge_images(aligned_images, reference_image_index, temporal_denoise=True, spatial_denoise=True):
    """Merge the given raw images using the given motion matrix"""
    # [TODO]: What is the point of DFT here? Why can't we just use the motion matrix? 
    # [TODO]: Also this is prohibitively slow

    # TODO: Extract noise model from the dataset. A single value is enough because the assumption 
    # is that the noise is the roughly the same across all images shot from the same device
    
    # TODO: Consider per-patch noise model in place of global noise model
    shot_noise = 30.0
    read_noise = 5.0

    num_rows = len(aligned_images[0])
    num_cols = len(aligned_images[0][0])
    num_aligned_images = len(aligned_images)

    aligned_image = []
    for row in tqdm(range(num_rows)):
        aligned_row = []
        for col in range(num_cols):
            reference_image_patch = aligned_images[reference_image_index][row][col]
            path_rms = np.sqrt(np.mean(np.square(reference_image_patch)))
            dft_ref = th.fft.fftn(th.from_numpy(reference_image_patch), dim=(0, 1)).numpy()
            noise_variance = shot_noise * path_rms + read_noise
            temporal_factor = 8
            k = np.prod(reference_image_patch.shape) / 8
            noise = k * temporal_factor * noise_variance

            if temporal_denoise:
                # Temporal Denoising
                final_patch = dft_ref.copy()
                for i in range(num_aligned_images):
                    if i == reference_image_index:
                        continue
                    alt_dft = np.array(aligned_images[i][row][col])
                    # if dft_ref.shape != alt_dft.shape:
                    #     continue
                    alt_dft = th.fft.fftn(th.from_numpy(alt_dft), dim=(0, 1)).numpy()

                    difference = dft_ref - alt_dft
                    difference_squared = np.square(difference.real) + np.square(difference.imag)
                    shrinkage_operator = difference_squared / (difference_squared + noise)
                    final_patch += (1 - shrinkage_operator) * alt_dft + shrinkage_operator * dft_ref
            else:
                final_patch = dft_ref.copy()
                for i in range(num_aligned_images):
                    if i == reference_image_index:
                        continue
                    alt_dft = np.array(aligned_images[i][row][col])
                    alt_dft = th.fft.fftn(th.from_numpy(alt_dft), dim=(0, 1)).numpy()
                    final_patch += alt_dft
            final_patch /= num_aligned_images
    
            if spatial_denoise:
                # Spatial Denoising
                spatial_variance = patch_shift(reference_image_patch.shape[-1], reference_image_patch.shape[-1], noise_variance, 0.05)
                final_patch_squared = np.square(final_patch.real) + np.square(final_patch.imag)
                
                if final_patch_squared.shape != spatial_variance.shape:
                    offset = np.array(spatial_variance.shape) // np.array(final_patch_squared.shape)
                    spatial_variance = spatial_variance[::offset[0], ::offset[1]]
                shrinkage_operator = final_patch_squared / (final_patch_squared + spatial_variance)
                final_patch = shrinkage_operator * final_patch
            final_patch = th.fft.ifftn(th.from_numpy(final_patch), dim=(0, 1)).numpy().real
            aligned_row.append(final_patch)
        aligned_image.append(aligned_row)
    return aligned_image


def merge_patches(final_patches):
    stacked_patches = []
    for row in final_patches:
        stacked_patches.append(np.hstack(row))
    return np.vstack(stacked_patches)