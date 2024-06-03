import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_rgb_values(image_path, bayer_array=None, **kwargs):
    with rawpy.imread(image_path) as raw:
        if bayer_array is not None:
            raw.raw_image[:] = bayer_array
        rgb = raw.postprocess(**kwargs)
    return rgb

def load_raw_images(files_path):
    """Load raw images from the given path"""
    raw_images = []
    for path in files_path:
        with rawpy.imread(path) as raw:
            # print noise model of the raw image
            raw_images.append(raw.raw_image_visible.copy())
    raw_images = np.stack(raw_images)
    raw_images = raw_images.astype(np.float32)
    return raw_images

def select_reference_image(raw_images):
    """Select the reference image from the given raw images"""
    gradient_magnitudes = []
    for i in range(raw_images[:3].shape[0]):
        di, dj = 0, 1
        # select the green channel
        image = raw_images[i, di::2, dj::2]
        gy, gx = np.gradient(image.astype(np.float32))
        gradient_magnitude = np.sqrt(gx**2 + gy**2).sum()
        gradient_magnitudes.append(gradient_magnitude)

    return gradient_magnitudes.index(max(gradient_magnitudes))

def upsample_image(image, height, width):
    """Nearest-neighbor upsample the given image to the given shape"""
    if len(image.shape) > 3:
        return np.array([upsample_image(image[i], height, width) for i in range(image.shape[0])])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

def display_crop(reference_rgb, merged_rgb):
    # figure
    font_size = 14
    fig, axs = plt.subplots(1, 4, figsize=[20, 16])

    # crop
    crop_y = [200, 500]
    crop_x = [200, 500]

    # reference image
    axs[0].imshow(reference_rgb)
    axs[0].set_title('Reference image (full)', fontsize=font_size)
    axs[2].imshow(reference_rgb[crop_y[0]:crop_y[1]:, crop_x[0]:crop_x[1], :])
    axs[2].set_title('Reference image (crop)', fontsize=font_size)

    # merged burst
    axs[1].imshow(merged_rgb)
    axs[1].set_title('Merged image (full)', fontsize=font_size)
    axs[3].imshow(merged_rgb[crop_y[0]:crop_y[1]:, crop_x[0]:crop_x[1], :])
    axs[3].set_title('Merged image (crop)', fontsize=font_size)

    for ax in axs:
        ax.set_aspect(1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'before_and_after.jpg', bbox_inches='tight')
    plt.show()

def demo_images(merged_bayer, ref_image_name, ref_image_path):
    brigthness = 2.5

    gt_image_path = folder_names(ref_image_name)[1] + 'merged.dng'
    gt_rgb = load_raw_images([gt_image_path])[0]
    # TODO: Use DCRAW postprocessing parameters
    gt_rgb = get_rgb_values(gt_image_path, bayer_array=gt_rgb, bright=brigthness, no_auto_bright=True)
    plt.imshow(gt_rgb)
    plt.show()

    # print("final merged bayer shape: ", merged_bayer.shape, "previous shape: ", np.array(merged_bayer).shape)
    merged_rgb = get_rgb_values(ref_image_path, merged_bayer, bright=brigthness, no_auto_bright=True)
    # print(merged_rgb.shape, merged_bayer.max(), merged_bayer.min())
    merged_rgb = cv2.resize(merged_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    plt.imsave('merged.jpg', merged_rgb)
    plt.imshow(merged_rgb)
    plt.show()

    display_crop(gt_rgb, merged_rgb)
    
def load_ground_truth_reference_image(ref_image_name):
    ref_image_path = folder_names(ref_image_name)[1] + 'reference_frame.txt'
    file = open(ref_image_path, 'r')
    ref_image_index = int(next(file))
    file.close()
    return ref_image_index

def folder_names(image_name):
    return "../../Dataset/20171106_subset/bursts/{}/".format(image_name), "../../Dataset/20171106_subset/results_20161014/{}/".format(image_name)