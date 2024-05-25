import os
from utils import load_raw_images, get_rgb_values, select_reference_image, load_ground_truth_reference_image, demo_images, folder_names
from align import naive_align
from align_and_merge import align_and_merge_channel
import matplotlib.pyplot as plt


def main(perform_naive_align=False):
    image_name = "0127_20161107_171749_524"
    folder_path = folder_names(image_name)[0]
    dng_files_path = ['{}/{}'.format(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".dng")]
    raw_images = load_raw_images(dng_files_path)
    
    gt_reference_image_index = load_ground_truth_reference_image(image_name)
    ref_image_index = select_reference_image(raw_images)
    print("Reference image index: ", ref_image_index)
    print("Ground truth reference image index: ", gt_reference_image_index)

    merged_bayer = align_and_merge_channel(raw_images, ref_image_index)
    demo_images(merged_bayer, image_name, dng_files_path[ref_image_index])

    if perform_naive_align:
        naive_aligned_image = naive_align(raw_images)
        naive_rgb_image = get_rgb_values(dng_files_path[ref_image_index], bayer_array=naive_aligned_image, no_auto_bright=False)
        plt.imshow(naive_rgb_image)
        plt.show()

if __name__ == '__main__':
    main()