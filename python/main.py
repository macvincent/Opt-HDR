import os
from utils import *
from align import naive_align
from align_and_merge import *
import matplotlib.pyplot as plt
import time

def end_to_end_pipeline(perform_naive_align=False):
    image_name = "0127_20161107_171749_524"
    folder_path = folder_names(image_name)[0]
    dng_files_path = ['{}/{}'.format(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".dng")]
    raw_images = load_raw_images(dng_files_path)
    
    gt_reference_image_index = load_ground_truth_reference_image_index(image_name)
    print("Ground truth reference image index: ", gt_reference_image_index)

    ref_image_index = select_reference_image(raw_images)
    print("Reference image index: ", ref_image_index)

    merged_bayer = align_and_merge_channel(raw_images, ref_image_index)
    demo_images(merged_bayer, image_name, dng_files_path[ref_image_index])

    if perform_naive_align:
        naive_aligned_image = naive_align(raw_images)
        naive_rgb_image = get_rgb_values(dng_files_path[ref_image_index], bayer_array=naive_aligned_image, no_auto_bright=False)
        plt.imshow(naive_rgb_image)
        plt.show()

def load_raw_images_serially(image_name):
    folder_path = folder_names(image_name)[0]
    dng_files_path = ['{}/{}'.format(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".dng")]
    raw_images = load_raw_images(dng_files_path)
    return raw_images

def load_raw_images_parallel(image_name):
    folder_path = folder_names(image_name)[0]
    dng_files_path = ['{}/{}'.format(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".dng")]
    raw_images = parallel_load_raw_images(dng_files_path)
    return raw_images

def naive_speed_benchmark(image_name):
    print("Running naive speed benchmark")
    print("Loading images serially...")
    start = time.time()
    raw_images = load_raw_images_serially(image_name)
    memory_io_time_serial = time.time() - start
    print("Time taken for memory I/O: ", memory_io_time_serial)
 
    print("Selecting reference image...")
    start = time.time()
    ref_image_index = select_reference_image(raw_images)
    ref_image_selection_time = time.time() - start
    print("Time taken for reference image selection: ", ref_image_selection_time)

    print("Aligning and merging channels...")
    start = time.time()
    align_and_merge_channel(raw_images, ref_image_index)
    align_and_merge_time = time.time() - start
    print("Time taken: ", align_and_merge_time)
    
    total_time = memory_io_time_serial + ref_image_selection_time + align_and_merge_time
    print("Total time taken: ", total_time)
    return total_time

def parallel_speed_benchmark(image_name):
    print("\n\nRunning naive speed benchmark")
    print("Loading images parallely...")
    start = time.time()
    raw_images = load_raw_images_parallel(image_name)
    memory_io_time_parallel = time.time() - start
    print("Time taken for memory I/O: ", memory_io_time_parallel)
    
    print("Selecting reference image...")
    start = time.time()
    ref_image_index = parallel_select_reference_image(raw_images)
    ref_image_selection_time = time.time() - start
    print("Time taken for reference image selection: ", ref_image_selection_time)

    print("Aligning and merging channels...")
    start = time.time()
    parallel_align_and_merge_channel(raw_images, ref_image_index)
    align_and_merge_time = time.time() - start
    print("Time taken: ", align_and_merge_time)

    total_time = memory_io_time_parallel + ref_image_selection_time + align_and_merge_time
    print("Total time taken: ", total_time)
    return total_time
    
def main():
    image_name = "0127_20161107_171749_524"
    # naive_speed = naive_speed_benchmark(image_name)
    parallel_speed = parallel_speed_benchmark(image_name)
    # print("\n\nSpeedup: ", naive_speed/parallel_speed)
    
    # Time taken:  48.35582995414734
    # Total time taken:  50.69356656074524
if __name__ == '__main__':
    main()