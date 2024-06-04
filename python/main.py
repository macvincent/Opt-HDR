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

    print("Aligning...")
    start = time.time()
    downsampled_raw_images = np.array([gaussian_downsample(raw_image, 2) for raw_image in raw_images])

    # Generate alignment matrix using pyramid block matching
    motion_matrix = burst_align(ref_image_index, downsampled_raw_images)
    
    # Upsample the motion matrix to the original image size
    motion_matrix = upsample_image(motion_matrix, raw_images.shape[1], raw_images.shape[2]) * 2

    aligned_burst_patches = align(motion_matrix, raw_images)
    align_time = time.time() - start
    print("Time taken for alignment: ", align_time)

    # temporal and spatial denoising
    print("Merging...")
    start = time.time()
    final_merged_frame = merge_images(aligned_burst_patches, ref_image_index)

    final_merged_bayer = merge_patches(final_merged_frame)
    merge_time = time.time() - start
    print("Time taken: ", merge_time)

    total_time = memory_io_time_serial + ref_image_selection_time + align_time + merge_time
    print("Total time taken: ", total_time)
    return memory_io_time_serial, ref_image_selection_time, align_time, merge_time, total_time

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

    print("Aligning...")
    start = time.time()
    downsampled_raw_images = np.array([gaussian_downsample(raw_image, 2) for raw_image in raw_images])

    # Generate alignment matrix using pyramid block matching
    motion_matrix = burst_align(ref_image_index, downsampled_raw_images, parallel=True)

    # Upsample the motion matrix to the original image size
    motion_matrix = upsample_image(motion_matrix, raw_images.shape[1], raw_images.shape[2]) * 2

    aligned_burst_patches = parallel_align(motion_matrix, raw_images)

    align_time = time.time() - start
    print("Time taken for parallel alignment: ", align_time)

    print("Merging...")
    # temporal and spatial denoising
    start = time.time()
    aligned_burst_patches = np.array(aligned_burst_patches)
    final_merged_frame = aligned_burst_patches[0]
    final_merged_frame = parallel_merge_images(aligned_burst_patches, ref_image_index, final_merged_frame)

    final_merged_bayer = merge_patches(final_merged_frame)
    merge_time = time.time() - start

    total_time = memory_io_time_parallel + ref_image_selection_time + align_time + merge_time
    print("Total time taken: ", total_time)
    return memory_io_time_parallel, ref_image_selection_time, align_time, merge_time, total_time
    
def main():
    num_trials = 5
    image_name = "c1b1_20150226_144326_422"
    
    naive_speed_benchmark_scores = []
    parallel_speed_benchmark_scores = []

    for trial in range(num_trials):
        print("\n\nTrial: ", trial)
        memory_io_time_parallel, ref_image_selection_time_parallel, align_time_parallel, merge_time_parallel, total_time_parallel = parallel_speed_benchmark(image_name)
        memory_io_time_serial, ref_image_selection_time_serial, align_time_serial, merge_time_serial, total_time_serial = naive_speed_benchmark(image_name)

        naive_speed_benchmark_scores.append([memory_io_time_serial, ref_image_selection_time_serial, align_time_serial, merge_time_serial, total_time_serial])
        parallel_speed_benchmark_scores.append([memory_io_time_parallel, ref_image_selection_time_parallel, align_time_parallel, merge_time_parallel, total_time_parallel])

    # min speed from benchmark
    naive_speed_benchmark_scores = np.array(naive_speed_benchmark_scores)
    parallel_speed_benchmark_scores = np.array(parallel_speed_benchmark_scores)
    
    naive_speed_benchmark_scores = np.min(naive_speed_benchmark_scores, axis=0)
    parallel_speed_benchmark_scores = np.min(parallel_speed_benchmark_scores, axis=0)
    
    print("Stages: Memory I/O, Reference Image Selection, Alignment, Merge, Total")
    print("Naive speed benchmark: ", naive_speed_benchmark_scores)
    print("Parallel speed benchmark: ", parallel_speed_benchmark_scores)

if __name__ == '__main__':
    main()