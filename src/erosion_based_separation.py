import os
import sys
import cv2
import random
import psutil
import logging
import time

from scipy import ndimage
from pathlib import Path
from glob import glob
from collections import Counter
import numpy as np

from src import utils

import ray

def cleanup_labeled_image(lb_img, min_instance_size, BACKGROUND = 0):
    unqs, cnts = np.unique(lb_img, return_counts=True)
    for u in range(1, len(unqs)):
        if cnts[u] < min_instance_size:
            lb_img =  np.where(lb_img == u, BACKGROUND, lb_img)
    return lb_img

# Majority vote count function
def major_instance_in_neighbourhood(pxs, start_index, BORDER = 1):
    cp = []
    for p in pxs:
        if p >= start_index:
            cp.append(p)
    if len(cp) == 0:
        return None
    cn = Counter(cp)
    mn = cn.most_common(2)  # gives the first max. 4 most common classes
    if len(mn) == 1:
        return mn[0][0]

    m1i, m1c = mn[0]
    m2i, m2c = mn[1]
    if m1c == m2c:
        x = [m1i, m2i]
        random.shuffle(x)
        return x[0]
    return m1i

@ray.remote
def parallel_get_pixel_class(points, ds_prev, labels_from_last_iteration, regrow_kernel, xmax, ymax, start_index, BORDER):
    results = {} # A map of location (i,j) and it's corresponding label
    for (i,j) in points:
        neighbors = ds_prev[max(i-regrow_kernel, 0):min(i+regrow_kernel+1,xmax) , max(j-regrow_kernel,0):min(j+regrow_kernel+1,ymax)]
        old_neighbor_label = labels_from_last_iteration[max(i-regrow_kernel, 0):min(i+regrow_kernel+1,xmax), max(j-regrow_kernel,0):min(j+regrow_kernel+1,ymax)]

        old_neighbor_label = np.where(old_neighbor_label == ds_prev[i,j], 1, 0) # It should be same as using ds[i,j]
        neighbors = neighbors * old_neighbor_label
        ms = major_instance_in_neighbourhood( neighbors.flatten(), start_index, BORDER)

        if ms is not None:
            results[(i,j)] = ms

    return results

def parallel_regrow_to_original_size(clipped_labels_with_background, labels_from_last_iteration, start_index, cpu_count, BORDER = 1):
    ds = clipped_labels_with_background.copy()
    rem_border = []
    iteration = 1
    regrow_kernel = 1 # One side of the kernel
    xmax, ymax = ds.shape
    while True:
        borders_left = 0
        ds_prev = ds.copy()
        ax = np.nonzero((ds >= BORDER) & (ds < start_index))
        points = list(zip(ax[0], ax[1]))

        splits = np.arange(1, len(points), int(len(points)/cpu_count) )  # Split instances for parallel processing; 0 - background is ignored
        indx_s = [(splits[i-1], splits[i]) for i in range(1, len(splits)) ]

        # Do the main processing in parallel
        result_ids = [parallel_get_pixel_class.remote(points[i:j], ds_prev, labels_from_last_iteration, regrow_kernel, xmax, ymax, start_index, BORDER) for (i,j) in indx_s]
        results = ray.get(result_ids)

        # Accumulate the results in a single dictionary
        labelled_points = results[0]
        for r in results[1:]:
            labelled_points.update(r)

        borders_left = len(points) - len(labelled_points)

        for ((i,j), cc) in labelled_points.items():
            ds[i,j] = cc

        if borders_left == 0:
            logging.info(f"no border pixels are left")
            break
        if (iteration >= 10) and (len(set(rem_border[-2:])) == 1):
            logging.info(rem_border[-8:])
            logging.info("the border pixels did not change for the last 3 iterations")
            break
        logging.info(f'Border left after iteration {iteration}: {borders_left}')
        rem_border.append(borders_left)
        iteration+=1
    return ds

def regrow_to_original_size(clipped_labels_with_background, labels_from_last_iteration, start_index, BORDER = 1):
    ds = clipped_labels_with_background.copy()
    rem_border = []
    iteration = 1
    regrow_kernel = 1 # One side of the kernel
    while True:
        borders_left = 0
        xmax, ymax = ds.shape
        ds_prev = ds.copy()
        ax = np.nonzero((ds >= BORDER) & (ds < start_index))
        points = list(zip(ax[0], ax[1]))
        for (i,j) in points:
            neighbors = ds_prev[max(i-regrow_kernel, 0):min(i+regrow_kernel+1,xmax) , max(j-regrow_kernel,0):min(j+regrow_kernel+1,ymax)]
            old_neighbor_label = labels_from_last_iteration[max(i-regrow_kernel, 0):min(i+regrow_kernel+1,xmax), max(j-regrow_kernel,0):min(j+regrow_kernel+1,ymax)]

            old_neighbor_label = np.where(old_neighbor_label == ds[i,j], 1, 0)
            neighbors = neighbors * old_neighbor_label
            ms = major_instance_in_neighbourhood( neighbors.flatten(), start_index, BORDER)

            if ms == None:
                borders_left += 1
            else:
                ds[i,j] = ms
        if borders_left == 0:
            logging.info(f"no border pixels are left")
            break
        if (iteration >= 10) and (len(set(rem_border[-2:])) == 1):
            logging.info(rem_border[-8:])
            logging.info("the border pixels did not change for the last 3 iterations")
            break
        logging.info(f'Border left after iteration {iteration}: {borders_left}')
        rem_border.append(borders_left)
        iteration+=1
    return ds

def is_continues(unqs):
    cn = True
    for i in range(1, len(unqs)):
        if unqs[i] != unqs[i -1] + 1:
            cn = False
    return cn

def convert_to_continues_numberiung(lb_img_ori):
    lb_img = lb_img_ori.copy()
    unqs = np.unique(lb_img)
    i = 1
    if is_continues(unqs):
        return lb_img
    else:
        for u in unqs[1:]:
            if i != u:
                lb_img = np.where(lb_img == u, i, lb_img)
            i += 1
    return lb_img

# Refactored code

#1. Apply distance transform and clip on a low threshold
#2. Assign new ids to the clipped instances and regrow them to their original size
#3. Take the smaller instances to the final image and repeat the process for larger instance with higher threshold
def iteratively_erode_and_separate_objects(img_grey, size_thresh, clip_distance_list, min_instance_size, cpu_count, BORDER = 1, BACKGROUND = 0):
    """Separate into smaller trees

    Args:
        img_grey (2D np array): Segmentation mask on which the spliting is performed
        size_thresh (int, optional): Minimum size of the instance to be considered for spliting. Defaults to 500.
        clip_distance_list ([int], optional): Distance in pixel used for separating the images.  Defaults to np.arange(0.05, .31, .05).
    Returns:
        final_image (2D nd array): The relabelled image
        intermediate_shrunk ([2D nd array]): List of intermediate shrunk image
        intermediate_regrown ([2D nd array]): List of intermediate regrown images
    """

    intermediate_shrunk = []
    intermediate_regrown = []

    final_image, label_max_ori = ndimage.label(img_grey)

    final_image[final_image == BORDER] = label_max_ori +1
    label_max_ori += 1
    first_free_index = label_max_ori + 1 # The one after the Last index, so the first free index; 1 is unnecessary
    instance_to_split_further = final_image.copy()

    # Each progressive erosion relies on the output of the last step; There it is, unfortunately, not possible to parallelize here.
    for cdt in clip_distance_list:
        logging.info(f'Using distance threshold of {cdt} pixels')
        dist_transform = cv2.distanceTransform(instance_to_split_further.astype(np.uint8), cv2.DIST_L2,5)

        _, clipped_image = cv2.threshold(dist_transform, cdt, 1, 0)
        clipped_labels, _ = ndimage.label(clipped_image)
        clipped_labels = cleanup_labeled_image(clipped_labels, min_instance_size, BACKGROUND)

        # Move all the newly discovered labels by that order
        clipped_labels[clipped_labels > 0] += first_free_index
        logging.info(np.unique(clipped_labels))
        new_first_free_index = np.max(clipped_labels) + 1
        clipped_labels_with_background = np.where( clipped_labels > 0, clipped_labels, instance_to_split_further)

        if cpu_count <= 1:
            logging.info(f"Regrowing labels on a single core.")
            regrown_labels = regrow_to_original_size(clipped_labels_with_background, instance_to_split_further, first_free_index, BORDER = 1)
        else:
            logging.info(f"Regrowing labels using {cpu_count} CPU cores.")
            regrown_labels = parallel_regrow_to_original_size(clipped_labels_with_background, instance_to_split_further, first_free_index, cpu_count, BORDER = 1)
        logging.info(f'First free index {first_free_index}, new_first_free_index {new_first_free_index}')
        intermediate_shrunk.append(clipped_labels_with_background)
        intermediate_regrown.append(regrown_labels)

        assert not (regrown_labels==clipped_labels_with_background).all()

        # Three cases exists; There is no change to a instance but it gets a new label ->  Add to final if smaller than threshold, or give another try; Use the newer label
        # 2. An instance is now split into multiple -> For all splits; Add to final if smaller than threshold, or give another try
        # 3. An instance is now removed by the threshold -> Final should already contain it.

        unq_regrwn = np.unique(regrown_labels)
        instance_to_split_further = np.zeros_like(final_image)

        for u in unq_regrwn[1:]:
            ins_u = np.where(regrown_labels == u, u, 0)
            if np.count_nonzero(ins_u) > size_thresh:
                instance_to_split_further = np.where(ins_u > 0, ins_u, instance_to_split_further)
            else:
                final_image = np.where(ins_u > 0, ins_u, final_image)

        first_free_index = new_first_free_index

    # Add the final batch of images to the final image!
    unq_regrwn = np.unique(instance_to_split_further)
    for u in unq_regrwn[1:]:
        ins_u = np.where(instance_to_split_further == u, u, 0)
        final_image = np.where(ins_u > 0, ins_u, final_image)

    return final_image, intermediate_shrunk, intermediate_regrown

def separate_images_in_dir(input_dir, image_file_prefix, image_file_type, output_dir, size_thresh ,min_eroded_instance_size, clip_distance_list, force_overwrite, cpu_count):
    # size_thresh = 500
    # clip_distance_list = np.arange(3, 16, 2)
    # min_instance_size = 4

    # Get all input image paths
    files = glob( f"{input_dir}/{image_file_prefix}*{image_file_type}" )
    if len(files) == 0:
        raise Exception('No images found in the specified folder!')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file in files:
        print(f'Analysing {file}')
        logging.info(f'Analysing {file}')
        img, meta = utils.read_image(file)
        out_split_instances =  f"{output_dir}/erosion_split_{file.split('/')[-1]}"
        if not os.path.isfile(out_split_instances) or force_overwrite:
            start = time.time()
            sp, intermediate_shrunk, intermediate_regrown = iteratively_erode_and_separate_objects(img, size_thresh, clip_distance_list, min_eroded_instance_size, cpu_count)
            logging.info(f"Instance splitting using {cpu_count} CPU cores performed in {time.time() - start} sec")
            utils.save_image(sp, out_split_instances, meta)
            logging.info(f'Split instances written to {out_split_instances} for {file}')

if __name__ == '__main__':
    args = utils.get_args('erosion_based_separation')
    logs_file = utils.initialize_log_dir(args.log_dir)
    print(f'Writing logs to {logs_file}')
    phy_cpu = psutil.cpu_count(logical = False)
    if args.cpu == -1 or phy_cpu < args.cpu:
        cpu_count = phy_cpu
    else:
        cpu_count = args.cpu
    try:
        ray.init(num_cpus = cpu_count) # Number of cpu/gpus should be specified here, e.g. num_cpus = 4
        separate_images_in_dir(args.input_dir, args.image_file_prefix, args.image_file_type,
            args.output_dir, args.size_thresh, args.min_eroded_instance_size, args.clip_distance_list,
            args.force_overwrite, cpu_count)
        ray.shutdown()
    except:
        ray.shutdown()
        logging.info('Run interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

