import os, subprocess
import cv2
import numpy as np
import rasterio
from rasterio import windows
import geopandas as gpd
from scipy import ndimage
from scipy import stats
import time
import random
from random import randint
from collections import Counter
from datetime import timedelta
from tqdm.notebook import tqdm
from collections import deque


import os
import cv2
import numpy as np
import rasterio
from scipy import ndimage
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from src import utils
from glob import glob
import psutil

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

def regrow_to_original_size_with_limited_looping(clipped_labels_with_background, labels_from_last_iteration, start_index, BORDER = 1):
    ds = clipped_labels_with_background.copy()
    rem_border = []
    iteration = 1
    regrow_kernel = 1 # One side of the kernel
    while True:
        borders_left = 0
        xmax, ymax = ds.shape
        ds_prev = ds.copy()
        ns = np.nonzero((ds >= BORDER) & (ds < start_index))
        to_loop_over = list(zip(ns[0], ns[1]))
        for (i,j) in to_loop_over:
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
            print(f"no border pixels are left")
            break
        if (iteration >= 10) and (len(set(rem_border[-2:])) == 1):
            print(rem_border[-8:])
            print("the border pixels did not change for the last 3 iterations")
            break
        print(f'Border left after iteration {iteration}: {borders_left}')
        rem_border.append(borders_left)
        iteration+=1
    return ds

def regrow_to_original_size(clipped_labels_with_background, labels_from_last_iteration, start_index, BORDER = 1):
    ds = clipped_labels_with_background.copy()
    MAX_ITERATIONS = 25
    rem_border = []
    iteration = 1
    regrow_kernel = 1 # One side of the kernel
    while True:
        borders_left = 0
        # print("iteration number is: ", iteration)
        xmax, ymax = ds.shape
        ds_prev = ds.copy()
        for i in range(0, xmax):
            for j in range(0, ymax):
                # There is a bug in this code. Image an instance is split into two. And both the instances are now considered for further splitting. Now in the next erosion cycle, we get ride of one this this instance.
                # With the current approach we will eat this instance with the left over instance.
                # Solution: When deciding upon the new label, only consider intances that have the same base label.
                # We will need the labels from the last instance.
                if ds[i,j] >= BORDER and ds[i,j] < start_index: # It same as BORDER for this cycle;

                    neighbors = ds_prev[max(i-regrow_kernel, 0):min(i+regrow_kernel+1,xmax) , max(j-regrow_kernel,0):min(j+regrow_kernel+1,ymax)]
                    old_neighbor_label = labels_from_last_iteration[max(i-regrow_kernel, 0):min(i+regrow_kernel+1,xmax), max(j-regrow_kernel,0):min(j+regrow_kernel+1,ymax)]
                    # print(ds_prev[i,j], neighbors, old_neighbor_label)

                    old_neighbor_label = np.where(old_neighbor_label == ds[i,j], 1, 0)
                    neighbors = neighbors * old_neighbor_label
                    # print(ds_prev[i,j], neighbors, old_neighbor_label)
                    ms = major_instance_in_neighbourhood( neighbors.flatten(), start_index, BORDER)

                    # print(ds_prev[i,j], BORDER,start_index, ms)
                    if ms == None:
                        borders_left += 1
                    else:
                        ds[i,j] = ms
        if borders_left == 0:
            print(f"no border pixels are left")
            break
        if (iteration >= 10) and (len(set(rem_border[-2:])) == 1):
            print(rem_border[-8:])
            print("the border pixels did not change for the last 3 iterations")
            break
        print(f'Border left after iteration {iteration}: {borders_left}')
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
def iteratively_erode_and_separate_objects(img_grey, size_thresh, clip_distance_list, min_instance_size, BORDER = 1, BACKGROUND = 0):
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
        print(f'Using distance threshold of {cdt} pixels')
        dist_transform = cv2.distanceTransform(instance_to_split_further.astype(np.uint8), cv2.DIST_L2,5)

        _, clipped_image = cv2.threshold(dist_transform, cdt, 1, 0)
        clipped_labels, _ = ndimage.label(clipped_image)
        clipped_labels = cleanup_labeled_image(clipped_labels, min_instance_size, BACKGROUND)

        # Move all the newly discovered labels by that order
        clipped_labels[clipped_labels > 0] += first_free_index
        print(np.unique(clipped_labels))
        new_first_free_index = np.max(clipped_labels) + 1
        clipped_labels_with_background = np.where( clipped_labels > 0, clipped_labels, instance_to_split_further)
        regrown_labels = regrow_to_original_size_with_limited_looping(clipped_labels_with_background, instance_to_split_further, first_free_index, BORDER = 1)
        print(f'First free index {first_free_index}, new_first_free_index {new_first_free_index}')
        intermediate_shrunk.append(clipped_labels_with_background)
        intermediate_regrown.append(regrown_labels)
        # # plt.subplot(121)
        # # plt.imshow(clipped_labels_with_background, cmap='jet')
        # # plt.subplot(122)
        # # plt.imshow(regrown_labels, cmap='jet')
        # # plt.show()

        assert not (regrown_labels==clipped_labels_with_background).all()

        # Three cases exists; There is no change to a instance but it gets a new label ->  Add to final if smaller than threshold, or give another try
        # 2. An instance is now split into multiple -> Add to final if smaller than threshold, or give another try
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

def separate_images_in_dir(input_dir, image_file_prefix, image_file_type, output_dir, max_filter_size ,centers_only, force_overwrite, cpu_count):
    size_thresh = 500
    clip_distance_list = np.arange(3, 16, 2)
    min_instance_size = 4

    # Get all input image paths
    files = glob( f"{input_dir}/{image_file_prefix}*{image_file_type}" )
    if len(files) == 0:
        raise Exception('No images found in the specified folder!')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file in files:
        print(f'Analysing {file}')
        img, meta = utils.read_image(file)
        out_split_instances =  f"{output_dir}/erosion_split_{file.split('/')[-1]}"
        if not os.path.isfile(out_split_instances) or force_overwrite:
            sp, intermediate_shrunk, intermediate_regrown = iteratively_erode_and_separate_objects(img, size_thresh, clip_distance_list, min_instance_size )
            print(f'Split instances written to {out_split_instances} for {file}')
            utils.save_image(sp, out_split_instances, meta)

if __name__ == '__main__':
    args = utils.get_args()
    phy_cpu = psutil.cpu_count(logical = False)
    if args.cpu == -1 or phy_cpu < args.cpu:
        cpu_count = phy_cpu
    else:
        cpu_count = args.cpu
    ray.init(num_cpus = cpu_count) # Number of cpu/gpus should be specified here, e.g. num_cpus = 4

    separate_images_in_dir(args.input_dir, args.image_file_prefix, args.image_file_type, args.output_dir, args.max_filter_size ,args.save_only_centers, args.force_overwrite, cpu_count)

