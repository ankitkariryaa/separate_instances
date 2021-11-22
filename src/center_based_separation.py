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

def remove_neighbouring_pixels_with_same_value(x, eps = 10):
    cl_dbscan = DBSCAN(eps=eps, min_samples=1)

    # Sometimes multi-values close by would have the same distance from the boundary
    # Thus the max filter may have multiple centers for an instance
    # Here we filter the duplicates
    unq_vals, unq_counts = np.unique(x, return_counts=True)
    uir = []
    uic = []
    for vl, cn in zip(unq_vals, unq_counts):
        if vl != 0 :
            if cn == 1: # Keep that single element
                idx = np.nonzero(x==vl)
                uir.append(idx[0][0])
                uic.append(idx[1][0])
            elif cn >1 : #
                idx = np.nonzero(x==vl)
                idxp = list(zip(idx[0], idx[1]))

                # Cluster nearby values
                clustering = cl_dbscan.fit(idxp)
                clb = list(clustering.labels_)

                # Keep only the first element from each cluster
                fi = []
                for i in range(max(clustering.labels_) +1):
                    fi.append(clb.index(i))

                uir.extend([idxp[i][0] for i in fi])
                uic.extend([idxp[i][1] for i in fi])
    # Zeros array to filter x and keep only one value per cluster
    flx = np.zeros_like(x)
    flx[uir, uic] = 1

    return x * flx

def split_in_n_instances(labeled_instance, centers_on_instance, current_cntr, lg, background_img, erosion_kernel = np.ones((3,3), np.uint8)):
    center_count = np.sum(centers_on_instance)
    # print(f'Split in n called! {lg}, {np.count_nonzero(labeled_instance)}')

    lit = labeled_instance.copy()
    temp_imgs = []
    while center_count > 1 and np.count_nonzero(lit) > 0: # n instance left to split

        lit = cv2.erode(lit.astype(np.uint8), erosion_kernel, iterations = 1)
        im_diff = labeled_instance - lit

        # Add newely eroded pixels to background
        background_img = np.where(im_diff != 0, 1, background_img)
        # print('Sum of eroded image', np.sum(lit))
        niml, nlm  = ndimage.label(lit)
        if nlm == 1:
            continue # The erosion was useless, let's try again
        elif nlm >= 2: # Erosion created atleast two instances
            for lbi in range(1, nlm + 1):
                lbimg = np.where(niml == lbi, 1, 0)
                nci = centers_on_instance * lbimg
                cc = np.sum(nci)
                if cc == 0: # This is a useless instance, since it doesn't overlap with one of predicted centers.
                    pass
                elif cc == 1: # Yes, finally! We have an instance that separates one center
                    temp_imgs.append(lbimg * current_cntr)
                    current_cntr += 1
                else:
                    sti, current_cntr, background_img = split_in_n_instances(lbimg, nci, current_cntr, lg, background_img, erosion_kernel)
                    temp_imgs.extend(sti)
            return temp_imgs, current_cntr, background_img
    return temp_imgs, current_cntr, background_img


def combine_images(temp_images, shp):
    ni = np.zeros(shp)
    for i in temp_images:
        ni += i
    return ni

def subdivide_labels(labeled_im, img_grey, current_cntr):
    background_img = np.zeros_like(img_grey)
    temp_images = []
    for l in tqdm(range(2, labels_max + 2)):
        # Get that instance
        instance_to_split = np.where(labeled_im == l, 1, 0)
        centers_on_instance = m_centers * instance_to_split # Img will be zero everywhere expect for the instance
        center_count  = np.sum(centers_on_instance)

        if center_count <= 1: # No need to split
            temp_images.append(instance_to_split*l) # Give it back it's original label
            # print(f'No need to split {l}')
        else:
            # print(f'Spliting {l} into {center_count}')
            ti, current_cntr, background_img = split_in_n_instances(instance_to_split, centers_on_instance, current_cntr, l, background_img)
            temp_images.extend(ti)

    instance_composite = combine_images(temp_images, img_grey.shape)
    instance_composite, lm = ndimage.label(instance_composite) # Relabel
    instance_composite[ instance_composite == 1] = lm + 1 # We move the first label to keep a number for pure background
    return instance_composite, background_img

def grow_instances_to_nearest_border(inc, background_img, dilation_kernel= np.ones((3,3), np.uint8)):
    # Grow instance on the eroded background until you reach the reall background ( i.e 0)
    unqi = np.unique(inc)
    final_instances = np.zeros_like(inc)
    for u in unqi:
        ig = np.where(inc == u, 1, 0).astype(np.uint8)
        if u != 0 and u != 1: # Exclude 0 and 1
            while True:
                c = 0
                gri = cv2.dilate(ig, dilation_kernel, iterations = 1)
                idiff = gri - ig
                # print(np.sum(idiff))
                bd = idiff * background_img # Same as bd = np.where(idiff != 0, background_img, 0)
                if np.count_nonzero(bd) < np.count_nonzero(idiff):
                    break
                else:
                    ig = gri
                    c += 1
        final_instances = np.where(final_instances == 0, ig * u, final_instances)
    return final_instances

# Thanks for Mario Botsch, I still remember the easiest way of finding the points on a line segment connecting two points with a 8bit processor
def connecting_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    xf = 1 if x2 > x1 else -1
    yf = 1 if y2 > y1 else -1
    point_list = [p1]
    while y2 != y1 or x2 != x1:
        if yf*(y2 -y1) >= xf*(x2-x1):
            y1 += yf
        else:
            x1 += xf
        point_list.append((x1,y1))
    return point_list

def gaps_on_connecting_linesegment(p1, p2, img):
    cp = connecting_points(p1, p2)
    xs, ys = zip(*cp)
    return np.count_nonzero(img[xs, ys] == 0)

def get_distance_between_points(p1, p2, distance_metric):
    if distance_metric == 'euclidean':
        d = np.sqrt( ((p1[0] - p2[0]) * (p1[0] - p2[0])) + ((p1[1] - p2[1]) * (p1[1] - p2[1])))
    elif distance_metric == 'manhattan':
        d = np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
    else:
        raise NotImplementedError()
    return d

def get_closest_center(labeled_centers_on_instance, center_points, point, height_center_points, img, distance_metric, no_connection_penalty = 10000):
    if len(center_points) == 0:
        return 1
    elif len(center_points) == 1:
        return labeled_centers_on_instance[center_points[0]]
    selected_center = center_points[0]
    dt = get_distance_between_points(selected_center, point, distance_metric)
    weight = 1 / height_center_points[selected_center]
    gaps_on_linesegment = gaps_on_connecting_linesegment(selected_center, point, img)
    min_dist = (weight * dt) + (gaps_on_linesegment * no_connection_penalty)

    # Iterate over all centers until we find closest one
    for center in center_points[1:]:
        dt = get_distance_between_points(center, point, distance_metric)
        weight = 1 / height_center_points[center]
        gaps_on_linesegment = gaps_on_connecting_linesegment(center, point, img)
        dist  = (weight * dt) + (gaps_on_linesegment * no_connection_penalty)
        if dist < min_dist:
            min_dist = dist
            selected_center = center

    return labeled_centers_on_instance[selected_center]

@ray.remote
def parallel_split_instance_based_on_centers(labeled_grey_im, unique_instances, labeled_centers, max_center_points, distance_metric ):
    results = {} # A map of location (i,j) and it's corresponding label

    for u in unique_instances:
        ig = np.where(labeled_grey_im == u, 1, 0).astype(np.uint8)
        labeled_centers_on_instance = labeled_centers * ig # Get the centers on this instance
        cp = get_centers_as_points(labeled_centers_on_instance) # Get their coordinates

        ax = np.nonzero(ig == 1)
        points = list(zip(ax[0], ax[1]))
        for (i,j) in points:
            cc = get_closest_center(labeled_centers_on_instance, cp, (i,j), max_center_points, labeled_grey_im, distance_metric)
            results[(i,j)] = cc
    return results


# After reading about the various possibilities related to parallel programming in python, I decided to use Apache ray. The second and quite close candidate was joblib
# Reasons for using Ray
# 1. Support for shared objects (in joblib similar setup would require the user of numpy memmap)
# 2. Actor model (which I have also admired)
# 3. Cluster support
# Must read: https://docs.ray.io/en/latest/auto_examples/tips-for-first-time.html
# See also: https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
def parallel_find_pixel_class_by_distance(labeled_centers, labeled_grey_im, max_center_points, cpu_count, distance_metric = 'euclidean'):

    # We parallelize at this point. Each thread gets n unique instances to label
    unqi = np.unique(labeled_grey_im)
    splits = np.arange(1, len(unqi), int(len(unqi)/cpu_count) )  # Split instances for parallel processing; 0 - background is ignored
    indx_s = [(splits[i-1], splits[i]) for i in range(1, len(splits)) ]

    # Store that large object in the local object store, instead of passing them around
    labeled_centers_id = ray.put(labeled_centers)
    labeled_grey_im_id = ray.put(labeled_grey_im)
    max_center_points_id = ray.put(max_center_points)

    # Do the main processing in parallel
    start = time.time()
    result_ids = [parallel_split_instance_based_on_centers.remote(labeled_grey_im_id, unqi[i:j], labeled_centers_id, max_center_points_id, distance_metric) for (i,j) in indx_s]
    results = ray.get(result_ids)
    print(f"Duration for parallel splitting using {cpu_count} CPU: {time.time() - start}")

    # Accumulate the results in a single dictionary
    labelled_pixels = results[0]
    for r in results[1:]:
        labelled_pixels.update(r)

    final_img = np.zeros_like(labeled_grey_im)
    for ((i,j), cc) in labelled_pixels.items():
        final_img[i,j] = cc
    return final_img

def get_centers_as_points(centers_image):
    cn = np.nonzero(centers_image > 0)
    return list(zip(cn[0], cn[1]))

def find_pixel_class_by_distance(labeled_centers, labeled_grey_im, max_center_points, distance_metric = 'euclidean'):
    final_img = np.zeros_like(labeled_grey_im)
    unqi = np.unique(labeled_grey_im)
    for u in tqdm(unqi[1:]): # Iterate over all instances expect the background
        # That's one instance as per original labels
        ig = np.where(labeled_grey_im == u, 1, 0).astype(np.uint8)
        labeled_centers_on_instance = labeled_centers * ig # Get the centers on this instance
        cp = get_centers_as_points(labeled_centers_on_instance) # Get their coordinates

        ax = np.nonzero(ig == 1)
        points = list(zip(ax[0], ax[1]))
        for (i,j) in points:
            cc = get_closest_center(labeled_centers_on_instance, cp, (i,j), max_center_points, labeled_grey_im, distance_metric)
            final_img[i,j] = cc
    return final_img

def majority_filter(img_ori, voting_kernel = 2):
    img = img_ori.copy()
    xmax, ymax = img.shape
    ax = np.nonzero(img != 0)
    points = list(zip(ax[0], ax[1]))
    for (i,j) in points:
        cn = img_ori[max(i-voting_kernel, 0):min(i+voting_kernel+1,xmax) , max(j-voting_kernel,0):min(j+voting_kernel+1,ymax)]
        unq, unc = np.unique(cn, return_counts=True)
        unqc_sorted_index = np.argsort(-unc)

        mj = unq[unqc_sorted_index][0]
        if mj == 0: # If background is the majority class, then we take the next class
            mj = unq[unqc_sorted_index][1]
        if mj != img[i,j]: # Replace with majority class if it is a lone pixel of this class
            img[i,j] = mj
    return img

def read_image(img_path, fill_holes=True, dtype = np.uint8):
    im = rasterio.open(img_path)
    meta = im.meta
    img = im.read()
    img_grey = img[0,:,:]#.mean(axis=(0))) # Same as [0,:,:], for single band image
    if fill_holes:
        img_grey = ndimage.binary_fill_holes(img_grey)
    return img_grey.astype(dtype), meta

def save_image(img, img_path, meta, dtype = rasterio.uint32):
    meta.update(count=1,
            dtype=dtype,
            TILED='YES',
            COMPRESS='LZW',
            BIGTIFF='IF_SAFER',
            multithread=True,
            NUM_THREADS='ALL_CPUS')
    with rasterio.open(img_path, 'w', **meta) as dst:
        dst.write(img.astype(dtype),1)

def separate_objects(img_grey, max_filter_size, centers_only, cpu_count):
    time1 = time.time()
    max_filter_size = max_filter_size # Default 12
    dist_transform = cv2.distanceTransform(img_grey, cv2.DIST_L2,5)
    m_img = ndimage.maximum_filter(dist_transform, size=max_filter_size)
    max_center_points_with_duplicates = np.where(m_img == dist_transform, m_img, 0)
    max_center_points = remove_neighbouring_pixels_with_same_value(max_center_points_with_duplicates, 1.25 * max_filter_size)
    m_centers = np.where(max_center_points>0,1,0)
    m_centers_to_labels, _ = ndimage.label(m_centers)
    print(f'Centers found in {(time.time()-time1)*1000.0} ms!')
    if centers_only:
        return m_centers_to_labels, None
    else:
        # This is the slow step
        labeled_im, _ = ndimage.label(img_grey)
        if cpu_count <= 1: # Single threaded
            relabed_img = find_pixel_class_by_distance(m_centers_to_labels, labeled_im, max_center_points, 'euclidean')
        else: # Multi threaded
            relabed_img = parallel_find_pixel_class_by_distance(m_centers_to_labels, labeled_im, max_center_points, cpu_count, 'euclidean')
        relabed_img = majority_filter(relabed_img)
        return m_centers_to_labels, relabed_img

def separate_images_in_dir(input_dir, image_file_prefix, image_file_type, output_dir, max_filter_size ,centers_only, force_overwrite, cpu_count):
    # Get all input image paths
    files = glob(
         f"{input_dir}/{image_file_prefix}*{image_file_type}"
    )
    if len(files) == 0:
        raise Exception('No images found in the specified folder!')

    for file in files:
        print(f'Analysing {file}')
        img, meta = read_image(file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_center = f"{output_dir}/centers_{file.split('/')[-1]}"
        out_split_instances =  f"{output_dir}/split_{file.split('/')[-1]}"
        if not os.path.isfile(out_center) or force_overwrite:
            m_centers, relabed_img = separate_objects(img, max_filter_size ,centers_only, cpu_count)
            print(f'Centers written to {out_center} for {file}')
            save_image(m_centers, out_center, meta)
            if not centers_only:
                print(f'Relablled image written to {out_split_instances} for {file}')
                save_image(relabed_img, out_split_instances, meta)

if __name__ == '__main__':
    args = utils.get_args()
    phy_cpu = psutil.cpu_count(logical = False)
    if args.cpu == -1 or phy_cpu < args.cpu:
        cpu_count = phy_cpu
    else:
        cpu_count = args.cpu
    ray.init(num_cpus = cpu_count) # Number of cpu/gpus should be specified here, e.g. num_cpus = 4

    separate_images_in_dir(args.input_dir, args.image_file_prefix, args.image_file_type, args.output_dir, args.max_filter_size ,args.save_only_centers, args.force_overwrite, cpu_count)