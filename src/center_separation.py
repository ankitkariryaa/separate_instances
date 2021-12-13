import os
import sys
import logging
from scipy import ndimage
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from glob import glob
from numba import jit

from src import utils


def clustering_based_neigbourhood_cleanup(x, eps=10):
    cl_dbscan = DBSCAN(eps=eps, min_samples=1)

    # Sometimes multi-values close by would have the same distance from the boundary
    # Thus the max filter may have multiple centers for an instance
    # Here we filter the duplicates
    unq_vals, unq_counts = np.unique(x, return_counts=True)
    uir = []
    uic = []
    for vl, cn in zip(unq_vals, unq_counts):
        if vl != 0:
            if cn == 1:  # Keep that single element
                idx = np.nonzero(x == vl)
                uir.append(idx[0][0])
                uic.append(idx[1][0])
            elif cn > 1:
                idx = np.nonzero(x == vl)
                idxp = list(zip(idx[0], idx[1]))

                # Cluster nearby values
                clustering = cl_dbscan.fit(idxp)
                clb = list(clustering.labels_)

                # Keep only the first element from each cluster
                fi = []
                for i in range(max(clustering.labels_) + 1):
                    fi.append(clb.index(i))

                uir.extend([idxp[i][0] for i in fi])
                uic.extend([idxp[i][1] for i in fi])
    # Zeros array to filter x and keep only one value per cluster
    flx = np.zeros_like(x)
    flx[uir, uic] = 1

    return x * flx  # Return back original value


@jit(nopython=True)
def kernel_based_neighbourhood_cleanup(center_points_with_duplicates, cleanup_kernel):
    final_centers = center_points_with_duplicates.copy()
    ax = np.nonzero(center_points_with_duplicates >= 0)
    points = list(zip(ax[0], ax[1]))
    xmax, ymax = final_centers.shape
    for (i, j) in points:
        if final_centers[i, j] > 0:
            v = final_centers[i, j]
            # Remove other centers in the neightbourhood
            final_centers[max(i - cleanup_kernel, 0): min(i + cleanup_kernel + 1, xmax),
                          max(j - cleanup_kernel, 0): min(j + cleanup_kernel + 1, ymax)] = 0
            final_centers[i, j] = v  # Keep the original value
    return final_centers

# Thanks for Mario Botsch, I still remember the Bresenham algorithm


@jit(nopython=True)
def connecting_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    xf = 1 if x2 > x1 else -1
    yf = 1 if y2 > y1 else -1
    point_list_x = [x1]
    point_list_y = [y1]
    while y2 != y1 or x2 != x1:
        if yf*(y2 - y1) >= xf*(x2-x1):
            y1 += yf
        else:
            x1 += xf
        point_list_x.append(x1)
        point_list_y.append(y1)
    return point_list_x, point_list_y


@jit(nopython=True)
def gaps_on_connecting_linesegment(p1, p2, img):
    (xs, ys) = connecting_points(p1, p2)
    # Numba does not like numpy indexing so switching to a loop that numpy can optimize
    # ln = img[xs, ys]
    ln = []
    for i in range(len(xs)):
        ln.append(img[xs[i], ys[i]])

    r = np.count_nonzero(np.array(ln) == 0)
    return r


@jit(nopython=True)
def get_distance_between_points(p1, p2, distance_metric):
    if distance_metric == 'euclidean':
        d = np.sqrt(((p1[0] - p2[0]) * (p1[0] - p2[0])) + ((p1[1] - p2[1]) * (p1[1] - p2[1])))
    elif distance_metric == 'manhattan':
        d = np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
    else:
        raise NotImplementedError()
    return d


@jit(nopython=True)
def get_closest_center(
        labeled_centers_on_instance, center_points, point, edge_distance_of_center_points, img, distance_metric,
        no_connection_penalty=10):
    if len(center_points) == 0:
        return 1
    elif len(center_points) == 1:
        return labeled_centers_on_instance[center_points[0]]
    selected_center = center_points[0]
    dt = get_distance_between_points(selected_center, point, distance_metric)
    weight = 1 / edge_distance_of_center_points[selected_center]
    gaps_on_linesegment = gaps_on_connecting_linesegment(selected_center, point, img)
    min_dist = (weight * dt) + (gaps_on_linesegment * no_connection_penalty)

    # Iterate over all centers until we find closest one
    for center in center_points[1:]:
        dt = get_distance_between_points(center, point, distance_metric)
        weight = 1 / edge_distance_of_center_points[center]
        gaps_on_linesegment = gaps_on_connecting_linesegment(center, point, img)
        dist = (weight * dt) + (gaps_on_linesegment * no_connection_penalty)
        if dist < min_dist:
            min_dist = dist
            selected_center = center

    return labeled_centers_on_instance[selected_center]


def get_centers_as_points(centers_image):
    return list(zip(*np.nonzero(centers_image > 0)))


@jit(nopython=True)
def find_pixel_class_by_distance(labeled_centers, labeled_grey_im, max_center_points, distance_metric='euclidean'):
    final_img = np.zeros_like(labeled_grey_im)
    unqi = np.unique(labeled_grey_im)
    for u in unqi[1:]:  # Iterate over all instances expect the background
        # That's one instance as per original labels
        original_instance = np.where(labeled_grey_im == u, 1, 0)
        labeled_centers_on_instance = labeled_centers * original_instance  # Get the predicted centers on this instance
        centers_as_points = list(zip(*np.nonzero(labeled_centers_on_instance > 0)))  # Get their coordinates

        points_left_to_label = np.nonzero(original_instance == 1)
        for index in range(len(points_left_to_label[0])):
            i = points_left_to_label[0][index]
            j = points_left_to_label[1][index]
            cc = get_closest_center(labeled_centers_on_instance, centers_as_points, (i, j),
                                    max_center_points, labeled_grey_im, distance_metric)
            final_img[i, j] = cc
    return final_img


def separate_objects(img_grey, max_filter_size, centers_only):
    max_filter_size = max_filter_size  # Default 15
    dist_transform = ndimage.distance_transform_edt(img_grey)

    m_img = ndimage.maximum_filter(dist_transform, size=max_filter_size)
    max_center_points_with_duplicates = np.where(m_img == dist_transform, m_img, 0)
    max_center_points = kernel_based_neighbourhood_cleanup(max_center_points_with_duplicates, int(max_filter_size/2))
    # max_center_points is the center points with distance from the edge

    m_centers = np.where(max_center_points > 0, 1, 0)
    m_labeled_centers, _ = ndimage.label(m_centers)
    if centers_only:
        return m_labeled_centers, max_center_points, None
    else:
        # This is the slow step
        labeled_img_grey, _ = ndimage.label(img_grey)
        relabed_img = find_pixel_class_by_distance(m_labeled_centers, labeled_img_grey, max_center_points, 'euclidean')
        relabed_img = utils.majority_filter_based_upon_original_labels(relabed_img, labeled_img_grey, 3)
        return m_labeled_centers, max_center_points, relabed_img


def separate_images_in_dir(
        input_dir, file_prefix, file_type, output_dir, max_filter_size, centers_only, force_overwrite):
    # Get all input image paths
    files = glob(f"{input_dir}/{file_prefix}*{file_type}")
    if len(files) == 0:
        raise Exception('No images found in the specified folder!')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file in files:
        logging.info(f'Analysing {file}')
        img, meta = utils.read_image(file)
        out_center = f"{output_dir}/centers_{file.split('/')[-1]}"
        out_split_instances = f"{output_dir}/center_based_relabeled_{file.split('/')[-1]}"
        if not os.path.isfile(out_center) or force_overwrite:
            m_labeled_centers, max_center_points, relabed_img = separate_objects(img, max_filter_size, centers_only)
            logging.info(f'Centers written to {out_center} for {file}')
            utils.save_image(m_labeled_centers, out_center, meta)
            if not centers_only:
                logging.info(f'Relablled image written to {out_split_instances} for {file}')
                utils.save_image(relabed_img, out_split_instances, meta)


if __name__ == '__main__':
    args = utils.get_args('center_based_separation')
    logs_file = utils.initialize_log_dir(args.log_dir)
    print(f'Writing logs to {logs_file}')

    try:
        separate_images_in_dir(args.input_dir, args.file_prefix, args.file_type,
                               args.output_dir, args.max_filter_size, args.save_only_centers, args.force_overwrite)
    except:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
