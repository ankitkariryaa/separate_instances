import logging
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage

from src import utils


@njit(cache=True)
def suppress_nearby_centers(rows, cols, radius):
    """Greedy non-maximum suppression of candidate centers.

    Candidates must be sorted by decreasing distance from the edge, so that the strongest
    candidate in a neighbourhood is the one that is kept. Returns a boolean mask of kept candidates.
    """
    n = len(rows)
    keep = np.ones(n, dtype=np.bool_)
    for a in range(n):
        if not keep[a]:
            continue
        for b in range(a + 1, n):
            if keep[b] and abs(rows[a] - rows[b]) <= radius and abs(cols[a] - cols[b]) <= radius:
                keep[b] = False
    return keep


@njit(cache=True)
def gaps_on_connecting_line(mask, r0, c0, r1, c1):
    """Count the pixels outside the mask on the Bresenham line between (r0, c0) and (r1, c1)."""
    # Thanks for Mario Botsch, I still remember the Bresenham algorithm
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    gaps = 0
    while True:
        if not mask[r0, c0]:
            gaps += 1
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return gaps


@njit(cache=True)
def assign_pixels_to_centers(mask, center_rows, center_cols, center_radius, center_labels, no_connection_penalty):
    """Label each pixel of a single instance with the label of its closest center.

    The distance to a center is weighted by the inverse of the center's distance from the edge (so larger
    objects claim more pixels) and penalized for every pixel outside the instance on the connecting line.
    """
    out = np.zeros(mask.shape, dtype=np.int32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j]:
                continue
            best_dist = np.inf
            best_label = center_labels[0]
            for k in range(len(center_rows)):
                dr = i - center_rows[k]
                dc = j - center_cols[k]
                dist = np.sqrt(dr * dr + dc * dc) / center_radius[k]
                dist += no_connection_penalty * gaps_on_connecting_line(
                    mask, center_rows[k], center_cols[k], i, j)
                if dist < best_dist:
                    best_dist = dist
                    best_label = center_labels[k]
            out[i, j] = best_label
    return out


@njit(cache=True)
def majority_filter_within_instance(labels, mask, voting_kernel):
    """Replace each pixel's label with the most frequent label of the instance in its neighbourhood.

    Only pixels inside the mask vote; ties are resolved in favour of the smallest label.
    """
    out = labels.copy()
    xmax, ymax = labels.shape
    votes = np.empty((2 * voting_kernel + 1) ** 2, dtype=labels.dtype)
    for i in range(xmax):
        for j in range(ymax):
            if not mask[i, j]:
                continue
            n = 0
            for a in range(max(i - voting_kernel, 0), min(i + voting_kernel + 1, xmax)):
                for b in range(max(j - voting_kernel, 0), min(j + voting_kernel + 1, ymax)):
                    if mask[a, b]:
                        votes[n] = labels[a, b]
                        n += 1
            sorted_votes = np.sort(votes[:n])
            majority, majority_count = sorted_votes[0], 0
            current, current_count = sorted_votes[0], 0
            for v in sorted_votes:
                if v == current:
                    current_count += 1
                else:
                    current, current_count = v, 1
                if current_count > majority_count:
                    majority, majority_count = current, current_count
            out[i, j] = majority
    return out


def find_centers(labeled_img, dist_transform, max_filter_size):
    """Find the centers of each instance as the local maxima of the distance transform.

    The maxima are computed per instance, so a large instance can not suppress the centers of a smaller
    neighbouring one, and every instance gets at least one center.

    Returns:
        labeled_centers (2D int32 array): Centers labelled 1..n in raster order, 0 elsewhere.
        center_distances (2D float array): Distance from the edge at the centers, 0 elsewhere.
    """
    center_distances = np.zeros(labeled_img.shape, dtype=np.float64)
    for instance, sl in enumerate(ndimage.find_objects(labeled_img), start=1):
        if sl is None:
            continue
        mask = labeled_img[sl] == instance
        dist = np.where(mask, dist_transform[sl], 0)
        local_max = ndimage.maximum_filter(dist, size=max_filter_size, mode='constant', cval=0)
        rows, cols = np.nonzero(mask & (dist == local_max))

        # Plateaus of equal distance yield multiple maxima close to each other; keep the strongest one
        order = np.argsort(-dist[rows, cols], kind='stable')
        rows, cols = rows[order], cols[order]
        keep = suppress_nearby_centers(rows, cols, max_filter_size // 2)
        rows, cols = rows[keep], cols[keep]
        center_distances[sl][rows, cols] = dist[rows, cols]

    labeled_centers = np.zeros(labeled_img.shape, dtype=np.int32)
    center_points = np.nonzero(center_distances)
    labeled_centers[center_points] = np.arange(1, len(center_points[0]) + 1)
    return labeled_centers, center_distances


def find_pixel_class_by_distance(labeled_centers, labeled_img, center_distances, no_connection_penalty=10,
                                 voting_kernel=3):
    final_img = np.zeros(labeled_img.shape, dtype=np.int32)
    for instance, sl in enumerate(ndimage.find_objects(labeled_img), start=1):
        if sl is None:
            continue
        mask = labeled_img[sl] == instance
        centers = np.where(mask, labeled_centers[sl], 0)
        center_rows, center_cols = np.nonzero(centers)
        center_labels = centers[center_rows, center_cols]

        if len(center_labels) == 1:
            final_img[sl][mask] = center_labels[0]
            continue

        labels = assign_pixels_to_centers(
            mask, center_rows, center_cols, center_distances[sl][center_rows, center_cols], center_labels,
            no_connection_penalty)
        # Remove lone pixels that are assigned to a center that is not in sight
        labels = majority_filter_within_instance(labels, mask, voting_kernel)
        final_img[sl][mask] = labels[mask]
    return final_img


def separate_objects(img_grey, max_filter_size, centers_only, no_connection_penalty=10, voting_kernel=3,
                     connectivity=4):
    """Separate merged instances in a segmentation mask based upon their centers.

    Args:
        img_grey (2D np array): Segmentation mask, non-zero pixels are foreground.
        max_filter_size (int): Kernel size of the max filter, should be close to the size of an average instance.
        centers_only (bool): Only find the centers and skip the (slow) relabelling of the pixels.
        no_connection_penalty (float): Penalty per background pixel between a pixel and a center.
        voting_kernel (int): Half-size of the majority filter used to remove lone pixels.
        connectivity (int): Pixel connectivity (4 or 8) used to find the instances in the mask.

    Returns:
        labeled_centers (2D int32 array): Centers labelled 1..n.
        center_distances (2D float array): Distance from the edge at the centers, 0 elsewhere.
        relabeled_img (2D int32 array or None): Each pixel labelled with its center's label.
    """
    if max_filter_size < 1:
        raise ValueError(f'max_filter_size must be positive, got {max_filter_size}')
    if connectivity not in (4, 8):
        raise ValueError(f'connectivity must be 4 or 8, got {connectivity}')
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labeled_img, _ = ndimage.label(img_grey, structure=structure)
    dist_transform = ndimage.distance_transform_edt(img_grey)
    labeled_centers, center_distances = find_centers(labeled_img, dist_transform, max_filter_size)
    if centers_only:
        return labeled_centers, center_distances, None

    relabeled_img = find_pixel_class_by_distance(
        labeled_centers, labeled_img, center_distances, no_connection_penalty, voting_kernel)
    return labeled_centers, center_distances, relabeled_img


def separate_images_in_dir(
        input_dir, file_prefix, file_type, output_dir, max_filter_size, centers_only, force_overwrite,
        connectivity=4):
    files = sorted(Path(input_dir).glob(f"{file_prefix}*{file_type}"))
    if len(files) == 0:
        raise FileNotFoundError(f'No images matching {file_prefix}*{file_type} found in {input_dir}!')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file in files:
        out_center = Path(output_dir) / f"centers_{file.name}"
        out_split_instances = Path(output_dir) / f"center_based_relabeled_{file.name}"
        expected_outputs = [out_center] if centers_only else [out_center, out_split_instances]
        if all(f.is_file() for f in expected_outputs) and not force_overwrite:
            logging.info(f'Outputs for {file} already exist. Skipping it.')
            continue

        logging.info(f'Analysing {file}')
        img, meta = utils.read_image(file)
        labeled_centers, _, relabeled_img = separate_objects(
            img, max_filter_size, centers_only, connectivity=connectivity)
        utils.save_image(labeled_centers, out_center, meta)
        logging.info(f'Centers written to {out_center} for {file}')
        if not centers_only:
            utils.save_image(relabeled_img, out_split_instances, meta)
            logging.info(f'Relabelled image written to {out_split_instances} for {file}')


if __name__ == '__main__':
    args = utils.get_args('center_separation')
    logs_file = utils.initialize_log_dir(args.log_dir)
    print(f'Writing logs to {logs_file}')

    separate_images_in_dir(args.input_dir, args.file_prefix, args.file_type,
                           args.output_dir, args.max_filter_size, args.save_only_centers, args.force_overwrite,
                           args.connectivity)
