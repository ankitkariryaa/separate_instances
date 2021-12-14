import os
import psutil
import socket
import random
import argparse
import logging
import rasterio
from scipy import ndimage
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from glob import glob
from shapely.geometry import box

import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_vector_driver(ftype):
    if 'gpkg' in ftype:
        return "GPKG"
    elif 'shp' in ftype:
        return "ESRI Shapefile"
    elif 'json' in ftype:
        return "GeoJSON"
    else:
        raise NotImplementedError("Unknown filetype")


def build_images_table(base_path: str, image_file_type: str):
    cntr = 0
    reference_crs = None
    all_images = []
    files = glob(f"{base_path}/*{image_file_type}")

    unreadable_files = []
    for file in files:
        im_dict = {
            'image_id': cntr,
            'name': file.split(os.sep)[-1],
            'path': file,
            'image_file_type': image_file_type,
        }
        try:
            img = rasterio.open(file)
        except rasterio.errors.RasterioIOError:
            unreadable_files.append(file)
            continue
        nodata = np.array(img.nodatavals)
        for i in range(len(nodata)):
            if nodata[i] is None:
                nodata[i] = np.nan
        nodata = nodata.astype(np.float).reshape((-1, 1, 1))

        im_dict.update({
            'geometry': box(*img.bounds),
            'size': img.shape,
            'profile': img.profile,
            'transform': img.transform,
            'img_crs': img.crs,
            'resolution_per_pixel': img.res,
            'indexes': img.indexes,
            'dtypes': img.dtypes,
            'nodatavals': nodata,
        })
        if reference_crs is None:
            reference_crs = img.crs
        if reference_crs != img.crs:
            raise ValueError('All images must be in the same CRS!')

        img.close()

        cntr += 1
        all_images.append(im_dict)
    if len(unreadable_files) > 0:
        print(f"WARNING: following files couldn't be read:\n {unreadable_files}")
    if not all_images:
        raise ValueError('No images available! Please provide the correct path')
    if reference_crs is None:
        raise ValueError('The reference CRS cant be None!')

    return gpd.GeoDataFrame(all_images, crs=reference_crs)


def str2bool(arg_name):
    def str2bool_(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(
                f'Boolean value expected for argument {arg_name}.')

    return str2bool_


def read_image(img_path, fill_holes=True, dtype=np.uint8):
    im = rasterio.open(img_path)
    meta = im.meta
    img = im.read()
    img_grey = img[0, :, :]  # .mean(axis=(0))) # Same as [0,:,:], for single band image
    if fill_holes:
        img_grey = ndimage.binary_fill_holes(img_grey)
    return img_grey.astype(dtype), meta


def save_image(img, img_path, meta, dtype=rasterio.uint32):
    meta.update(count=1,
                dtype=dtype,
                TILED='YES',
                COMPRESS='LZW',
                BIGTIFF='IF_SAFER',
                multithread=True,
                NUM_THREADS='ALL_CPUS')
    with rasterio.open(img_path, 'w', **meta) as dst:
        dst.write(img.astype(dtype), 1)


def majority_filter_based_upon_original_labels(img_distance_labeled, img_ori_labeled, voting_kernel):
    final_img = img_distance_labeled.copy()
    xmax, ymax = img_distance_labeled.shape

    unqi = np.unique(img_ori_labeled)
    for u in unqi[1:]:  # Iterate over all instances expect the background
        ig = np.where(img_ori_labeled == u, 1, 0).astype(np.uint8)
        labeled_instance = img_distance_labeled * ig  # Get the centers on this instance
        ax = np.nonzero(labeled_instance != 0)
        points = list(zip(ax[0], ax[1]))

        for (i, j) in points:
            cn = labeled_instance[max(i - voting_kernel, 0): min(i + voting_kernel + 1, xmax),
                                  max(j - voting_kernel, 0): min(j + voting_kernel + 1, ymax)]
            unq, unc = np.unique(cn, return_counts=True)
            unqc_sorted_index = np.argsort(-unc)

            mj = unq[unqc_sorted_index][0]
            if mj == 0:  # If background is the majority class, then we take the next class
                mj = unq[unqc_sorted_index][1]
            if mj != final_img[i, j]:  # Replace with majority class if it is a current class is a minority
                final_img[i, j] = mj
    return final_img


def majority_filter(img_ori, voting_kernel):
    img = img_ori.copy()
    xmax, ymax = img.shape
    ax = np.nonzero(img != 0)
    points = list(zip(ax[0], ax[1]))
    for (i, j) in points:
        cn = img_ori[max(i - voting_kernel, 0): min(i + voting_kernel + 1, xmax),
                     max(j - voting_kernel, 0): min(j + voting_kernel + 1, ymax)]
        unq, unc = np.unique(cn, return_counts=True)
        unqc_sorted_index = np.argsort(-unc)

        mj = unq[unqc_sorted_index][0]
        if mj == 0:  # If background is the majority class, then we take the next class
            mj = unq[unqc_sorted_index][1]
        if mj != img[i, j]:  # Replace with majority class if it is a lone pixel of this class
            img[i, j] = mj
    return img


def initialize_log_dir(log_dir):

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    rs = random.Random(datetime.now())
    hash = rs.getrandbits(4)
    os.makedirs(log_dir, exist_ok=True)
    logs_file = os.path.join(log_dir, f'{current_time}_{hash}_{socket.gethostname()}_logs.txt')

    # BasicConfig must be called before any logs are written!
    logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Writing the logs to {logs_file}')
    return logs_file


def get_cpu_count(cpu):
    phy_cpu = psutil.cpu_count()#logical=True)
    if cpu == -1 or phy_cpu < cpu:
        cpu_count = phy_cpu
    else:
        cpu_count = cpu
    return cpu_count


def get_args(task):
    parser = argparse.ArgumentParser(description='Post process the predicted segmentations and separate trees in there',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input-dir', type=str, default='./sample_images',
                        help='where to read the input segmentation masks')
    parser.add_argument('-ft', '--file-type', type=str, default='.tif',
                        help='File type of raster images /vectorfiles to process')
    parser.add_argument('-p', '--file-prefix', type=str, default='',
                        help='Prefix of the raster images / vectorfiles to process')
    parser.add_argument('-o', '--output-dir', type=str, default='./output',
                        help='where to save the output')
    parser.add_argument('-l', '--log-dir', type=str, default='./runs',
                        help='where to save the logs')
    parser.add_argument(
        '--cpu', type=int, default=-1,
        help='How many CPUs to use for the parallelized tasks. -1 means use all. To disable parallel processing, use 1.')
    parser.add_argument('-f', '--force-overwrite', type=str2bool("force_overwrite"),
                        dest='force_overwrite', default=False, help='Whether to overwrite exisiting files.')
    parser.add_argument('-ms', '--min-size', metavar='B', type=int, default=20,
                        help='Minimum size (in pixels) of the instance to be considered for spliting.')
    if task == 'center_separation' or task == 'polygon_separation':
        parser.add_argument(
            '-m', '--max-filter-size', metavar='B', type=int, default=15,
            help='One of the hyperparameters. The kernel size of the max filter operation (in pixels). It should be close to width/height of an average instance.')
        parser.add_argument('-c', '--save-only-centers', type=str2bool("save_only_centers"),
                            dest='save_only_centers', default=False, help='Whether to save the only the centers.')
    if task == 'erosion_separation' or task == 'polygon_separation':
        parser.add_argument('-eis', '--min-eroded-instance-size', metavar='B', type=int, default=4,
                            help='Minimum size of an eroded instance to be considered for spliting.')
        parser.add_argument('-dl', '--clip_distance_list', nargs='+', default=[3, 8, 15],
                            help='Distance in pixel used for separating the images.')
    if task == 'polygon_separation':
        parser.add_argument('--task', type=str, default='center',
                            help='Whether to use center or erosion based approach for separting the polygons.')
        parser.add_argument(
            '-oaa', '--only-approximate-area', type=str2bool("only_approximate_area"),
            dest='only_approximate_area', default=False,
            help='Only calculate the number of centers and approximate the area of each. Only used when task is center.')

        # For vectors we need the corresponding raster to get the transform and resolution

        parser.add_argument(
            '-crd', '--corresponding-raster-dir', type=str, default=None,
            help='where to read the rasters files corresponding to provided vector files. E.g., ./sample_images. Either res or crd should be specified.')
        parser.add_argument('-crt', '--corresponding-raster-type', type=str, default=None,
                            help='File type of corresponding raster images. E.g., .tif')
        parser.add_argument(
            '-res', '--resolution-per-pixel', nargs='+', type=float, default=None,
            help='per pixel resolution of the original raster from where the vectors are extracted. Either res or crd should be specified.')

    return parser.parse_args()
