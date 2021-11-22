import argparse
import rasterio
from scipy import ndimage
import numpy as np

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

def get_args():
    parser = argparse.ArgumentParser(description='Post process the predicted segmentations and separate trees in there',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i','--input-dir', type=str, default='./sample_images',
                        help='where to read the input segmentation masks')
    parser.add_argument('-ft', '--image-file-type', type=str, default='.tif',
                        help='File type of images to process')
    parser.add_argument('-p', '--image-file-prefix', type=str, default='',
                        help='Prefix of the image to process')
    parser.add_argument('-o','--output-dir', type=str, default='./output',
                        help='where to save the output')
    parser.add_argument('-l','--log-dir', type=str, default='./runs',
                        help='where to save the logs')
    parser.add_argument('--cpu', type=int, default=-1,
                        help='How many CPUs to use for the parallelized tasks. -1 means use all. To disable parallel processing, use 1.')
    parser.add_argument('-m', '--max-filter-size', metavar='B', type=int, default=12,
                        help='One of the hyperparameters. The kernel size of the max filter operation (in pixels). It should be close to width/height of an average instance.')
    parser.add_argument('-c', '--save-only-centers', type=str2bool("save_only_centers"), dest='save_only_centers', default=False,
                            help='Whether to save the only the centers.')
    parser.add_argument('-f', '--force-overwrite', type=str2bool("force_overwrite"), dest='force_overwrite', default=False,
                            help='Whether to overwrite exisiting files.')
    return parser.parse_args()