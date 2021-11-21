import argparse

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
    parser.add_argument('-m', '--max-filter-size', metavar='B', type=int, default=12,
                        help='One of the hyperparameters. The kernel size of the max filter operation (in pixels). It should be close to width/height of an average instance.')
    parser.add_argument('-c', '--save-only-centers', type=str2bool("save_only_centers"), dest='save_only_centers', default=False,
                            help='Whether to save the only the centers.')
    parser.add_argument('-f', '--force-overwrite', type=str2bool("force_overwrite"), dest='force_overwrite', default=False,
                            help='Whether to overwrite exisiting files.')                            
    return parser.parse_args()