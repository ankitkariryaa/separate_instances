import time
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio import windows
from shapely.geometry import shape

from src import utils, center_separation

def get_shape_transform_in_original_resolution(pbounds, im_transform):
    wn = windows.from_bounds(*pbounds, im_transform)
    wnt = windows.transform(wn, im_transform)
    wn_shape = windows.shape(wn) 
    wns =(int(wn_shape[0]), int(wn_shape[1])) # Convert to int
    return ( wns, wnt)    

def polygon_separation(polygon, im_transform, size_thresh):
    poly_shape, poly_transform = get_shape_transform_in_original_resolution(polygon.bounds, im_transform)
    if poly_shape[0] * poly_shape[1] < size_thresh:
        return [polygon], None, None, None
    else: # Else we separate ;)
        pimg = rasterio.features.rasterize([polygon], out_shape=poly_shape, transform=poly_transform)
        final_image, _ = center_separation.separate_objects(pimg, 15, False)        
        final_image = final_image.astype(np.uint8)
        dp = []
        for feature, _ in rasterio.features.shapes(final_image, mask = final_image, connectivity=4, transform=poly_transform):
            dp.append(shape(feature))
        
        if len(dp) == 1:
            return [polygon]
        else:
            return dp


def polygon_file_processing(polygons_file, ori_raster_tranform, ori_raster_crs, size_thresh):
    """   
    Start with a vector file
    Iterate over all polygons (can be parallelized)
    Rasterize individually
    Split the rasterized polygons - either using erosion or center based approach
    Repolygonize
    Write to file
    """
    final_polygons = []
    time1 = time.time()
    polygons = gpd.read_file(polygons_file)
    logging.info(f"Read {polygons_file} in {(time.time() -time1)*1e6} seconds")
    for index, row in tqdm(polygons.iterrows()):
        new_polys = polygon_separation(row['geometry'], ori_raster_tranform, size_thresh)
        final_polygons.extend(new_polys)

    fdf = gpd.GeoDataFrame({"geometry": final_polygons})
    fdf.set_geometry(col='geometry', inplace=True)
    fdf.to_file('new_polygons.gpkg', driver="GPKG", crs=ori_raster_crs, layer="trees")

def separate_instances_in_dir(input_dir, image_file_prefix, image_file_type, output_dir, size_thresh):
    # Get all input image paths
    ori_im = 'sample_images_large/image1.tif'
    ori_raster = rasterio.open(ori_im)
    ori_raster_transform = ori_raster.transform
    ori_raster_crs = ori_raster.crs
    files = glob( f"{input_dir}/{image_file_prefix}*{image_file_type}" )
    if len(files) == 0:
        raise Exception('No images found in the specified folder!')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file in files:
        print(f'Analysing {file}')
        
        logging.info(f'Analysing {file}')
        polygon_file_processing(file, ori_raster_transform, ori_raster_crs, size_thresh)


if __name__ == '__main__':
    args = utils.get_args('erosion_based_separation')
    logs_file = utils.initialize_log_dir(args.log_dir)
    print(f'Writing logs to {logs_file}')

    separate_instances_in_dir(args.input_dir, args.image_file_prefix, args.image_file_type, args.output_dir, 
            args.size_thresh)
