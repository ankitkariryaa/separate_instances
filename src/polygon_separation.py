import time
import logging
from rasterio.features import Affine
import fiona
import numpy as np
from pathlib import Path
from numpy.core.numeric import ones
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from glob import glob
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio import windows
from shapely.geometry import shape
import matplotlib.pyplot as plt

from src import utils, center_separation


def get_shape_transform_in_original_resolution(pbounds, im_transform):
    wn = windows.from_bounds(*pbounds, im_transform)
    wnt = windows.transform(wn, im_transform)
    wn_shape = windows.shape(wn)
    wns = (int(wn_shape[0]), int(wn_shape[1]))  # Convert to int
    return (wns, wnt)


def polygon_center_separation(polygon, poly_shape, poly_transform, max_filter_size):
    # poly_shape, poly_transform = get_shape_transform_in_original_resolution(polygon.bounds, im_transform)
    poly_tf = Affine(*poly_transform[:6])
    pimg = rasterio.features.rasterize([polygon], out_shape=poly_shape, transform=poly_tf)
    _, _, final_image = center_separation.separate_objects(pimg, max_filter_size, centers_only=False)
    # plt.imshow(final_image)
    # plt.show()
    final_image = final_image.astype(np.uint8)

    dp = []
    for feature, _ in rasterio.features.shapes(final_image, mask=final_image, connectivity=4, transform=poly_tf):
        dp.append(shape(feature))

    if len(dp) == 1:
        return [polygon]
    else:
        return dp


def polygon_center_separation_with_area_approximation(
        polygon, polygon_area, poly_shape, poly_transform, max_filter_size):
    poly_tf = Affine(*poly_transform[:6])

    pimg = rasterio.features.rasterize([polygon], out_shape=poly_shape, transform=poly_tf)
    _, final_image, _ = center_separation.separate_objects(pimg, max_filter_size, centers_only=True)
    # Get the centers and the corresponding
    if np.count_nonzero(final_image) == 1:
        return [{
            'geometry': polygon.centroid,
            'approximate_area': polygon_area,
            'approximate_area_proportion': 1,
        }]

    dp = []
    rows, cols = np.nonzero(final_image)
    x, y = rasterio.transform.xy(poly_tf, rows, cols)  # rowcol to xy
    tr = np.sum(np.square(final_image))
    for i in range(len(rows)):  # assuming they are in image coordinates
        pr = np.square((final_image[rows[i], cols[i]])) / tr
        dc = {
            'geometry': Point(x[i], y[i]),
            'approximate_area': polygon_area * pr,
            'approximate_area_proportion': pr
        }
        dp.append(dc)

    return dp


def get_original_raster_transform(corresponding_raster_table, fbounds):
    xmin, ymin, xmax, ymax = fbounds
    overlapping_images = corresponding_raster_table.cx[xmin:xmax, ymin:ymax]
    # print(overlapping_images)
    # import ipdb; ipdb.set_trace()
    assert len(overlapping_images) == 1
    overlapping_images = overlapping_images.iloc[0]

    return overlapping_images['transform'], overlapping_images['img_crs']


def get_transform_from_resolution(resolution_per_pixel, fbounds):
    xmin, ymin, _, _ = fbounds

    af = np.zeros((2, 3))  # Double check
    af[0, 1] = resolution_per_pixel[0]
    af[1, 0] = resolution_per_pixel[1]
    af[0, 2] = ymin
    af[1, 2] = xmin
    af = Affine(*af.flatten())
    return af


def polygon_file_processing(
        polygons_file, corresponding_raster_table, resolution_per_pixel, min_size, max_filter_size,
        only_approximate_area, area_preserving_crs='EPSG:6933'):
    """
    Start with a vector file
    Iterate over all polygons (can be parallelized)
    Rasterize individually
    Split the rasterized polygons - either using erosion or center based approach
    Repolygonize
    Write to file
    """
    time1 = time.time()
    polygons = gpd.read_file(polygons_file)
    ori_crs = polygons.crs
    time2 = time.time()
    logging.info(f"Read {polygons_file} in {time2 -time1} seconds")

    if corresponding_raster_table is not None:
        ori_raster_tranform, ori_raster_crs = get_original_raster_transform(
            corresponding_raster_table, polygons.total_bounds)
    else:
        ori_raster_crs = polygons.crs
        ori_raster_tranform = get_transform_from_resolution(resolution_per_pixel)

    polygons[['minx', 'miny', 'maxx', 'maxy']] = polygons.bounds
    polygons['shape_and_transform'] = polygons.apply(lambda x: get_shape_transform_in_original_resolution(
        [x['minx'], x['miny'], x['maxx'], x['maxy']], ori_raster_tranform), axis=1)
    polygons['shape'], polygons['transform'] = zip(*polygons['shape_and_transform'])
    polygons['width'], polygons['height'] = zip(*polygons['shape'])

    small_polygons = polygons[(polygons["width"] * polygons["height"] <= min_size)]
    large_polygons = polygons[(polygons["width"] * polygons["height"] > min_size)]

    logging.info(f"""Filtered small polygons in {(time.time() -time2)} seconds.
    \nNow processing {len(large_polygons)} polygons larger than the size threshold.""")

    if only_approximate_area:
        small_polygons = small_polygons.to_crs(area_preserving_crs)
        # It's the actual area but for consistency we call it "approximation"
        small_polygons['approximate_area'] = small_polygons.area
        small_polygons = small_polygons.to_crs(ori_crs)
        small_polygons['geometry'] = small_polygons.centroid
        small_polygons['approximate_area_proportion'] = 1
        final_polygons = small_polygons[['geometry', 'approximate_area',
                                         'approximate_area_proportion']].to_dict('records')

        large_polygons = large_polygons.to_crs(area_preserving_crs)
        large_polygons['approximate_area'] = large_polygons.area
        large_polygons = large_polygons.to_crs(ori_crs)

        for _, row in tqdm(large_polygons.iterrows(), total=len(large_polygons)):
            new_polys_with_areas = polygon_center_separation_with_area_approximation(
                row['geometry'], row['approximate_area'], row['shape'], row['transform'],  max_filter_size)
            final_polygons.extend(new_polys_with_areas)
        fdf = gpd.GeoDataFrame(final_polygons)
        fdf.set_geometry(col='geometry', inplace=True, crs=ori_crs)
    else:
        final_polygons = list(small_polygons['geometry'])  # Add small polygons directly
        for _, row in tqdm(large_polygons.iterrows(), total=len(large_polygons)):  # Split large polygons and add them
            new_polys = polygon_center_separation(row['geometry'], row['shape'], row['transform'], max_filter_size)
            final_polygons.extend(new_polys)
        fdf = gpd.GeoDataFrame({"geometry": final_polygons})
        fdf.set_geometry(col='geometry', inplace=True, crs=ori_crs)
        fdf = fdf.to_crs(area_preserving_crs)
        fdf['area'] = fdf.area
        fdf = fdf.to_crs(ori_crs)
    return fdf, ori_crs

# def divide_bounds(bounds, processor_count):
#     left, bottom, right, top = bounds
#     if processor_count <= 1:
#         return [bounds]
#     elif processor_count <= 3:
#         # Divide along the longer edge
#         if np.abs(right - left) >= np.abs(top - bottom):
#             dbounds = []
#             for i in range(processor_count):
#                 ci = (right - left) / processor_count
#                 dbounds.append( [ (left + (i*ci)), bottom, (left + ((1+i)*ci)),top])

#     else:
#         # Divide along x and y
#     return dbounds
# def parallel_polygon_separation(polygons_file, ori_raster_tranform, ori_raster_crs, size_thresh):
#     fpf = fiona.open(polygons_file)
#     total_bounds = fpf.bounds


def separate_instances_in_dir(input_dir, file_prefix, file_type,
                              corresponding_raster_dir, corresponding_raster_type, resolution_per_pixel,
                              output_dir, min_size, max_filter_size, only_approximate_area):

    files = glob(f"{input_dir}/{file_prefix}*{file_type}")
    if len(files) == 0:
        raise Exception('No files found in the specified folder!')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assert corresponding_raster_dir is not None or resolution_per_pixel is not None

    if corresponding_raster_dir is not None:
        corresponding_raster_table = utils.build_images_table(
            base_path=corresponding_raster_dir, image_file_type=corresponding_raster_type)
    else:
        corresponding_raster_table = None

    for file in files:
        print(f'Analysing {file}')
        logging.info(f'Analysing {file}')
        processed_df, orcs = polygon_file_processing(file, corresponding_raster_table, resolution_per_pixel,
                                                     min_size, max_filter_size, only_approximate_area)
        out_file = f"{output_dir}/separated_{'centers_approx_' if only_approximate_area else ''}{file.split('/')[-1]}"
        driver = utils.get_vector_driver(file_type)
        processed_df.to_file(out_file, driver=driver, crs=orcs, layer="trees")


if __name__ == '__main__':
    args = utils.get_args('polygon_separation')
    logs_file = utils.initialize_log_dir(args.log_dir)
    print(f'Writing logs to {logs_file}')
    separate_instances_in_dir(args.input_dir, args.file_prefix, args.file_type,
                              args.corresponding_raster_dir, args.corresponding_raster_type, args.resolution_per_pixel,
                              args.output_dir, args.min_size, args.max_filter_size, args.only_approximate_area)


# Read the bounds
# Split the bounds depending upon the number of processors
# Read polygons for each bound in parallel
# Split the polygons
# Write to file
