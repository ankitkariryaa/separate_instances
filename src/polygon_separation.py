from os import cpu_count
import time
import logging
from rasterio.features import Affine
import fiona
import math
import numpy as np
from pathlib import Path
from numpy.core.numeric import ones
from shapely.geometry import polygon
from shapely.geometry.geo import box
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from glob import glob
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio import windows
from shapely.geometry import shape
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src import utils, center_separation


def get_shape_transform_in_original_resolution(pbounds, im_transform):
    wn = windows.from_bounds(*pbounds, im_transform)
    wnt = windows.transform(wn, im_transform)
    wn_shape = windows.shape(wn)
    wns = (round(wn_shape[0]), round(wn_shape[1]))  # Convert to int
    return (wns, wnt)


def polygon_center_separation(polygon, poly_shape, poly_transform, max_filter_size):
    poly_tf = Affine(*poly_transform[:6])
    pimg = rasterio.features.rasterize([polygon], out_shape=poly_shape, all_touched=False, transform=poly_tf)
    _, _, final_image = center_separation.separate_objects(pimg, max_filter_size, centers_only=False)

    final_image = final_image.astype(np.uint8)

    dp = []
    for feature, _ in rasterio.features.shapes(final_image, mask=final_image, connectivity=8, transform=poly_tf):
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
    if len(overlapping_images) == 0:
        raise ValueError(f"No overlapping image found for {fbounds}")
    if len(overlapping_images) > 1:
        logging.error("Multiple images overlap with this area.")
        logging.error(f"Overlapping_images: {overlapping_images[['image_id', 'geometry']]}")
        logging.error(f"This may lead to problems, please double check.\n Proceeding with the first overlap for now.")

    overlapping_images = overlapping_images.iloc[0]
    return overlapping_images['transform'], overlapping_images['img_crs']

# Alternate method for getting the raster transform
# def get_original_raster_transform(corresponding_raster_table, fbounds):
#     bbx = box(*fbounds)
#     a_df = gpd.GeoDataFrame(gpd.GeoSeries(bbx), columns=['geometry'])
#     a_df = a_df.set_crs(corresponding_raster_table.crs)
#     overlapping_images = gpd.overlay(corresponding_raster_table, a_df, how='intersection')
#     if len(overlapping_images) == 0:
#         raise ValueError(f"No overlapping image found for {fbounds}")
#     if len(overlapping_images) > 1:
#         logging.error("Multiple images overlap with this area.")
#         logging.error(f"Overlapping_images: {overlapping_images[['image_id', 'geometry']]}")
#         logging.error(f"This may lead to problems, please double check.\n Proceeding with the first overlap for now.")

#     overlapping_images = overlapping_images.iloc[0]

#     return overlapping_images['transform'], overlapping_images['img_crs']


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
        polygons_file, bbox, rows_slice, corresponding_raster_table, resolution_per_pixel, min_size, max_filter_size,
        only_approximate_area, area_preserving_crs='EPSG:6933', pbar=None):
    """
    Start with a vector file
    Iterate over all polygons (can be parallelized)
    Rasterize individually
    Split the rasterized polygons - either using erosion or center based approach
    Repolygonize
    Write to file
    """
    time1 = time.time()
    polygons = gpd.read_file(polygons_file, bbox=bbox, rows=rows_slice)
    ori_crs = polygons.crs
    time2 = time.time()
    logging.info(f"Read {polygons_file} in {time2 -time1} seconds")

    if len(polygons) == 0:
        logging.error(f"The read files contain no polygons in the given bounds: {bbox}")
        logging.error(f"Returning empty dataframe!")
        fdf = gpd.GeoDataFrame({"geometry": []})
        fdf.set_geometry(col='geometry', inplace=True, crs=ori_crs)
        return fdf, ori_crs

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
        if pbar is not None:
            pbar.update(len(small_polygons))

        large_polygons = large_polygons.to_crs(area_preserving_crs)
        large_polygons['approximate_area'] = large_polygons.area
        large_polygons = large_polygons.to_crs(ori_crs)

        for _, row in large_polygons.iterrows():  # , total=len(large_polygons)):
            new_polys_with_areas = polygon_center_separation_with_area_approximation(
                row['geometry'], row['approximate_area'], row['shape'], row['transform'],  max_filter_size)
            final_polygons.extend(new_polys_with_areas)
            if pbar is not None:
                pbar.update(1)
        fdf = gpd.GeoDataFrame(final_polygons)
        fdf.set_geometry(col='geometry', inplace=True, crs=ori_crs)
    else:
        final_polygons = list(small_polygons['geometry'])  # Add small polygons directly
        if pbar is not None:
            pbar.update(len(small_polygons))

        for _, row in large_polygons.iterrows():  # , total=len(large_polygons)):  # Split large polygons and add them
            new_polys = polygon_center_separation(row['geometry'], row['shape'], row['transform'], max_filter_size)
            final_polygons.extend(new_polys)
            if pbar is not None:
                pbar.update(1)
        fdf = gpd.GeoDataFrame({"geometry": final_polygons})
        fdf.set_geometry(col='geometry', inplace=True, crs=ori_crs)
        fdf = fdf.to_crs(area_preserving_crs)
        fdf['area'] = fdf.area
        fdf = fdf.to_crs(ori_crs)
    return fdf, ori_crs


def divide_bounds(bounds, cpu, total_polygons):
    left, bottom, right, top = bounds
    dbounds = []
    if cpu <= 1:
        return [bounds]
    else:  # Divide along x and y
        # It's better to divide into a large number so that if a threads get a dense area than others can finish the rest in the meantime
        xf = int(math.log(total_polygons) / 4)
        if xf < 4:
            xf = 4
        cpu = cpu * xf
        logging.info(f"Dividing the bounds into {cpu} parts, xf: {xf} ")
        cpu_i = int(math.sqrt(cpu))
        cpu_j = int(cpu / cpu_i)
        dx = (right - left) / (cpu_i)
        dy = (top - bottom) / (cpu_j)
        logging.info(f"File bounds; left, bottom, right, top: {left, bottom, right, top}")
        logging.info(f"Divisions of the file along x axis {cpu_i} and along y axis{cpu_j}")
        logging.info(f"dx along x and y axis: {dx} and {dy}")
        for i in range(cpu_i):
            for j in range(cpu_j):
                lst = (left + (i*dx))
                # Make sure that we don't miss anything for numeric reasons
                lend = (left + ((1+i)*dx)) if i != cpu_i - 1 else right
                bst = (bottom + (j*dy))
                bend = (bottom + ((1+j)*dy)) if j != cpu_j - 1 else top
                assert lst != lend and bst != bend
                dbounds.append((lst, bst, lend, bend))
    return tuple(dbounds)


def divide_count(cpu, total_polygons):
    if cpu <= 1:
        return ((0, total_polygons))
    else:  # Divide along x and y
        cpu = cpu * 2  # Create more splits for efficiency
        logging.info(f"Dividing the total count into {cpu} (+1)  parts, xf: {2} ")

        # Split instances for parallel processing; 0 - background is ignored
        splits = np.arange(0, total_polygons, int(total_polygons/cpu))
        indx_s = [(splits[i-1], splits[i]) for i in range(1, len(splits))]
        if splits[-1] < total_polygons:  # The likely case
            indx_s.append((splits[-1], total_polygons))
    return tuple(indx_s)


def multi_thread_file_processing(file, file_type, output_dir, corresponding_raster_table, resolution_per_pixel,
                                 min_size, max_filter_size, only_approximate_area, cpu):
    print(f'Analysing {file}')
    logging.info(f'Analysing {file}')
    fpf = fiona.open(file)
    total_polygons = len(fpf)

    # The following can be used for dividing the image along bounds;
    # Bounds can lead to duplicates due to "intersection" with bounds and also results in the following:
    ## """RuntimeWarning: Sequential read of iterator was interrupted. Resetting iterator. This can negatively impact the performance."""
    # which is the main reason why reading a large file is slow.
    # total_bounds = fpf.bounds
    # dbounds = divide_bounds(total_bounds, cpu, total_polygons)
    # dbbx = [box(*b) for b in dbounds]

    dindexes = divide_count(cpu, total_polygons)
    dslices = [slice(*di) for di in dindexes]
    df_collection = []
    file_crs = None

    with utils.tqdm_joblib(tqdm(desc="Subset progress", total=len(dslices))) as progress_bar:
        processing_result = Parallel(n_jobs=cpu)(
            delayed(polygon_file_processing)
            (file, None, di, corresponding_raster_table, resolution_per_pixel, min_size, max_filter_size,
                only_approximate_area)
            for di in dslices
        )
        # Single thread equivalent; Useful for debugging
        # for db in dbbx:
        #     processing_result = polygon_file_processing(file, db, None, corresponding_raster_table, resolution_per_pixel,
        #                                             min_size, max_filter_size, only_approximate_area)

    for p in processing_result:
        df_collection.append(p[0])
        if file_crs is None:
            file_crs = p[1]

    processed_df = pd.concat(df_collection, ignore_index=True, sort=False)
    processed_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
    ofn = 'centers_approx_' if only_approximate_area else ''
    cfn = f"cpu_{cpu}_"
    out_file = f"{output_dir}/separated_{ofn}{cfn}{file.split('/')[-1]}"
    print(f'Writing the separated polygons to {out_file}')
    logging.info(f'Writing the separated polygons to {out_file}')
    driver = utils.get_vector_driver(file_type)
    processed_df.to_file(out_file, driver=driver, crs=file_crs, layer="trees")


def single_thread_file_processing(file, file_type, output_dir, corresponding_raster_table, resolution_per_pixel,
                                  min_size, max_filter_size, only_approximate_area):
    fpf = fiona.open(file)
    total_polygons = len(fpf)
    print(f'Analysing {file} with {total_polygons} polygons')
    logging.info(f'Analysing {file} with {total_polygons} polygons')

    with tqdm(total=total_polygons) as pbar:
        processed_df, file_crs = polygon_file_processing(
            file, None, None, corresponding_raster_table, resolution_per_pixel, min_size, max_filter_size,
            only_approximate_area, pbar=pbar)
    processed_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

    ofn = 'centers_approx_' if only_approximate_area else ''
    out_file = f"{output_dir}/separated_{ofn}{file.split('/')[-1]}"
    print(f'Writing the separated polygons to {out_file}')
    logging.info(f'Writing the separated polygons to {out_file}')
    driver = utils.get_vector_driver(file_type)
    processed_df.to_file(out_file, driver=driver, crs=file_crs, layer="trees")


def separate_instances_in_dir(input_dir, file_prefix, file_type,
                              corresponding_raster_dir, corresponding_raster_type, resolution_per_pixel,
                              output_dir, min_size, max_filter_size, only_approximate_area, cpu):

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
        if cpu == 1 or cpu == 0:
            single_thread_file_processing(file, file_type, output_dir, corresponding_raster_table, resolution_per_pixel,
                                          min_size, max_filter_size, only_approximate_area)
        else:
            multi_thread_file_processing(file, file_type, output_dir, corresponding_raster_table, resolution_per_pixel,
                                         min_size, max_filter_size, only_approximate_area, cpu)


# Known issue; logging does not work with joblib parallel
# Possible workaround: https://github.com/joblib/joblib/issues/1017
if __name__ == '__main__':
    args = utils.get_args('polygon_separation')
    logs_file = utils.initialize_log_dir(args.log_dir)
    print(f'Writing logs to {logs_file}')
    cpu_count = utils.get_cpu_count(args.cpu)
    separate_instances_in_dir(args.input_dir, args.file_prefix, args.file_type,
                              args.corresponding_raster_dir, args.corresponding_raster_type, args.resolution_per_pixel,
                              args.output_dir, args.min_size, args.max_filter_size, args.only_approximate_area, cpu_count)


# Read the bounds
# Split the bounds depending upon the number of processors
# Read polygons for each bound in parallel
# Split the polygons
# Write to file
