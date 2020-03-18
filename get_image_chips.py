"""Extract image chips from raster for training on neural network

Given a raster and locations of tree centers, extract image chips as a 
function of the tree height and save into a directory.
"""

import argparse
import os
import rasterio
import shapely
import geopandas as gpd 
from mon.utils import get_buffer
from rasterio.mask import mask
from shapely import speedups
from tqdm import tqdm

if (speedups.available):
    speedups.enable()

def main():
    # Apply a buffer as a function of tree height
    trees['geometry'] = trees.apply(lambda x: x.geometry.buffer(get_buffer(x.Z, SPATIAL_RES)), axis=1)
    # Make the buffer square
    trees['geometry'] = trees.envelope

    for index, row in tqdm(trees.iterrows(), total=len(trees), desc="Extracting Chips..."):
        # Extract the coordinates of the masking geometry
        coords = [shapely.geometry.mapping(row['geometry'])]
        out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
        # Copy metadata from parent raster
        out_meta = raster.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": out_img.shape[1],
                        "width": out_img.shape[2],
                        "transform": out_transform,
                        "crs": raster.crs.data}
                    )
        # Write to destination
        out_file = os.path.join(OUT_DIR, "{}.tif".format(row['treeID']))
        with rasterio.open(out_file, "w", **out_meta) as dest: 
            dest.write(out_img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_raster", type=str, 
                        help="filepath of raster to perform inference")
    parser.add_argument("--tree_centers", type=str, 
                        help="shapefile of identified tree centers")
    parser.add_argument("--out_dir", type=str, 
                        help="directory to place image chips")

    FLAGS = parser.parse_args()

    TREES = FLAGS.tree_centers
    RASTER = FLAGS.in_raster
    OUT_DIR = FLAGS.out_dir

    print("Reading Data...")
    raster = rasterio.open(RASTER)
    trees = gpd.read_file(TREES)

    SPATIAL_RES = raster.res[0]

    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    
    main()
