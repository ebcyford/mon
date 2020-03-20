"""Retrieve model and infer multiple rasters

This script loads the best performing model, along with a raster and positions
of tree centers, and performs inference on self-generated image chips.
These image chips are a function of the identified tree's height.
"""
import argparse
import cv2
import os
import rasterio
import shapely
import geopandas as gpd
import numpy as np
import tensorflow as tf
from mon.infer import get_trees
from rasterio.mask import mask
from shapely import speedups
from tqdm import tqdm

# Environment settings
if (speedups.available):
    speedups.enable()

tf.get_logger().setLevel("INFO")


def get_tiles(raster_dir, tree_dir):
    """Returns a list of tile names.

    Given a directory of rasters and a directory of tree centers, find
    tile names in common between the two. This ensures that the correct
    tree centers are aligned with the correct raster. 

    Arguments: 
        raster_dir: str, location of directory of rasters
        tree_dir: str, location of directory of tree centers
    """
    tifs = []
    shps = []

    for file in os.listdir(raster_dir):
        if (file.endswith(".tif")):
            tile = file.rsplit("_", 1)[0]
            tifs.append(tile)
        else:
            continue
    
    for file in os.listdir(tree_dir):
        if (file.endswith(".shp")):
            tile = file.rsplit("_",1 )[0]
            shps.append(tile)
        else:
            continue
    
    tif_set = set(tifs)
    shps_set = set(shps)
    tiles = tif_set.intersection(shps_set)

    print("{} of {} rasters and {} shapefiles found".format(len(tiles), len(tifs), len(shps)))
    return tiles

def main():
    tiles = get_tiles(IMG_DIR, TREE_DIR)

    for tile in tiles:
        print("Processing tile {}...".format(tile))
        raster_file = os.path.join(IMG_DIR, tile + "_ortho.tif")
        tree_file = os.path.join(TREE_DIR, tile + "_treeCenters.shp")
        out_file = os.path.join(OUT_DIR, tile + "{}_treesClassified.shp".format(MODEL))

        raster = rasterio.open(raster_file)
        trees = gpd.read_file(tree_file)

        SPATIAL_RES = raster.res[0]

        trees_classified = get_trees(raster, trees, model)

        trees_classified.to_file(out_file)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, 
                        help="location of CNN model")
    parser.add_argument("--img_dir", type=str, 
                        help="directory of rasters to perform inference")
    parser.add_argument("--tree_dir", type=str, 
                        help="directory of identified tree centers shapefiles")
    parser.add_argument("--out_dir", type=str,
                        help="directory of output trees and predictions")

    FLAGS = parser.parse_args()

    MODEL = FLAGS.model
    IMG_DIR = FLAGS.img_dir
    TREE_DIR = FLAGS.tree_dir
    OUT_DIR = FLAGS.out_dir 

    model = tf.keras.models.load_model(MODEL)
    
    IMG_SIZE = model.get_input_shape_at(0)[1]

    main()