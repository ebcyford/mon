"""Retrieve model and infer on raster data

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


def main():
    trees_classified = get_trees(raster, trees, model)

    print("Writing to " + OUT_FILE)
    trees_classified.to_file(OUT_FILE)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_OUT = os.path.join(BASE_DIR, "output.shp")

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, 
                        help="location of CNN model")
    parser.add_argument("--in_raster", type=str, 
                        help="filepath of raster to perform inference")
    parser.add_argument("--tree_centers", type=str, 
                        help="shapefile of identified tree centers")
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, 
                        help="shapefile of output trees and predictions [default:output.shp]")

    FLAGS = parser.parse_args()

    MODEL = FLAGS.model
    TREES = FLAGS.tree_centers
    RASTER = FLAGS.in_raster
    OUT_FILE = FLAGS.out_file

    print("Reading Data...")
    raster = rasterio.open(RASTER)
    trees = gpd.read_file(TREES)
    model = tf.keras.models.load_model(MODEL)

    SPATIAL_RES = raster.res[0]
    IMG_SIZE = model.get_input_shape_at(0)[1]
    main()