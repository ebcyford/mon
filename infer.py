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
from mon.utils import get_buffer, prep_chip
from rasterio.mask import mask
from shapely import speedups
from tqdm import tqdm

if (speedups.available):
    speedups.enable()


def main():
    trees['geometry'] = trees.apply(lambda x: x.geometry.buffer(get_buffer(x.Z, SPATIAL_RES)), axis=1)
    trees['geometry'] = trees.envelope
    trees['prediction'] = 0

    tree_chips = []
    for index, row in tqdm(trees.iterrows(), desc="Finding Trees...", total=len(trees)): 
        coords = [shapely.geometry.mapping(row['geometry'])]
        out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
        img_arr = prep_chip(out_img, IMG_SIZE)
        tree_chips.append(img_arr)

    tree_chips = np.array(tree_chips)
    tree_chips = tf.keras.utils.normalize(tree_chips)

    print("Predicting species...")
    predictions = model.predict_classes(tree_chips, batch_size=128)

    trees['prediction'] = predictions

    print("Writing to " + OUT_FILE)
    trees.to_file(OUT_FILE)


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
    parser.add_argument("--spatial_resolution", type=float, default=0.1,
                        help="spatial resolution in units of input raster [default:0.1]")
    parser.add_argument("--img_size", type=int, default=80, 
                        help="size of image as fed into model when training [default:80]")
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, 
                        help="shapefile of output trees and predictions [default:output.shp]")

    FLAGS = parser.parse_args()

    MODEL = FLAGS.model
    TREES = FLAGS.tree_centers
    RASTER = FLAGS.in_raster
    OUT_FILE = FLAGS.out_file
    SPATIAL_RES = FLAGS.spatial_resolution
    IMG_SIZE = FLAGS.img_size

    print("Reading Data...")
    raster = rasterio.open(RASTER)
    trees = gpd.read_file(TREES)
    model = tf.keras.models.load_model(MODEL)
    main()