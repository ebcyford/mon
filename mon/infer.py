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
from rasterio.mask import mask
from shapely import speedups
from tqdm import tqdm

if (speedups.available):
    speedups.enable()


def prep_chip(img_arr, img_size=80):
    """Returns a numpy array

    If the input img is channels first, transposes it to channels last. 
    Reshapes the image chip to the expected size of the input tensor.

    Arguments:
        img_arr: ndarray 
        img_size: image size expected by model
    """
    if (img_arr.shape[0] < img_arr.shape[-1]):
        img_arr = img_arr.transpose((1,2,0))
    img_arr = cv2.resize(img_arr, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    return img_arr


def get_buffer(tree_height, spatial_res=0.1):
    """Returns a float

    Given height of the tree, create a buffer distance as a function of the height.

    Arguments:
        tree_height: float
        spatial_res: float, ensure height and resolution are in the same units
    """
    raw_buff = 0.025 * tree_height + 2
    return round(spatial_res * round(raw_buff/spatial_res),1)


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