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

MODEL = r"D:\capstone\neon\output\LidR\byCenters\models\model27.model"
TREES = r"D:\capstone\neon\output\LidR\GRSM_275000_3941000_treeCenters8.shp"
RASTER = r"D:\capstone\neon\data\ortho\mosaic\2018_GRSM_4_275000_3941000_image.tif"
OUT_FILE = r"D:\capstone\neon\output\LidR\byCenters\GRSM_275000_3941000_treesPredicted.shp"
SPATIAL_RES = 0.1
IMG_SIZE = 80

raster = rasterio.open(RASTER)
trees = gpd.read_file(TREES)
model = tf.keras.models.load_model(MODEL)

def prep_chip(img_arr):
    img_arr = img_arr.transpose((1,2,0))
    img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img_arr

def get_buffer(tree_height, spatial_res=SPATIAL_RES):
    raw_buff = 0.025 * tree_height + 2
    return round(spatial_res * round(raw_buff/spatial_res),1)

trees['geometry'] = trees.apply(lambda x: x.geometry.buffer(get_buffer(x.Z)), axis=1)
trees['geometry'] = trees.envelope
trees['prediction'] = 0

tree_chips = []
for index, row in tqdm(trees.iterrows()): 
    coords = [shapely.geometry.mapping(row['geometry'])]
    out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
    img_arr = prep_chip(out_img)
    tree_chips.append(img_arr)

tree_chips = np.array(tree_chips)
tree_chips = tf.keras.utils.normalize(tree_chips)

predictions = model.predict_classes(tree_chips, batch_size=128)

trees['prediction'] = predictions

print("ran successfully")