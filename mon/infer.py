import rasterio
import shapely 
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from shapely import speedups
from tensorflow.keras.models import load_model
from tqdm import tqdm
from utils import prep_chip, get_buffer

if (speedups.available):
    speedups.enable()

def get_spruce(raster, trees, model,
               height_col="Z"):
    """Returns GeoPandas dataframe of bounding boxes of spruce trees

    Given a raster and tree centers corresponding to the raster, 
    make a bounding box (as a function of tree height), and perform 
    inference with given model on extracted image chip. Return only the rows
    of the GeoPandas dataframe such that the classification is Spruce.

    Arguments: 
        raster: rasterio.io.DatasetReader, rasteriod dataset of input raster
        trees: GeoPandas dataframe, of tree centers
        model: tensorflow.python.keras.saving.save_model.load.Sequential
               model to be used for inference
        height_col: str, column name of height of tree 
    """
    model_input_img_size = model.get_input_shape_at(0)[1]
    spatial_res = raster.res[0]

    trees["geometry"] = trees.apply(lambda x: x.geometry.buffer(get_buffer(x[height_col], spatial_res)), axis=1)
    trees["geometry"] = trees.envelope

    tree_chips = []
    
    for index, row in tqdm(trees.iterrows(), desc="Finding Trees...", total=len(trees)): 
        coords = [shapely.geometry.mapping(row["geometry"])]
        out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
        img_arr = prep_chip(out_img, model_input_img_size)
        tree_chips.append(img_arr)

    tree_chips = np.array(tree_chips)

    print("Predicting Species...")
    predictions = model.predict_classes(tree_chips, batch_size=128)
    trees["prediction"] = predictions

    print("Finding Spruces...")
    spruces = trees[trees["prediction"] == 1]

    return spruces