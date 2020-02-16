"""
Emily Cyford
Shepherd University 
2/13/2020

Documentation and code for taking an HDF5 file full of image chips and 
preparing them for training on neural network.
Images are resized to a specified square numpy array. 
Images are rotated 90, 180, and 270 degrees, flipped vertically and 
horizontally for each orientation, increasing sample size seven-fold.
Transformed image chips, as well as their resized originals, are saved in 
specified output file.

Image suffixes are as follows:
    _0 = original image resized
    _1 = resized image flipped horizontally
    _2 = resized rotated 90 degrees counterclockwise
    _3 = resized rotated 90 degrees counterclockwise flipped vertically
    _4 = resized rotated 180 degrees
    _5 = resized rotated 180 degrees flipped horizontally
    _6 = resized rotated 90 degrees clockwise
    _7 = resixed rotated 90 degrees clockwise flipped vertically
      
"""
import argparse
import cv2
import h5py
import os
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEFAULT_OUT = os.path.join(DATA_DIR, 'training_data.h5')

parser = argparse.ArgumentParser(
    description="Augmentation through rotation and reflection")

parser.add_argument("--in_file", type=str,
                    help="H5 file to augment")
parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, 
                    help="output file of augmented chips [default:/data/training_data.h5")
parser.add_argument("--img_size", type=int, default=64, 
                    help="height of square image chip in pixels [default:64]")

FLAGS = parser.parse_args()

IN_FILE = FLAGS.in_file
OUT_FILE = FLAGS.out_file
IMG_SIZE = FLAGS.img_size

DATA_COLS = ["1_R", "2_G", "3_B","4_NIR"]
DT = np.dtype('uint8')

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

with h5py.File(IN_FILE, "r") as f_in:  
    with h5py.File(OUT_FILE, "w") as f_out:
        for img in tqdm(f_in.keys()):
            img_name = img.split(".")[0]

            # Read each channel 
            red = f_in[img]['1_R']
            green = f_in[img]['2_G']
            blue = f_in[img]['3_B']
            nir = f_in[img]['4_NIR']

            # Create numpy ndarray 
            img_arr = np.c_[red, green, blue, nir]

            # Resize 
            img_resized = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE), \
                                                interpolation=cv2.INTER_CUBIC)
            fliplr = np.fliplr(img_resized)

            rot90 = np.rot90(img_resized)
            rot90flipud = np.flipud(np.rot90(img_resized))
            
            rot180 = np.rot90(img_resized, 2)
            rot180fliplr =np.fliplr(np.rot90(img_resized, 2))

            rot270 = np.rot90(img_resized, 3)
            rot270flipud = np.flipud(np.rot90(img_resized, 3))

            imgs = [
                img_resized, fliplr, 
                rot90, rot90flipud,
                rot180, rot180fliplr,
                rot270, rot270flipud
                ]

            for i in range(8):
                grp = f_out.create_group("{}_{}".format(img_name, i))
                for i in DATA_COLS: 
                    idx = DATA_COLS.index(i)
                    grp.create_dataset("{}".format(i), 
                                    data=imgs[idx][:, :, idx:idx+1], 
                                    dtype=DT)
