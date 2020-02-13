"""
Emily Cyford
Shepherd University 
2/10/2020

Documentation and code for taking a directory full of image chips and 
preparing them for training on neural network.
Images are resized to a specified square numpy array. 
Images are rotated 90, 180, and 270 degrees, flipped vertically and 
horizontally for each orientation, increasing sample size seven-fold.
Transformed image chips, as well as their resized originals, are saved in 
specified output directory.

Image suffixes are as follows:
    _0 = original image resized
    _1 = resized image flipped horizontally
    _2 = resized rotated 90 degrees counterclockwise
    _3 = resized rotated 90 degrees counterclockwise flipped vertically
    _4 = resized rotated 180 degrees
    _5 = resized rotated 180 degrees flipped horizontally
    _6 = resized rotated 90 degrees clockwise
    _7 = resixed rotated 90 degrees clockwise flipped vertically

TODO: Determine format of image chips
      Must eventually be a numpy array
      
"""
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE_DIR, 'prepped_chips')

parser = argparse.ArgumentParser(
    description="Augmentation through rotation and reflection")

parser.add_argument("--in_dir", type=str, default=BASE_DIR, 
                    help="directory of images to augment [default:base dir]")
parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT, 
                    help="output directory of augmented chips")
parser.add_argument("--img_size", type=int, default=64, 
                    help="height of square image chip in pixels [default:64]")

FLAGS = parser.parse_args()

IN_DIR = FLAGS.in_dir
OUT_DIR = FLAGS.out_dir
IMG_SIZE = FLAGS.img_size

if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)

for img in tqdm(os.listdir(IN_DIR)):
    img_name = img.split(".")[0]    
    in_path = os.path.join(IN_DIR, img)
    out_path = os.path.join(OUT_DIR, img_name)

    img_array = cv2.imread(in_path)

    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
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
        cv2.imwrite(str(out_path + "_{}.jpg").format(i), imgs[i])