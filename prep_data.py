"""Script to augment data.

Take an HDF5 file full of image chips and prepare them for training on 
neural network. Images rotated 90, 180, and 270 degrees, flipped vertically 
and horizontally for each orientation, increasing sample size seven-fold.
Transformed image chips, as well as their originals, are saved in 
specified output file.

Image suffixes are as follows:
    _0 = original image
    _1 = image flipped horizontally
    _2 = rotated 90 degrees counterclockwise
    _3 = rotated 90 degrees counterclockwise flipped vertically
    _4 = rotated 180 degrees
    _5 = rotated 180 degrees flipped horizontally
    _6 = rotated 90 degrees clockwise
    _7 = rotated 90 degrees clockwise flipped vertically
"""
import argparse
import cv2
import h5py
import os
import numpy as np
from tqdm import tqdm
from mon.utils import prep_chip, make_square


def augment(img):
    """Returns a list of image arrays after augmentation
    
    Perform rotation and reflection
    
    Arguments: 
        img: ndarray
    """
    # Returns a list, to iterate later
    imgs = []
    imgs.append(img)
    imgs.append(np.fliplr(img))

    imgs.append(np.rot90(img))
    imgs.append(np.flipud(np.rot90(img)))
    
    imgs.append(np.rot90(img, 2))
    imgs.append(np.fliplr(np.rot90(img, 2)))

    imgs.append(np.rot90(img, 3))
    imgs.append(np.flipud(np.rot90(img, 3)))

    return imgs


def main():
    """Writes to HDF5 file"""
    with h5py.File(IN_FILE, "r") as f_in:  
        with h5py.File(OUT_FILE, "w") as f_out:
            for img in tqdm(f_in, desc="Writing to {}".format(OUT_FILE.split("\\")[-1])):
                img_name = img.split(".")[0]
                datagroup = f_in[img]

                # Read each channel and create a numpy ndarray
                iteration = iter(datagroup)
                first_channel = next(iteration)
                img_arr = np.array(datagroup[first_channel])
                for c in iteration:
                    img_arr = np.c_[img_arr, np.array(datagroup[c])]

                # Get classification
                img_class = datagroup.attrs['classification']

                # Add padding if image is not a square
                if (img_arr.shape[0] != img_arr.shape[1]):
                    img_arr = make_square(img_arr)
                else:
                    img_arr = img_arr

                # Augment 
                imgs = augment(img_arr)

                # Write each augmented image to HDF5 as separate datasets
                for i in range(8):
                    grp = f_out.create_group("{}_{}".format(img_name, i))
                    grp.attrs['classification'] = img_class
                    for j in DATA_COLS: 
                        idx = DATA_COLS.index(j)
                        grp.create_dataset("{}".format(j), 
                                        data=imgs[i][:, :, idx:idx+1], 
                                        dtype=DT)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DEFAULT_OUT = os.path.join(DATA_DIR, 'training_data.h5')
    DEFAULT_DATA_COLS = ["1_R", "2_G", "3_B"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str,
                        help="H5 file to augment")
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, 
                        help="output file of augmented chips [default:/data/training_data.h5")
    parser.add_argument("--data_cols", nargs="+", default=DEFAULT_DATA_COLS,
                        help="ordered list of channels in TIF file [default:1_R, 2_G, 3_B]")

    FLAGS = parser.parse_args()

    IN_FILE = FLAGS.in_file
    OUT_FILE = FLAGS.out_files
    DATA_COLS = FLAGS.data_cols

    DT = np.dtype(float)

    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    
    main()