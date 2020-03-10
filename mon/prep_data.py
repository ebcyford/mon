"""Script to augment data.

Take an HDF5 file full of image chips and prepare them for training on 
neural network. Images are resized to a specified square numpy array. 
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


def make_square(arr):
    """Returns a square image array
    
    If height is greater than width, find the difference between the two. 
    If the difference is even, add to both sides. 
    If the difference is odd, add extra pixel column to the left.

    If the width is greater than the height, find the difference.
    If the difference is even, add to top and bottom.
    If the differnce is odd, add extra pixel row to the top.

    Arguments:
        arr: ndarray, image to be padded
    """
    if (arr.shape[0] > arr.shape[1]):
        pad = arr.shape[0] - arr.shape[1]
        if (pad % 2 == 0):
            pad_left = int(pad / 2)
            pad_right = int(pad / 2)
        else: 
            pad_left = int((pad + 1) / 2)
            pad_right = pad_left - 1
        img_arr = np.pad(arr, ((0,0), (pad_left, pad_right), (0,0)), mode='constant')
    elif (arr.shape[0] < arr.shape[1]):
        pad = arr.shape[1] - arr.shape[0]
        if (pad % 2 == 0):
            pad_up = int(pad / 2)
            pad_down = int(pad / 2)
        else: 
            pad_up = int((pad + 1) / 2)
            pad_down = pad_up - 1
        img_arr = np.pad(arr, ((pad_up, pad_down), (0,0), (0,0)), mode='constant')
    # Should not get this far, but if it does, just return the array
    else:
        img_arr = arr
    return img_arr


def augment(img):
    """Returns a list of image arrays after augmentation
    
    Perform rotation and reflection
    
    Arguments: 
        img: ndarray
    """
    # Returns a list, to iterate later
    imgs = []
    imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE), \
                                        interpolation=cv2.INTER_CUBIC))
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str,
                        help="H5 file to augment")
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, 
                        help="output file of augmented chips [default:/data/training_data.h5")
    parser.add_argument("--img_size", type=int, default=80, 
                        help="height of square image chip in pixels [default:80]")

    FLAGS = parser.parse_args()

    IN_FILE = FLAGS.in_file
    OUT_FILE = FLAGS.out_file
    IMG_SIZE = FLAGS.img_size

    DATA_COLS = ["1_R", "2_G", "3_B"]
    DT = np.dtype('uint8')

    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    
    main()