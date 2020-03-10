"""Take a directory full of TIF image chips and make one singular HDF5 file.

The directory is assumed to have the following structure:
root
   |- Non-class
   |   |- img1.tif
   |   |- img2.tif
   |- Class 1
   |   |- img1.tif
   |   |- img2.tif
   |- Class 2... 
"""
import argparse
import h5py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(): 
    counter = 0
    with h5py.File(OUT_FILE, "w") as f:
        for classification in CLASSIFICATIONS:
            for img in tqdm(os.listdir(os.path.join(IN_DIR, classification)), 
                            desc="Writing Class: {}...".format(classification)):

                # Group names are simply increasing numbers
                img_name = counter
                img_path = os.path.join(IN_DIR, classification, img)
                grp = f.create_group("{}".format(img_name))

                # Read TIF file as a numpy array
                img_arr = np.array(Image.open(img_path))

                # Write each channel as a separate dataset
                for i in DATA_COLS: 
                    idx = DATA_COLS.index(i)
                    grp.create_dataset("{}".format(i), data=img_arr[:, :, idx:idx+1], dtype=DT)
                grp.attrs["classification"] = CLASSIFICATIONS.index(classification)
                counter += 1


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DEFAULT_OUT = os.path.join(DATA_DIR, "tif_to_hdf5.h5")
    DEFAULT_DATA_COLS = ["1_R", "2_G", "3_B"]

    parser = argparse.ArgumentParser(
        description="Read TIF chips into singular HDF5 file")

    parser.add_argument("--in_dir", type=str,
                        help="directory of TIF chips")
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, 
                        help="output file of chips [default:/data/tif_to_hdf5.h5")
    parser.add_argument("--classes", nargs="+", default=["not_spruce", "spruce"],
                        help="list of classifications to look for [default:not_spruce, spruce")
    parser.add_argument("--data_cols", nargs="+", default=DEFAULT_DATA_COLS,
                        help="ordered list of channels in TIF file [default:1_R, 2_G, 3_B]")

    FLAGS = parser.parse_args()

    IN_DIR = FLAGS.in_dir
    OUT_FILE = FLAGS.out_file
    CLASSIFICATIONS = FLAGS.classes
    DATA_COLS = FLAGS.data_cols
    DT = np.dtype(float)

    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

    main()