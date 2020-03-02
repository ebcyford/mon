"""
Emily Cyford
Shepherd University
3/2/2020

Documentation and code for building convolutional neural network to identify
spruce trees. 

Additional arguments may be passed in the command line, 
but optimal parameters will be coded in as defaults.
"""
import argparse
import h5py
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(BASE_DIR, r"data\training_data.h5")
DEFAULT_LOG = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

parser = argparse.ArgumentParser()

parser.add_argument("--in_file", type=str, default=DEFAULT_IN,
                    help="HDF5 file of training data chips [default:data\training_data.h5")
parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG,
                    help="directory of log files [default:logs")
parser.add_argument("--batch_size", type=int, default=128,
                    help="number images to process in each batch [default:128")
parser.add_argument("--epochs", type=int, default=15, 
                    help="number of epochs to train model [default:15]")
parser.add_argument("--validation_size", type=int, default=0.3,
                    help="size of validations set [default:0.3")
parser.add_argument("--dense_layers", type=int, nargs="+", default=[0,1,2],
                    help="numerical list of number of layers to try model [default:0,1,2]")
parser.add_argument("--layer_sizes", type=int, nargs="+", default=[32,64,128],
                    help="numerical list of layer sized to try model [default:32,64,128]")
parser.add_argument("--conv_layers", type=int, nargs="+", default=[1,2,3],
                    help="numerical list of number of conv layers to try model [default:1,2,3]")
parser.add_argument("--save", action="store_true", default=False,
                    help="save model or not [default:False]")

FLAGS = parser.parse_args()

IN_FILE = FLAGS.in_file
LOG_DIR = FLAGS.log_dir
BATCH_SIZE = FLAGS.batch_size
EPOCHS = FLAGS.epochs
VALIDATION_SIZE = FLAGS.validation_size
DENSE_LAYERS = FLAGS.dense_layers
LAYER_SIZES = FLAGS.layer_sizes
CONV_LAYERS = FLAGS.conv_layers
SAVE = FLAGS.save

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)

def get_data():
    data = []
    with h5py.File(IN_FILE, 'r') as f: 
        for img in tqdm(f.keys()):
            # Read each channel 
            red = f[img]['1_R']
            green = f[img]['2_G']
            blue = f[img]['3_B']

            # Get classification
            img_class = f[img].attrs['classification']

            # Create numpy ndarray 
            img_arr = np.c_[red, green, blue]

            data.append([img_arr, img_class])

    random.shuffle(data)
    X = []
    y = []

    for feature, label in data: 
        X.append(feature)
        y.append(label)
    
    X = np.array(X)
    X = X.reshape(-1, X.shape[1], X.shape[2], X.shape[3])
    y = np.array(y)

    return X, y

def build_model():
    features, labels = get_data()

    norm_features = normalize(features)

    for dense_layer in DENSE_LAYERS:
        for layer_size in LAYER_SIZES:
            for conv_layer in CONV_LAYERS:
                log_path = os.path.join(LOG_DIR, "{}xConv-{}xNodes-{}xDense-{}".format(conv_layer, layer_size, dense_layer, datetime.now().strftime("%Y%m%d_%H%M%S")))
                tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)

                model = Sequential()
                model.add(Conv2D(layer_size, (3, 3), input_shape=(norm_features.shape[1], norm_features.shape[2], norm_features.shape[3])))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
    
                for i in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2,2)))
    
                model.add(Flatten())
                for i in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation("relu"))
    
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
    
                model.add(Dense(1))
                model.add(Activation("sigmoid"))
    
                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
    
                model.fit(norm_features, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                          validation_split=VALIDATION_SIZE, callbacks=[tensorboard])
                
                if SAVE:
                    path = os.path.join(MODEL_DIR, "{}xConv-{}xNodes-{}xDense-CNN.model".format(conv_layer, layer_size, dense_layer))
                    model.save(path)

build_model()