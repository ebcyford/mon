"""Build network and train on prepared data.

This architecture was tested and hyperparameters tweaked such that loss
was minimized and accuracy within training and validation were strongly
correlated. Additional arguments may be passed in the command line, 
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
from mon.utils import prep_chip
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import normalize
from tqdm import tqdm


def get_data(training_file, img_size=80):
    """Returns two numpy arrays
    
    Arguments: 
        training_file: str, path to HDF5 training data file
        img_size: int, size of image chip to train on
    """
    data = []
    with h5py.File(training_file, 'r') as f: 
        for img in tqdm(f.keys(), desc="Reading Training Data"):
            datagroup = f[img]

            # Read each channel and create a numpy array
            iteration = iter(datagroup)
            first_channel = next(iteration)
            img_arr = np.array(datagroup[first_channel])
            for c in iteration:
                img_arr = np.c_[img_arr, np.array(datagroup[c])]

            # Get classification 
            img_class = datagroup.attrs['classification']

            # Resize to specified size
            img_arr = prep_chip(img_arr, img_size=img_size)

            # Add to output array
            data.append([img_arr, img_class])
    
    # randomly shuffle data 
    random.shuffle(data)

    #split into features and labels
    X = []
    y = []
    for feature, label in data: 
        X.append(feature)
        y.append(label)

    # Back to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Normalize features and reshape for input tensor
    X = normalize(X)
    X = X.reshape(-1, X.shape[1], X.shape[2], X.shape[3])

    return X, y


def build_model(in_shape):
    """Returns a tensorflow.python.keras.saving.saved_model

    Arguments:
        in_shape: tuple, shape of input tensor channels last
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), 
                     input_shape=in_shape))
    model.add(Conv2D(64, (2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.025))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.025))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(64))
    model.add(Activation("relu"))
    
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    return model


def main():
    # Retrieve training data
    print("Retrieving Data...")
    features, labels = get_data(IN_FILE, img_size=IMG_SIZE)

    # Build and compile model
    print("Building Model...")
    model = build_model(features.shape[1:])
    adam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, 
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-07,
                                    amsgrad=False)
    model.compile(loss="binary_crossentropy",
                  optimizer=adam, 
                  metrics=["accuracy"])
    model.summary()

    # Train model on training data
    print("Training Model...")
    model.fit(features, labels, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            validation_split=VALIDATION_SIZE, 
            callbacks=[tensorboard])

    print("Saving...")
    model.save(SAVE_FILE)
    print("Model Saved to {}".format(SAVE_FILE))


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DEFAULT_IN = os.path.join(BASE_DIR, "training_data.h5")
    DEFAULT_LOG = os.path.join(BASE_DIR, "logs")
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DEFAULT_SAVE = os.path.join(MODEL_DIR, "model_" + TIMESTAMP)

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, default=DEFAULT_IN,
                        help="HDF5 file of training data chips [default:data\\training_data.h5")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG,
                        help="directory of log files [default:logs")
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE, 
                        help="where to save model file [default:models\\model_+TIMESTAMP.model]")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="number images to process in each batch [default:512]")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="number of epochs to train model [default:20]")
    parser.add_argument("--validation_size", type=int, default=0.1,
                        help="size of validations set [default:0.1]")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for ADAM optimizer")
    parser.add_argument("--name", type=str, default=TIMESTAMP,
                        help="name of model, timestamp appended")
    parser.add_argument("--img_size", type=int, default=80, 
                        help="height of square image chip in pixels [default:80]")

    FLAGS = parser.parse_args()

    IN_FILE = FLAGS.in_file
    LOG_DIR = FLAGS.log_dir
    SAVE_FILE = FLAGS.save_file
    BATCH_SIZE = FLAGS.batch_size
    EPOCHS = FLAGS.epochs
    VALIDATION_SIZE = FLAGS.validation_size
    LEARNING_RATE = FLAGS.learning_rate
    NAME = FLAGS.name
    IMG_SIZE = FLAGS.img_size

    if NAME != TIMESTAMP:
        log_path = os.path.join(LOG_DIR, NAME + "_" + TIMESTAMP)
    else: 
        log_path = os.path.join(LOG_DIR, NAME)

    print("Logging to: " + log_path)
    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)

    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)

    main()