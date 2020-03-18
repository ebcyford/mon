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
from mon.utils import prep_chip, normalize
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tqdm import tqdm

tf.get_logger().setLevel("INFO")

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


def build_model(in_shape, learning_rate=0.0001):
    """Returns a tensorflow.python.keras.saving.saved_model

    Arguments:
        in_shape: tuple, shape of input tensor channels last
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), 
                     input_shape=in_shape,
                     activation="relu",
                     padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-07,
                                         amsgrad=False)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer, 
                  metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model


def main():
    # Retrieve training data
    print("Retrieving Data...")
    features, labels = get_data(IN_FILE, img_size=IMG_SIZE)

    if (VALIDATION_SIZE == 0.0):
        # Cross validation
        cross_val = KFold(n_splits=FOLDS)
        fold = 1

        for train_idx, test_idx in cross_val.split(features):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                
                log_path = os.path.join(LOG_DIR, NAME + "_fold" + str(fold))

                save_file = os.path.join(SAVE_DIR, "{}_fold{}.model".format(NAME, fold))

                fold += 1

                print("Model File: " + save_file)
                print("Log Path: " + log_path)
                
                tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)
                
                model = build_model(in_shape=X_train.shape[1:])
                model.fit(x=X_train, y=y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=[tensorboard])
                
                model.save(save_file)
    else:
        log_path = os.path.join(LOG_DIR, NAME)
        save_file = os.path.join(MODEL_DIR, "{}.model".format(NAME))

        print("Model File: " + save_file)
        print("Log Path: " + log_path)

        tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)
        model = build_model(in_shape=features.shape[1:])
        model.fit(x=features, y=labels,
                  batch_size=BATCH_SIZE, 
                  epochs=EPOCHS,
                  validation_split=VALIDATION_SIZE,
                  callbacks=[tensorboard])
        model.save(save_file)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DEFAULT_IN = os.path.join(DATA_DIR, "training_data.h5")
    DEFAULT_NAME = TIMESTAMP

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, default=DEFAULT_IN,
                        help="HDF5 file of training data chips [default:data\\training_data.h5")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="directory to save log files [default:logs]")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR,
                        help="directory to save models [default:models]")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help="name of model, timestamp appended")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="number of images to process in each batch [default:256]")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train model [default:10]")
    parser.add_argument("--folds", type=int, default=10,
                        help="number of fold in k-fold cross validation [default:10]")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="learning rate for ADAM optimizer [default:0.0001]")
    parser.add_argument("--img_size", type=int, default=80,
                        help="height of square image chip in pixels [default:80]")
    parser.add_argument("--validation_size", type=float, default=0.0,
                        help="size of validation set if not using k-fold [default:0.0]")

    FLAGS = parser.parse_args()

    IN_FILE = FLAGS.in_file 
    LOG_DIR = FLAGS.log_dir
    MODEL_DIR = FLAGS.model_dir
    NAME = FLAGS.name
    BATCH_SIZE = FLAGS.batch_size
    EPOCHS = FLAGS.epochs
    FOLDS = FLAGS.folds
    LEARNING_RATE = FLAGS.learning_rate
    IMG_SIZE = FLAGS.img_size
    VALIDATION_SIZE = FLAGS.validation_size

    if NAME != DEFAULT_NAME:
        NAME = NAME + "_" + TIMESTAMP

    SAVE_DIR = os.path.join(MODEL_DIR, NAME)

    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
    if not os.path.exists(SAVE_DIR): os.mkdir(SAVE_DIR)

    main()