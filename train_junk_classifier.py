#!/usr/bin/python3.6
""" Combines pretrained network with softmax layer and possibly a FC layer. """

from typing import *
import os
from glob import glob
import numpy as np                  # type: ignore
from skimage.io import imread       # type: ignore
from tqdm import tqdm               # type: ignore
from skimage.transform import resize                    # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
from keras.utils import to_categorical                  # type: ignore
from keras.models import Sequential                     # type: ignore
from keras.layers import Dense, Dropout, Flatten        # type: ignore
from keras.layers import Conv2D, MaxPooling2D           # type: ignore
from keras.callbacks import ModelCheckpoint, CSVLogger  # type: ignore
from keras.callbacks import ReduceLROnPlateau           # type: ignore

NpArray = Any
IMAGE_SIZE = 250        # test set now uses resolution 256x256

def load_pics_from_dir(path: str) -> NpArray:
    """ Reads all images from the directory. """
    def load(img: NpArray) -> NpArray:
        return resize(imread(img), (IMAGE_SIZE, IMAGE_SIZE))

    print("reading %s" % path)
    res = np.array([load(img) for img in tqdm(glob(os.path.join(path, "*.jpg")))])
    print("dataset shape", res.shape)
    return res

if __name__ == "__main__":
    positives = load_pics_from_dir("data/junk_classifier/true_classes")
    negatives = load_pics_from_dir("data/junk_classifier/false_classes")

    x = np.concatenate((positives, negatives), axis=0)
    y = np.concatenate(([1] * positives.shape[0], [0] * negatives.shape[0]), axis=0)

    x -= np.mean(x)
    x = x / np.std(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    batch_size = 1
    epochs = 50

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()

    cb = [
        ModelCheckpoint("models/no_class.epoch_{epoch:02d}-acc_{val_acc:.4f}.hdf5",
                        monitor='val_acc', save_best_only=True),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5, verbose=1),
        CSVLogger("models/training_no_class_log.csv")
    ]

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=cb,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
