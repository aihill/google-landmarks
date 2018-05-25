#!/usr/bin/python3.6
""" Performs prediction, using one of trained models.
Saved weights from the file must match the model code from train.py. """

import os, sys
from glob import glob
from typing import *
import numpy as np                      # type: ignore
from skimage.io import imread           # type: ignore
from tqdm import tqdm                   # type: ignore
import pandas as pd                     # type: ignore
from keras.models import load_model     # type: ignore
from skimage.transform import resize    # type: ignore

NpArray = Any
IMAGE_SIZE = 250
# BATCH_SIZE = 300
BATCH_SIZE = 10

def load_test_data(path: str) -> List[str]:
    """ Loads CSV files into memory. """
    print("reading data...")
    data = pd.read_csv(path)
    x = data["id"].tolist()
    print("len(x)", len(x))
    print()
    return x

def load_submission(path: str) -> List[str]:
    """ Loads CSV files into memory. """
    print("reading data...")
    data = pd.read_csv(path)
    y = data["landmarks"].tolist()
    print("len(x)", len(y))
    print()
    return y

def load_image(image_name: str) -> NpArray:
    """ Returns whether image is landmark or not. """
    try:
        path = os.path.join("data/test/", image_name + ".jpg")
        img = imread(path)
        # img = imread(image_name)

        startx, starty = (img.shape[0]-IMAGE_SIZE) // 2, (img.shape[1]-IMAGE_SIZE) // 2
        img = img[startx : startx + IMAGE_SIZE, starty : starty + IMAGE_SIZE, :]
        return img
    except FileNotFoundError:
        # print("error reading %s" % image_name)
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

def load_batch(images: List[str]) -> NpArray:
    batch = [load_image(image) for image in images]
    data = np.array(batch)

    data = data - np.mean(data, dtype=np.float32)
    data = data / np.std(data, dtype=np.float32)

    return data

def predict_on_batch(images: NpArray) -> List[bool]:
    batch = load_batch(images)
    predictions = np.zeros(batch.shape[0], dtype=bool)

    # since I trained model on CPU, batch size is 1
    for i in range(batch.shape[0]):
        pred = model.predict(batch[i:i+1])
        predictions[i] = pred[0] >= 0.5
        # print("image=%s result=%s" % (images[i], predictions[i]))

    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: %s <result.csv> <model_path> <candidate.csv>" % sys.argv[0])
        sys.exit(0)

    result_csv, candidate_csv = sys.argv[1], sys.argv[3]
    print("loading model")
    model = load_model(sys.argv[2])
    print("loading test data")
    x_test = load_test_data("data/test.csv")
    print("loading submission")
    y_test = load_submission(candidate_csv)

    # # debug validation
    # x_test = list(glob("../google_landmark/data/junk_classifier/false_classes/*"))

    batches = [x_test[i : i + BATCH_SIZE] for i in range(0, len(x_test), BATCH_SIZE)]
    non_landmarks: Set[str] = set()

    print("testing images")
    for batch in tqdm(batches):
        pred = predict_on_batch(batch)

        for image, res in zip(batch, pred):
            if not res:
                non_landmarks.add(image)

    for i, image in enumerate(x_test):
        if image in non_landmarks:
            y_test[i] = ''

    print("generating submission file")
    df = pd.DataFrame({"id": x_test, "landmarks": y_test})
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    df.to_csv(result_csv, index=False)
