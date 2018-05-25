#!/usr/bin/python3.6
""" Performs prediction, using one of trained models.
Saved weights from the file must match the model code from train.py. """

import os, sys
from typing import *
import numpy as np                      # type: ignore
from skimage.io import imread           # type: ignore
from tqdm import tqdm                   # type: ignore
import pandas as pd                     # type: ignore
from keras.models import load_model     # type: ignore
from skimage.transform import resize    # type: ignore
from predict import load_test_data

NpArray = Any
IMAGE_SIZE = 250

def load_submission(path: str) -> List[str]:
    """ Loads CSV files into memory. """
    print("reading data...")
    data = pd.read_csv(path)
    y = data["landmarks"].tolist()
    print("len(x)", len(y))
    print()
    return y

def test_image(image_name: str) -> Tuple[str, bool]:
    """ Returns whether image is landmark or not. """
    try:
        img = imread(os.path.join("data/test/", image_name + ".jpg"))
    except FileNotFoundError:
        return image_name, False

    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.expand_dims(img, axis=0)
    # img -= np.mean(img)
    # img = img / np.std(img)

    prediction = model.predict(img)
    print("prediction", prediction)
    res = prediction[0] >= 0.5
    print("res", res)
    return res

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: %s <result.csv> <model_path> <candidate.csv>" % sys.argv[0])
        sys.exit(0)

    result_csv, candidate_csv = sys.argv[1], sys.argv[3]
    print("loading model")
    model = load_model(sys.argv[2])
    print("loading test data")
    x_test = load_test_data("data/recognition/test.csv")
    print("loading submission")
    y_test = load_submission(candidate_csv)

    # TODO: implement normalization
    # x_test -= np.mean(x_test)
    # x_test = x_test / np.std(x_test)

    print("testing images...")
    for image_name in tqdm(x_test):
        res = test_image(image_name)
        if not res:
            y_test[image_name] = ''

    print("done")
    df = pd.DataFrame({"id": x_test, "landmarks": y_test})
    os.makedirs("./results/", exist_ok=True)
    df.to_csv(result_csv, index=False)
