#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

from typing import *
import os
from shutil import copyfile
import numpy as np,  pandas as pd       # type: ignore
from tqdm import tqdm                   # type: ignore
from matplotlib import pyplot as plt    # type: ignore

NpArray = Any

def load_test_data(path: str) -> List[str]:
    """ Loads CSV files into memory. """
    print("reading data...")
    data = pd.read_csv(path)
    x = data["id"].tolist()
    print("len(x)", len(x))
    print()
    return x

if __name__ == "__main__":
    filename = "experiments/2B/pred.npz"
    data = np.load(filename)
    print(data)

    pred_indices = data["pred_indices"]
    pred_scores = data["pred_scores"]
    images = data["images"]
    confidences = data["pred_confs"]

    print("pred_indices", pred_indices.shape)
    print(pred_indices)
    print("pred_scores", pred_scores.shape)
    print(pred_scores)
    print("images", images.shape)
    print(images)
    print("confidences", confidences.shape)
    print(confidences)

    predictions = dict()

    for i in range(images.size):
        name = os.path.splitext(images[i])[0]
        index = pred_indices[i, 0]
        conf = confidences[i, 0]
        value = "%d %f" % (index, conf)
        # print("predictions[%s] = %s" % (name, value))
        predictions[name] = value

    x_test = load_test_data("recognition/test.csv")

    values = [""] * len(x_test)

    for i, image in enumerate(x_test):
        if image in predictions:
            values[i] = predictions[image]

    df = pd.DataFrame({"id": x_test, "landmarks": values})
    os.makedirs("submissions/", exist_ok=True)
    df.to_csv("submissions/prediction.csv", index=False)
