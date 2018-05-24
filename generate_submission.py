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

    print("pred_indices", pred_indices.shape, pred_indices)
    print("pred_scores", pred_scores.shape, pred_scores)
    print("images", images.shape, images)

    predictions = dict()

    for i in range(images.size):
        name = os.path.splitext(images[i])[0]
        weights = pred_scores[i, :]
        p = np.exp(weights[0]) / np.sum(np.exp(weights))
        print("weights", weights)
        print("p", p)

        predictions[name] = pred_indices[i, 0]
        print("predictions[%s] = %d %f" % (name, predictions[name], p))

    x_test = load_test_data("recognition/test.csv")
    num_classes = 14951

    values = [""] * len(x_test)

    for i, image in enumerate(x_test):
        if image in predictions:
            values[i] = predictions[image]

    df = pd.DataFrame({"id": x_test, "landmarks": values})
    os.makedirs("./results/", exist_ok=True)
    df.to_csv("results/prediction.csv", index=False)
