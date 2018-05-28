#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

from typing import *
from collections import defaultdict
import os, os.path as osp
import numpy as np,  pandas as pd       # type: ignore

NpArray = Any

FEATURES_TEST_FILE          = "experiments/feature_extractor/features_test_0.npz"
RETRIEVAL_DISTANCES_FILE    = "experiments/retrieval/distances.npz"
TEST_CSV                    = "retrieval/test.csv"

EPSILON = 0.2

if __name__ == "__main__":
    distances = np.load(RETRIEVAL_DISTANCES_FILE)
    print(distances)

    landmarks = np.transpose(distances["indices"])
    distances = np.transpose(distances["distances"])
    print("landmarks", landmarks.shape)
    print(landmarks)
    print("distances", distances.shape)
    print(distances)

    test_classes = np.load(FEATURES_TEST_FILE)
    print(test_classes)

    images = test_classes["images"]
    features = test_classes["features"]

    print("images", images.shape)
    print(images)
    images = [os.path.splitext(name)[0] for name in images]

    data: DefaultDict[str, str] = defaultdict(str)

    for img, candidates, dists in zip(images, landmarks, distances):
        pairs = sorted(zip(candidates, dists), key=lambda pair: pair[1], reverse=True)
        L = [os.path.splitext(lm)[0] for lm, d in pairs if d < EPSILON]
        data[img] = " ".join(L)

    print("len(data)", len(data))

    csv_data = pd.read_csv(TEST_CSV)
    x_test = csv_data["id"].tolist()
    print("len(x_test)", len(x_test))
    print()

    values = [data[x] for x in x_test]

    df = pd.DataFrame({"id": x_test, "images": values})
    os.makedirs("submissions_retrieval/", exist_ok=True)
    df.to_csv("submissions_retrieval/prediction.csv", index=False)
