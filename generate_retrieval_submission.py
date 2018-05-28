#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

from typing import *
from collections import defaultdict
import os, os.path as osp, shutil
import numpy as np,  pandas as pd       # type: ignore
from tqdm import tqdm                   # type: ignore

NpArray = Any

FEATURES_TEST_FILE          = "experiments/feature_extractor/features_test_0.npz"
RETRIEVAL_DISTANCES_FILE    = "experiments/retrieval/distances.npz"
TEST_CSV                    = "retrieval/test.csv"

ENABLE_DEBUGGING = True
EPSILON = 1.0

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
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

    if ENABLE_DEBUGGING:
        root_dir = osp.abspath(osp.dirname(__file__)) + "/"
        debug_dir = root_dir + "experiments/debug/"
        print("removing old debug data")
        shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)

    print("analyzing all test cases")
    all = list(zip(images, landmarks, distances))

    for img, candidates, dists in tqdm(all):
        pairs = sorted(zip(candidates, dists), key=lambda pair: pair[1])

        if ENABLE_DEBUGGING:
            directory = debug_dir + img + "/"
            os.makedirs(directory, exist_ok=True)
            os.symlink(root_dir + "retrieval/test/fakeclass/" + img + ".jpg",
                       directory + "original.jpg")

            for lm, dist in pairs[:10]:
                if dist < EPSILON:
                    lm = os.path.splitext(lm)[0]
                    os.symlink(root_dir + "retrieval/train/fakeclass/" + lm + ".jpg",
                               directory + lm + "_" + str(dist) + ".jpg")

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
