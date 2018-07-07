#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

from typing import *
from collections import defaultdict
import math, os, os.path as osp, shutil
import numpy as np,  pandas as pd       # type: ignore
from tqdm import tqdm                   # type: ignore

NpArray = Any

FEATURES_TEST_FILE          = "experiments/feature_extractor_retrieval/features_test_0.npz"
RETRIEVAL_DISTANCES_FILE    = "experiments/retrieval/distances.npz"
NON_LANDMARK_CLASSIFIER     = "experiments/no_class_1/pred.npz"
TEST_CSV                    = "retrieval/test.csv"

ENABLE_DEBUGGING            = True
ENABLE_DETAIL_DEBUGGING     = False

MAX_DISTANCE    = 1500
MAX_GAP         = 300

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    print("loading non-landmark classifier")
    junk = np.load(NON_LANDMARK_CLASSIFIER)
    pred_indices = junk["pred_indices"]
    images = junk["images"]
    images = [os.path.splitext(path)[0] for path in images]
    print("first 10 images from non-landmark classifier table:", images[:10])
    non_landmarks = {img for img, pred in zip(images, pred_indices) if pred[0] == 0}

    print("reading distances and indices arrays")
    distances = np.load(RETRIEVAL_DISTANCES_FILE)

    landmarks = np.transpose(distances["indices"])
    distances = np.transpose(distances["distances"])
    print("landmarks", landmarks.shape)
    print(landmarks)
    print("distances", distances.shape)
    print(distances)

    test_classes = np.load(FEATURES_TEST_FILE)
    images = test_classes["images"]
    print("images", images.shape)
    print(images)
    images = [os.path.splitext(name)[0] for name in images]

    data: DefaultDict[str, str] = defaultdict(str)

    if ENABLE_DEBUGGING or ENABLE_DETAIL_DEBUGGING:
        root_dir = osp.abspath(osp.dirname(__file__)) + "/"
        debug_dir = root_dir + "experiments/statistics/"
        print("removing old debug data")

        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)

        os.makedirs(debug_dir)

    print("analyzing all test cases")
    all = list(zip(images, landmarks, distances))

    for img, candidates, dists in tqdm(all):
        if img in non_landmarks:
            continue

        pairs = sorted(zip(candidates, dists), key=lambda pair: pair[1])
        first_distance = pairs[0][1]

        if ENABLE_DETAIL_DEBUGGING:
            # create directories structure, according to the distances
            d = pairs[0][1]
            d = math.ceil(d / 100) * 100

            directory = debug_dir + str(d) + "/"
            os.makedirs(directory, exist_ok=True)

            os.symlink(root_dir + "retrieval/test/fakeclass/" + img + ".jpg",
                       directory + img + ".jpg")
        elif ENABLE_DEBUGGING and first_distance < MAX_DISTANCE:
            directory = debug_dir + img + "/"
            os.makedirs(directory)
            os.symlink(root_dir + "retrieval/test/fakeclass/" + img + ".jpg",
                       directory + "00_original.jpg")
            i = 1

            for lm, dist in pairs:
                if dist < MAX_DISTANCE and dist < first_distance + MAX_GAP:
                    lm = os.path.splitext(lm)[0]
                    name = "%02d" % i; i += 1
                    os.symlink(root_dir + "retrieval/train/fakeclass/" + lm + ".jpg",
                               directory + name + "_" + str(dist) + ".jpg")

        L = [os.path.splitext(lm)[0] for lm, d in pairs if d < MAX_DISTANCE and
             d < first_distance + MAX_GAP]
        data[img] = " ".join(L)

    print("number of test samples analyzed:", len(data))
    print("number of non-empty test samples:", len([img for img in data if data[img]]))

    csv_data = pd.read_csv(TEST_CSV)
    x_test = csv_data["id"].tolist()
    print("len(x_test)", len(x_test))
    print()

    values = [data[x] for x in x_test]

    df = pd.DataFrame({"id": x_test, "images": values})
    os.makedirs("submissions_retrieval/", exist_ok=True)
    df.to_csv("submissions_retrieval/prediction.csv", index=False)
