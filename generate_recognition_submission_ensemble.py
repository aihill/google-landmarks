#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file.
Uses ensemble of networks. """

from typing import *
import math, os, sys
from collections import defaultdict
from shutil import copyfile
import numpy as np,  pandas as pd       # type: ignore
from tqdm import tqdm                   # type: ignore
from matplotlib import pyplot as plt    # type: ignore

NpArray = Any
NON_LANDMARK_CLASSIFIER     = "experiments/no_class_1/pred.npz"

def load_test_data(path: str) -> List[str]:
    """ Loads CSV files into memory. """
    print("reading data...")
    data = pd.read_csv(path)
    x = data["id"].tolist()
    print("len(x)", len(x))
    print()
    return x

def merge_results(all_classes: List[NpArray], all_confidences: List[NpArray],
                  sampleIdx: int) -> Tuple[int, float]:
    """ Merges predictions from all models for a single sample. """
    assert(len(all_classes) == 2 and len(all_confidences) == 2)

    classes0 = zip(all_classes[0][sampleIdx, :], all_confidences[0][sampleIdx, :])
    classes1 = {class_: conf for class_, conf in zip(all_classes[1][sampleIdx, :],
                                                     all_confidences[1][sampleIdx, :])}

    best_prob, best_class = 0.0, -1
    for class_, conf1 in classes0:
        if class_ in classes1:
            prob = math.sqrt(conf1 * classes1[class_])
            if prob > best_prob:
                best_prob, best_class = prob, class_

    if best_prob > 0 and best_class >= 0:
        return best_class, best_prob
    elif all_confidences[0][sampleIdx, 0] > all_confidences[1][sampleIdx, 0]:
        return all_classes[0][sampleIdx, 0], all_confidences[0][sampleIdx, 0]
    else:
        return all_classes[1][sampleIdx, 0], all_confidences[1][sampleIdx, 0]

if __name__ == "__main__":
    ###########################################################################
    # load junk classifier
    ###########################################################################
    print("loading non-landmark classifier")
    junk = np.load(NON_LANDMARK_CLASSIFIER)
    pred_indices = junk["pred_indices"]
    images = junk["images"]
    images = [os.path.splitext(path)[0] for path in images]
    print("first 10 images from non-landmark classifier table:", images[:10])
    non_landmarks = {img for img, pred in zip(images, pred_indices) if pred[0] == 0}
    print("first 10 non-landmarks:", sorted(list(non_landmarks))[:10])

    ###########################################################################
    # load prediction data
    ###########################################################################
    predictions = [
        "experiments/3B/pred.npz",
        "experiments/4A/pred.npz"
        ]
    image_list: List[str] = []
    all_confidences: List[NpArray] = []
    all_classes: List[NpArray] = []

    for filename in predictions:
        print("getting results from", filename)
        data = np.load(filename)
        print(data)

        classes = data["pred_indices"]
        images = data["images"]
        confidences = data["pred_confs"]

        print("classes", classes.shape, classes)
        print("images", images.shape, images)
        print("confidences", confidences.shape, confidences)
        all_confidences.append(confidences)
        all_classes.append(classes)

        if len(image_list) == 0:
            image_list = images
        else:
            assert(np.all(image_list == images))

    ###########################################################################
    # generate results
    ###########################################################################
    data = defaultdict(str)
    for i in range(images.size):
        name = os.path.splitext(image_list[i])[0]
        class_, conf = merge_results(all_classes, all_confidences, i)

        if name not in non_landmarks:
            data[name] = "%d %f" % (class_, conf)

    x_test = load_test_data("recognition/test.csv")
    values = [""] * len(x_test)

    for i, image in enumerate(x_test):
        values[i] = data[image]

    df = pd.DataFrame({"id": x_test, "landmarks": values})
    os.makedirs("submissions_recognition/", exist_ok=True)
    df.to_csv("submissions_recognition/prediction.csv", index=False)
