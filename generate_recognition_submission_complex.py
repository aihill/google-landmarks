#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file.
Uses ensemble of networks. """

from typing import *
import math, os, shutil, sys
from collections import defaultdict
from shutil import copyfile
import numpy as np,  pandas as pd       # type: ignore
from tqdm import tqdm                   # type: ignore
from matplotlib import pyplot as plt    # type: ignore

NpArray = Any

FEATURES_TEST_FILE          = "experiments/feature_extractor/features_test_0.npz"
RETRIEVAL_DISTANCES_FILE    = "experiments/recognition/distances.npz"
NON_LANDMARK_CLASSIFIER     = "experiments/no_class_1/pred.npz"

ENABLE_DEBUGGING            = False
DISTANCE_FADE               = 150

def prepare_debugging(debug_dir: str) -> DefaultDict[int, List[str]]:
    print("removing old debug data")

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)

    os.makedirs(debug_dir)

    print("reading csv...")
    csv_path = "data/train.csv"
    data = pd.read_csv(csv_path)
    x: NpArray = data["id"].values
    y: NpArray = data["landmark_id"].values
    print("len(x)", len(x), "y.shape", y.shape)
    print()

    print("filtering data...")
    ok = [os.path.exists("data/train/" + p + ".jpg") for p in tqdm(x)]
    x, y = x[ok], y[ok]
    print("after filtering: len(x)", len(x), "y.shape", y.shape)

    print("parsing classes")
    classes: DefaultDict[int, List[str]] = defaultdict(list)

    for filename, class_ in zip(x, y):
        if len(classes[class_]) <= 20:
            classes[class_].append(filename)

    return classes

def load_test_data(path: str) -> List[str]:
    """ Loads CSV files into memory. """
    print("reading testset...")
    data = pd.read_csv(path)
    x = data["id"].tolist()
    print("len(x)", len(x))
    print()
    return x

def merge_two_lists(classes1: List[int], conf1: List[float], classes2: List[int], conf2:
                    List[float]) -> Tuple[List[int], List[float]]:
    """ Merges two lists of classes + confidence arrays. """
    second = {class_: conf for class_, conf in zip(classes2, conf2)}
    merged_classes, merged_confs = [], []

    for class_, confidence in zip(classes1, conf1):
        if class_ in second:
            merged_classes.append(class_)
            merged_confs.append(confidence * second[class_])

    return merged_classes, merged_confs

def merge_results(all_classes: List[NpArray], all_confidences: List[NpArray],
                  sampleIdx: int) -> Tuple[int, float]:
    """ Merges predictions from all models for a single sample. """
    L = len(all_classes)
    assert(L == len(all_confidences))
    mclasses, mconfs = all_classes[0][sampleIdx, :], all_confidences[0][sampleIdx, :]

    for classes, confs in zip(all_classes[1:], all_confidences[1:]):
        mclasses, mconfs = merge_two_lists(mclasses, mconfs, classes[sampleIdx, :],
                                           confs[sampleIdx, :])

    if not mclasses:
        return 0, 0
    else:
        assert(len(mclasses) == len(mconfs))
        conf = max(mconfs)
        return mclasses[mconfs.index(conf)], math.pow(conf, 1 / L)

def unsupervised_multiplier(landmarks: List[str], distances: List[float],
                            landmark2class: Dict[str, int], class_: int) -> float:
    """ Returns low number if the test image is too far from its landmark. """
    min_dist = distances[0]
    d = [dist for lm, dist in zip(landmarks, distances) if landmark2class[lm] == class_]

    if not d:
        return 0
    else:
        dist = d[0]
        assert dist >= min_dist
        return math.exp(-(dist - min_dist) / DISTANCE_FADE)

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
        "experiments/2B/pred.npz",
        "experiments/3A/pred.npz",
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
    # read information about distances
    ###########################################################################
    print("reading distances and indices arrays")
    distances = np.load(RETRIEVAL_DISTANCES_FILE)

    landmarks = np.transpose(distances["indices"])
    distances = np.transpose(distances["distances"])
    print("landmarks", landmarks.shape)
    print(landmarks)
    print("distances", distances.shape)
    print(distances)

    train_csv_path = "data/train.csv"
    data = pd.read_csv(train_csv_path)
    x: List[str] = data["id"].tolist()
    y: NpArray = data["landmark_id"].values
    landmark2class = {lm + ".jpg" : class_ for lm, class_ in zip(x, y)}

    ###########################################################################
    # generate results
    ###########################################################################
    data = defaultdict(str)

    if ENABLE_DEBUGGING:
        root_dir = os.path.abspath(os.path.dirname(__file__)) + "/"
        debug_dir = root_dir + "experiments/recognition_stats/"

        classes_dict = prepare_debugging(debug_dir)

    print("processing test set")

    for i in tqdm(range(images.size)):
        test_img = os.path.splitext(image_list[i])[0]
        if test_img in non_landmarks:
            continue

        class_, conf = merge_results(all_classes, all_confidences, i)

        conf *= unsupervised_multiplier(landmarks[i, :], distances[i, :], landmark2class,
                                        class_)

        if conf != 0:
            data[test_img] = "%d %f" % (class_, conf)

            if ENABLE_DEBUGGING:
                directory = debug_dir + str(class_) + "/"

                if not os.path.exists(directory):
                    # create directory, copy a few originals
                    os.mkdir(directory)
                    classes = classes_dict[class_]

                    for filename in classes:
                        os.symlink(root_dir + "data/train/" + filename + ".jpg",
                                   directory + "orig_" + filename + ".jpg")

                # copy test sample, mention probabilities
                os.symlink(root_dir + "data/test/" + test_img + ".jpg",
                           directory + str(conf) + "_" + test_img + ".jpg")

    x_test = load_test_data("recognition/test.csv")
    values = [""] * len(x_test)

    for i, image in enumerate(x_test):
        values[i] = data[image]

    df = pd.DataFrame({"id": x_test, "landmarks": values})
    os.makedirs("submissions_recognition/", exist_ok=True)
    df.to_csv("submissions_recognition/prediction.csv", index=False)
