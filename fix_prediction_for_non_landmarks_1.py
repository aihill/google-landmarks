#!/usr/bin/python3.6
""" Performs prediction, using one of trained models.
Saved weights from the file must match the model code from train.py. """

import os, os.path as osp, sys
from glob import glob
from typing import *
import numpy as np                      # type: ignore
from tqdm import tqdm                   # type: ignore
import pandas as pd                     # type: ignore
from easydict import EasyDict as edict  # type: ignore
from world import cfg

opt = edict()

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = 'no_class_1'
opt.EXPERIMENT.TASK = 'test'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.TEST = edict()
opt.TEST.OUTPUT = osp.join(opt.EXPERIMENT.DIR, 'pred.npz')

NpArray = Any
IMAGE_SIZE          = 256
BATCH_SIZE          = 10
DEBUG_VALIDATION    = False

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: %s <result.csv> <candidate.csv>" % sys.argv[0])
        sys.exit(0)

    result_csv, candidate_csv = sys.argv[1], sys.argv[2]
    print("loading predictions")
    data = np.load(opt.TEST.OUTPUT)

    pred_indices = data["pred_indices"]
    images = data["images"]
    images = [os.path.splitext(path)[0] for path in images]
    print(images[:10])

    if DEBUG_VALIDATION:
        x_test = list(glob("data/junk_classifier/true_classes/*.jpg"))
        # x_test = list(glob("data/junk_classifier/false_classes/*.jpg"))
        x_test = [os.path.splitext(os.path.basename(path))[0] for path in x_test]
    else:
        print("loading test data")
        x_test = load_test_data("data/test.csv")
        print("loading submission")
        y_test = load_submission(candidate_csv)

    non_landmarks = {img for img, pred in zip(images, pred_indices) if pred[0] == 0}
    # print(list(non_landmarks)[:10])

    if DEBUG_VALIDATION:
        for img in x_test:
            print("image=%s test=%s" % (img, img not in non_landmarks))
    else:
        for i, image in enumerate(x_test):
            if image in non_landmarks:
                y_test[i] = ''

        print("generating submission file")
        df = pd.DataFrame({"id": x_test, "landmarks": y_test})
        os.makedirs(os.path.dirname(result_csv), exist_ok=True)
        df.to_csv(result_csv, index=False)
