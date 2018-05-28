#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

from typing import *
from collections import defaultdict
import os, os.path as osp
import numpy as np,  pandas as pd       # type: ignore
from easydict import EasyDict as edict          # type: ignore
from world import cfg, create_logger

NpArray = Any
opt = edict()

opt.FEATURES = edict()
opt.FEATURES.CODENAME = 'feature_extractor'
opt.FEATURES.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.FEATURES.CODENAME)
opt.FEATURES.TEST = osp.join(opt.FEATURES.DIR, 'features_test_0.npz')

opt.RETRIEVAL = edict()
opt.RETRIEVAL.CODENAME = 'retrieval'
opt.RETRIEVAL.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.RETRIEVAL.CODENAME)
opt.RETRIEVAL.DISTANCES = osp.join(opt.RETRIEVAL.DIR, 'distances.npz')

opt.TEST = edict()
opt.TEST.CSV = "retrieval/test.csv"

EPSILON = 0.5

if __name__ == "__main__":
    distances = np.load(opt.RETRIEVAL.DISTANCES)
    print(distances)

    landmarks = np.transpose(distances["indices"])
    distances = np.transpose(distances["distances"])
    print("landmarks", landmarks.shape)
    print(landmarks)
    print("distances", distances.shape)
    print(distances)

    test_classes = np.load(opt.FEATURES.TEST)
    print(test_classes)

    images = test_classes["images"]
    features = test_classes["features"]

    print("images", images.shape)
    print(images)
    images = [os.path.splitext(name)[0] for name in images]

    data: DefaultDict[str, str] = defaultdict(str)

    for img, candidates, dists in zip(images, landmarks, distances):
        L = [os.path.splitext(lm)[0] for lm, d in zip(candidates, dists) if d < EPSILON]
        data[img] = " ".join(L)

    csv_data = pd.read_csv(opt.TEST.CSV)
    x_test = csv_data["id"].tolist()
    print("len(x_test)", len(x_test))
    print()

    values = [data[x] for x in x_test]

    df = pd.DataFrame({"id": x_test, "images": values})
    os.makedirs("submissions_retrieval/", exist_ok=True)
    df.to_csv("submissions_retrieval/prediction.csv", index=False)
