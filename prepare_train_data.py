#!/usr/bin/python3.6
""" Parses data, builds train/dev datasets, bakes data into big files. """

from typing import *
import os
from shutil import copyfile
import numpy as np,  pandas as pd       # type: ignore
from tqdm import tqdm                   # type: ignore
from matplotlib import pyplot as plt    # type: ignore

NpArray = Any

if __name__ == "__main__":
    csv_path = "../data/recognition/train.csv"

    print("reading csv...")
    data = pd.read_csv(csv_path)
    x: List[str] = data["id"].tolist()
    y: NpArray = data["landmark_id"].values
    print("len(x)", len(x), "y.shape", y.shape)

    L = list(zip(x, y))

    for id, cls in tqdm(L):
        src = "../data/recognition/train/%s.jpg" % id
        directory = "recognition/%d" % cls
        dst = "recognition/%d/%s.jpg" % (cls, id)

        # print("copy '%s' to '%s'" % (src, dst))
        os.makedirs(directory, exist_ok=True)

        try:
            copyfile(src, dst, follow_symlinks=True)
        except FileNotFoundError:
            pass
