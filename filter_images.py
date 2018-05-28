#!/usr/bin/python3.6
""" Crops and downsamples image sets. Saves result into jpeg. """

import glob, multiprocessing, os, sys
from typing import *
import pandas as pd         # type: ignore
from tqdm import tqdm       # type: ignore
from PIL import Image       # type: ignore

def check_image(path: str) -> int:
    """ Loads image file, crops it and resizes into the proper resolution. """
    name = os.path.splitext(os.path.basename(path))[0]
    if name in blacklist:
        # print("%s blacklisted" % name)
        os.unlink(path)
        return 1

    try:
        img = Image.open(path)
        return 0
    except (IOError, OSError):
        print("%s read error" % name)
        os.unlink(path)
        return 1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <image_dir>" % sys.argv[0])
        sys.exit(0)

    csv = pd.read_csv("retrieval/deleted_or_offline_index_image_keys.txt", header=None)
    blacklist = csv[0]
    print("blacklist:\n", blacklist)
    blacklist = set(blacklist)

    image_dir = sys.argv[1]
    file_list = list(glob.glob(os.path.join(image_dir, "*.jpg")))
    pool = multiprocessing.Pool(processes=16)
    failures = sum(tqdm(pool.imap_unordered(check_image, file_list), total=len(file_list)))
    print('total number of filtered files:', failures)
    pool.close()
    pool.terminate()
