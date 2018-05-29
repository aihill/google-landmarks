#!/usr/bin/python3.6
""" For every image from test set, searches for the top-100 closes landmarks. """

from typing import *
from glob import glob
import os, os.path as osp, pprint
import numpy as np          # type: ignore
from tqdm import tqdm       # type: ignore
import faiss                # type: ignore

NpArray     = Any
TwoNpArrays = Tuple[NpArray, NpArray]

FEATURES_DIR                = "experiments/feature_extractor_recognition/"
FEATURES_TEST_FILE          = FEATURES_DIR + "features_test_0.npz"
RETRIEVAL_DISTANCES_FILE    = "experiments/recognition/distances.npz"

K = 100

def search_against_fragment(fragment: NpArray, test_vectors: NpArray) -> TwoNpArrays:
    # build a flat index (CPU)
    index_flat = faiss.IndexFlatL2(d)

    # make it into a GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    print("loading", fragment)
    landmark_data = np.load(fragment)
    index_names, index_vectors = landmark_data["images"], landmark_data["features"]
    print("index_names:", index_names.shape)
    print("vectors shape", index_vectors.shape)

    gpu_index_flat.add(index_vectors)
    print("total size of index:", gpu_index_flat.ntotal)

    # print("sanity search...")
    # distances, index = gpu_index_flat.search(index_vectors[:10], K)  # actual search
    # print(index[0])
    # print(distances[0])

    print("searching")
    distances, index = gpu_index_flat.search(test_vectors, K)  # actual search
    index = index_names[index]
    print(index[:10, :5])
    print(distances[:10, :5])
    return index, distances

def merge_results(index1: NpArray, distances1: NpArray, index2: NpArray,
                  distances2: NpArray) -> TwoNpArrays:
    """ Returns top-K of two sets. """
    print("merging results")
    assert index1.shape == distances1.shape and index2.shape == distances2.shape
    assert index1.shape[1] == index2.shape[1]

    joint_index = np.hstack((index1, index2))
    joint_distances = np.hstack((distances1, distances2))
    print("joint_index", joint_index.shape, "joint_distances", joint_distances.shape)
    assert joint_index.shape == joint_distances.shape

    best_indices = np.zeros((index1.shape[0], K), dtype=object)
    best_distances = np.zeros((index1.shape[0], K), dtype=np.float32)

    for sample in range(joint_index.shape[0]):
        closest_indices = np.argsort(joint_distances[sample, :])
        # print("closest_indices", closest_indices.shape, closest_indices)

        best_indices[sample, :] = joint_index[sample, closest_indices][:K]
        # print("best_indices[sample]", best_indices[sample])

        best_distances[sample, :] = joint_distances[sample, closest_indices][:K]

    print("best_index", best_index.shape, "best_distances", best_distances.shape)
    print(best_indices[:10, :5])
    print(best_distances[:10, :5])
    return best_indices, best_distances

if __name__ == "__main__":
    if not osp.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    """ Iterates through all train files. """
    test_data = np.load(FEATURES_TEST_FILE)
    test_names, test_vectors = test_data["images"], test_data["features"]

    print(test_names.shape)
    print(test_names)
    print(test_vectors.shape)
    print(test_vectors)
    print("first vector:")
    print("shape", test_vectors[0].shape, "non-zeros", np.count_nonzero(test_vectors[0]))
    d = test_vectors[0].shape[0]

    print("initializing CUDA")
    res = faiss.StandardGpuResources()

    best_index, best_distance = None, None
    for fragment in glob(osp.join(FEATURES_DIR, "features_train_*.npz")):
        idx, dist = search_against_fragment(fragment, test_vectors)

        if best_index is None:
            best_index, best_distances = idx, dist
        else:
            best_index, best_distances = merge_results(best_index, best_distances, idx, dist)

    print("saving results to", RETRIEVAL_DISTANCES_FILE)
    np.savez(RETRIEVAL_DISTANCES_FILE,
             indices=best_index.T, distances=best_distances.T) # type:ignore
