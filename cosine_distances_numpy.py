#!/usr/bin/python3.6
""" For every image from test set, searches for the top-100 closes landmarks. """

from typing import *
from glob import glob
import os, os.path as osp, pprint
import numpy as np                              # type: ignore
from tqdm import tqdm                           # type: ignore

NpArray = Any

FEATURES_DIR                = "experiments/feature_extractor/"
FEATURES_TEST_FILE          = FEATURES_DIR + "features_test_0.npz"
RETRIEVAL_DISTANCES_FILE    = "experiments/retrieval/distances.npz"

K = 100

def find_landmarks(test_vectors: NpArray, best_indices: NpArray, best_distances: NpArray,
                   db_indices: NpArray, db_vectors: NpArray) -> Tuple[NpArray, NpArray]:
    """ Searches for the top-100 nearest landmarks for every test image. """
    print("test_vectors", test_vectors.shape)
    print("best_indices", best_indices.shape, "best_distances", best_distances.shape)
    print("db_indices", db_indices.shape, "db_vectors", db_vectors.shape)

    for base in tqdm(range(0, db_indices.shape[0], K)):
        # sizes (100, 115k) and (100, 1600), might be less then 100 in the end of array
        # indices = db_indices[base: base + K]
        expanded_shape = (K, best_distances.shape[1])
        indices = np.zeros(expanded_shape)
        indices[:] = db_indices[base: base + K].reshape(-1, 1)
        vectors = db_vectors[base: base + K, :]
        print("\nindices", indices.shape, "vectors", vectors.shape)

        # size (100, 115k)
        # new_distances = torch.mm(vectors, test_vectors)
        new_distances = np.matmul(vectors, test_vectors)
        print("new_distances", new_distances.shape)

        # both sizes are (200, 115k)
        # joint_distances = torch.cat((best_distances, new_distances))
        # joint_indices = torch.cat((best_indices, indices))
        joint_distances = np.vstack((best_distances, new_distances))
        joint_indices = np.vstack((best_indices, indices))
        print("joint_distances", joint_distances.shape, "joint_indices", joint_indices.shape)

        # both sizes are (100, 115k)
        # best_distances, indices = torch.topk(joint_distances, k=K, dim=0, largest=False)
        indices = np.argsort(joint_distances, axis=0)[0:K, :]
        best_distances = joint_distances[indices]
        print("best_distances", best_distances.shape, "indices", indices.shape)

        # size (100, 115k)
        # best_indices = torch.gather(joint_indices, dim=0, index=indices)
        best_indices = joint_indices[indices]
        print("best_indices", best_indices.shape)

    return best_indices, best_distances

if __name__ == "__main__":
    if not osp.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    """ Iterates through all train files. """
    test_data = np.load(FEATURES_TEST_FILE)
    test_names, test_vectors = test_data["images"], test_data["features"]

    SUBSET = 10
    test_names, test_vectors = test_names[:SUBSET], test_vectors[:SUBSET]

    print(test_names)
    print(test_vectors)
    print("first vector:")
    print("length", test_vectors[0].size, "non-zeros", np.count_nonzero(test_vectors[0]))

    # note transpose
    # test_vectors = torch.tensor(test_vectors.T, dtype=torch.float).cuda(async=True)
    test_vectors = test_vectors.T

    print("test_names:", test_names.shape)
    print("test_vectors:", test_vectors.shape)

    all_landmark_names: List[str] = []
    num_test_samples = test_vectors.shape[1]
    # best_indices = torch.zeros(K, num_test_samples, dtype=torch.int).cuda(async=True)
    best_indices = np.zeros((K, num_test_samples), dtype=int)
    # best_distances = torch.ones(K, num_test_samples, dtype=torch.float).cuda(async=True)
    best_distances = np.ones((K, num_test_samples), dtype=float)

    # test
    for fragment in glob(osp.join(FEATURES_DIR, "features_train_*.npz")):
        print("\n\nmatching against ", fragment)
        landmark_data = np.load(fragment)
        names, vectors = landmark_data["images"], landmark_data["features"]
        print("names", names.shape, names)
        print("vectors", vectors.shape, vectors)

        indices = np.arange(len(all_landmark_names), len(all_landmark_names) + len(names))
        all_landmark_names.extend(list(names))
        print("indices", indices.shape, indices)

        # indices = torch.tensor(indices, dtype=torch.int).cuda(async=True)
        # vectors = torch.tensor(vectors, dtype=torch.float).cuda(async=True)

        best_indices, best_distances = find_landmarks(test_vectors, best_indices,
                                                      best_distances, indices, vectors)

    print("saving results to", RETRIEVAL_DISTANCES_FILE)
    best_indices = best_indices.cpu().numpy()
    best_distances = best_distances.cpu().numpy()
    best_indices = np.vectorize(lambda i: all_landmark_names[i])(best_indices)
    np.savez(RETRIEVAL_DISTANCES_FILE, indices=best_indices, distances=best_distances)
