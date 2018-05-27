#!/usr/bin/python3.6
""" For every image from test set, searches for the top-100 closes landmarks. """

from typing import *
from glob import glob
import os, os.path as osp, pprint
from easydict import EasyDict as edict          # type: ignore
from world import cfg, create_logger
import numpy as np                              # type: ignore
import torch                                    # type: ignore
from torch.autograd import set_grad_enabled     # type: ignore
from tqdm import tqdm                           # type: ignore

PTArray = Any
opt = edict()

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = 'retrieval'
opt.EXPERIMENT.TASK = 'retrieval'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)
opt.EXPERIMENT.OUTPUT = osp.join(opt.EXPERIMENT.DIR, 'distances.npz')

opt.FEATURES = edict()
opt.FEATURES.CODENAME = 'feature_extractor'
opt.FEATURES.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.FEATURES.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

TOPK = 100

def find_landmarks(test_vectors: PTArray, best_indices: PTArray, best_distances: PTArray,
                   db_indices: PTArray, db_vectors: PTArray) -> Tuple[PTArray, PTArray]:
    """ Searches for the top-100 nearest landmarks for every test image. """
    print("test_vectors", test_vectors.shape)
    print("best_indices", best_indices.shape, "best_distances", best_distances.shape)
    print("db_indices", db_indices.shape, "db_vectors", db_vectors.shape)

    for base in tqdm(range(0, db_indices.shape[0], TOPK)):
        # sizes (100, 115k) and (100, 1600), might be less then 100 in the end of array
        indices = db_indices[base: base + TOPK]
        expanded_shape = (indices.shape[0], best_distances.shape[1])
        indices = db_indices[base: base + TOPK].unsqueeze(-1).expand(expanded_shape)
        vectors = db_vectors[base: base + TOPK, :]
        # print("\nindices, vectors", indices.shape, vectors.shape)

        # size (100, 115k)
        new_distances = torch.mm(vectors, test_vectors)
        # print(new_distances.shape)

        # both sizes are (200, 115k)
        joint_distances = torch.cat((best_distances, new_distances))
        joint_indices = torch.cat((best_indices, indices))
        # print(joint_distances.shape, joint_indices.shape)

        # both sizes are (100, 115k)
        best_distances, indices = torch.topk(joint_distances, k=TOPK, dim=0, largest=False)
        # print(best_distances.shape, indices.shape)

        # size (100, 115k)
        best_indices = torch.gather(joint_indices, dim=0, index=indices)
        # print(best_indices.shape)

    return best_indices, best_distances

def process_all_files() -> None:
    """ Iterates through all train files. """
    test_data = np.load(osp.join(opt.FEATURES.DIR, "features_test_0.npz"))
    test_names, test_vectors = test_data["images"], test_data["features"]

    print(test_names)
    print(test_vectors)
    print("first vector:")
    print("length", test_vectors[0].size, "non-zeros", np.count_nonzero(test_vectors[0]))

    # note transpose
    test_vectors = torch.tensor(test_vectors.T, dtype=torch.float).cuda(async=True)

    print("test_names:", test_names.shape)
    print("test_vectors:", test_vectors.shape)

    all_landmark_names: List[str] = []
    num_test_samples = test_vectors.shape[1]
    best_indices = torch.zeros(TOPK, num_test_samples, dtype=torch.int).cuda(async=True)
    best_distances = torch.ones(TOPK, num_test_samples, dtype=torch.float).cuda(async=True)

    # test
    for fragment in glob(osp.join(opt.FEATURES.DIR, "features_train_*.npz")):
        print("\n\nmatching against ", fragment)
        landmark_data = np.load(fragment)
        names, vectors = landmark_data["images"], landmark_data["features"]

        indices = np.arange(len(all_landmark_names), len(all_landmark_names) + len(names))
        indices = torch.tensor(indices, dtype=torch.int).cuda(async=True)
        all_landmark_names.extend(list(names))
        vectors = torch.tensor(vectors, dtype=torch.float).cuda(async=True)

        best_indices, best_distances = find_landmarks(test_vectors, best_indices,
                                                      best_distances, indices, vectors)

    print("saving results to", opt.EXPERIMENT.OUTPUT)
    best_indices = best_indices.cpu().numpy()
    best_distances = best_distances.cpu().numpy()
    best_indices = np.vectorize(lambda i: all_landmark_names[i])(best_indices)
    np.savez(opt.EXPERIMENT.OUTPUT, indices=best_indices, distances=best_distances)

if __name__ == "__main__":
    if not osp.exists(opt.EXPERIMENT.DIR):
        os.makedirs(opt.EXPERIMENT.DIR)

    logger = create_logger(opt.LOG.LOG_FILE)    # type: ignore
    logger.info('\n\nOptions:')
    logger.info(pprint.pformat(opt))

    set_grad_enabled(False)
    process_all_files()
