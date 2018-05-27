#!/usr/bin/python3.6
""" Calculates a feature vector for every image from both train and test sets. """
__version__ = '0.3.17'

import os
import os.path as osp

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import set_grad_enabled
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#import argparse
import visdom
import logging
import numpy as np
import random
import time
import datetime
import pprint
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from world import cfg, create_logger, AverageMeter, accuracy



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cudnn.benchmark = True

timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'densenet169'
opt.MODEL.PRETRAINED = True
# opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 224 # crop size

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = 'feature_extractor'
opt.EXPERIMENT.TASK = 'test'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

opt.TEST = edict()
opt.TEST.CHECKPOINT = 'experiments/feature_extractor.pk'
opt.TEST.WORKERS = 12
opt.TEST.BATCH_SIZE = 128
opt.TEST.OUTPUT = osp.join(opt.EXPERIMENT.DIR, 'features_%s_%d.npz')

opt.DATASET = 'recognition'

opt.VISDOM = edict()
opt.VISDOM.PORT = 8097
opt.VISDOM.ENV = '[' + opt.DATASET + ']' + opt.EXPERIMENT.CODENAME




if not osp.exists(opt.EXPERIMENT.DIR):
    os.makedirs(opt.EXPERIMENT.DIR)




logger = create_logger(opt.LOG.LOG_FILE)
logger.info('\n\nOptions:')
logger.info(pprint.pformat(opt))


DATA_INFO = cfg.DATASETS[opt.DATASET.upper()]

# Data-loader of testing set
transform_test = transforms.Compose([
    # transforms.Resize((opt.MODEL.IMAGE_SIZE)),
    transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

# create model
if opt.MODEL.PRETRAINED:
    logger.info("=> using pre-trained model '{}'".format(opt.MODEL.ARCH ))
    model = models.__dict__[opt.MODEL.ARCH](pretrained=True)
else:
    raise NotImplementedError


if opt.MODEL.ARCH.startswith('resnet'):
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    model.fc = nn.Linear(model.fc.in_features, DATA_INFO.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
elif opt.MODEL.ARCH.startswith('densenet'):
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    model.classifier = nn.Linear(model.classifier.in_features, DATA_INFO.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
else:
    raise NotImplementedError

last_checkpoint = torch.load(opt.TEST.CHECKPOINT)
assert(last_checkpoint['arch']==opt.MODEL.ARCH)
model.module.load_state_dict(last_checkpoint['state_dict'])
logger.info("Checkpoint '{}' was loaded.".format(opt.TEST.CHECKPOINT))

extractor = nn.Sequential(model.module.features).cuda()

# vis = visdom.Visdom(port=opt.VISDOM.PORT)
# vis.close()
# vis.text('HELLO', win=0, env=opt.VISDOM.ENV)

model.eval()
extractor.eval()
set_grad_enabled(False)

def extract_features(dataset, name):
    # Don't create files with size 2048*1M floats = 8Gb, let's split them into groups.
    group_max_size = 125000
    max_batches = group_max_size // opt.TEST.BATCH_SIZE
    images = [osp.basename(image) for image, _ in dataset.imgs]

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.TEST.BATCH_SIZE, shuffle=False, num_workers=opt.TEST.WORKERS)

    group = []
    base = 0
    group_idx = 0

    for i, (input, target) in enumerate(tqdm(data_loader)):
        # compute output
        # print("got input: ", input.shape)
        output = extractor(input.cuda())

        features = output.data
        features = F.relu(features, inplace=True)
        features = F.avg_pool2d(features, kernel_size=7, stride=1).view(features.size(0), -1)
        features = features.cpu().numpy()

        # print("got features: ", features.shape)
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features /= norm
        group.append(features)

        if len(group) >= max_batches or i + 1 >= len(data_loader):
            features = np.concatenate(group)
            print("dumping features: ", features.shape)

            images_subset = images[base * opt.TEST.BATCH_SIZE : (i+1)*opt.TEST.BATCH_SIZE]
            filename = opt.TEST.OUTPUT % (name, group_idx)
            np.savez(filename, images=images_subset, features=features)
            logger.info("results were saved to " + filename)

            group = []
            base = i + 1
            group_idx += 1

categories_dataset = datasets.ImageFolder(DATA_INFO.TRAIN_DIR, transform_test)
logger.info('{} images are found for categories'.format(len(categories_dataset.imgs)))
extract_features(categories_dataset, "train")

queries_dataset = datasets.ImageFolder(DATA_INFO.TEST_DIR, transform_test)
logger.info('{} images are found for queries'.format(len(queries_dataset.imgs)))
extract_features(queries_dataset, "test")
