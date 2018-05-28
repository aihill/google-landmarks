#!/usr/bin/python3.6
""" Generates predictions for 3B with the best of 5 crops. """

import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import set_grad_enabled
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import visdom, logging, numpy as np, random, time, datetime, pprint
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

from world import cfg, create_logger, AverageMeter, accuracy


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cudnn.benchmark = True

timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan', precision=8,
                    suppress=False, threshold=1000, formatter=None)

opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'densenet169'
opt.MODEL.PRETRAINED = True
opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 224 # crop size

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = '3B_crop'
opt.EXPERIMENT.TASK = 'test'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

opt.TEST = edict()
opt.TEST.CHECKPOINT = osp.join(opt.EXPERIMENT.DIR, 'best_model.pk')
opt.TEST.WORKERS = 12
opt.TEST.BATCH_SIZE = 32
opt.TEST.OUTPUT = osp.join(opt.EXPERIMENT.DIR, 'pred.npz')

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
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

#train_dataset = datasets.ImageFolder(DATA_INFO.TRAIN_DIR, transform_test)
test_dataset = datasets.ImageFolder(DATA_INFO.TEST_DIR, transform_test)
logger.info('{} images are found for test'.format(len(test_dataset.imgs)))

test_list = pd.read_csv(osp.join(DATA_INFO.ROOT_DIR, 'test.csv'))
test_list = test_list['id']
logger.info('{} images are expected for test'.format(len(test_list)))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.TEST.BATCH_SIZE, shuffle=False, num_workers=opt.TEST.WORKERS)


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
    model = torch.nn.DataParallel(model).cuda()


last_checkpoint = torch.load(opt.TEST.CHECKPOINT)
assert(last_checkpoint['arch']==opt.MODEL.ARCH)
model.module.load_state_dict(last_checkpoint['state_dict'])
logger.info("Checkpoint '{}' was loaded.".format(opt.TEST.CHECKPOINT))

last_epoch = last_checkpoint['epoch']

# vis = visdom.Visdom(port=opt.VISDOM.PORT)
# vis.close()
# vis.text('HELLO', win=0, env=opt.VISDOM.ENV)

softmax = torch.nn.Softmax(dim=1).cuda()

pred_classes = []
pred_confs = []

model.eval()
set_grad_enabled(False)

K = 20

for i, (input, target) in enumerate(tqdm(test_loader)):
    target = target.cuda(async=True)

    crops = []
    start = (opt.MODEL.IMAGE_SIZE - opt.MODEL.INPUT_SIZE) // 2
    crops.append(input[:, :, start : start+opt.MODEL.INPUT_SIZE, start : start+opt.MODEL.INPUT_SIZE])
    crops.append(input[:, :, 0 : opt.MODEL.INPUT_SIZE, 0 : opt.MODEL.INPUT_SIZE])
    crops.append(input[:, :, opt.MODEL.IMAGE_SIZE - opt.MODEL.INPUT_SIZE :, 0 : opt.MODEL.INPUT_SIZE])
    crops.append(input[:, :, 0 : opt.MODEL.INPUT_SIZE, -opt.MODEL.INPUT_SIZE :])
    crops.append(input[:, :, opt.MODEL.IMAGE_SIZE - opt.MODEL.INPUT_SIZE :, opt.MODEL.IMAGE_SIZE - opt.MODEL.INPUT_SIZE :])

    best_classes = np.zeros((opt.TEST.BATCH_SIZE, K), dtype=int)
    best_confs = np.zeros((opt.TEST.BATCH_SIZE, K), dtype=float)
    best_conf = 0

    for crop in crops:
        # compute output
        output = model(crop)
        _, top_classes = torch.topk(output, k=K)
        top_classes = top_classes.data.cpu().numpy()

        confs = softmax(output)
        top_confs, _ = torch.topk(confs, k=K)
        top_confs = top_confs.data.cpu().numpy()

        # select classes where we are most confident
        for i in range(opt.TEST.BATCH_SIZE):
            if top_confs[i, 0] > best_confs[i, 0]:
                best_confs[i, :]    = top_confs[i, :]
                best_classes[i, :]  = top_classes[i, :]

    pred_classes.append(best_classes)
    pred_confs.append(best_confs)

pred_classes = np.concatenate(pred_classes)
pred_confs = np.concatenate(pred_confs)

images = [osp.basename(image) for image, _ in test_dataset.imgs]

np.savez(opt.TEST.OUTPUT, pred_indices=pred_classes,
         pred_confs=pred_confs, images=images, checkpoint=opt.TEST.CHECKPOINT)
logger.info("Results were saved to '{}'.".format(opt.TEST.OUTPUT))
