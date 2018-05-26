import os
import os.path as osp
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.ROOT_DIR = osp.abspath(osp.dirname(__file__))
__C.EXPERIMENT_DIR = osp.join(__C.ROOT_DIR, 'experiments')
if not osp.exists(__C.EXPERIMENT_DIR):
    os.makedirs(__C.EXPERIMENT_DIR)


# Prepare the datasets ---------------------------------------------------
__C.DATASETS = edict()

__C.DATASETS.RECOGNITION = edict()
__C.DATASETS.RECOGNITION.ROOT_DIR = osp.join(__C.ROOT_DIR, 'recognition')
__C.DATASETS.RECOGNITION.TRAIN_DIR = osp.join(__C.DATASETS.RECOGNITION.ROOT_DIR, 'train')
__C.DATASETS.RECOGNITION.TEST_DIR = osp.join(__C.DATASETS.RECOGNITION.ROOT_DIR, 'test')
__C.DATASETS.RECOGNITION.NUM_CLASSES = 14951         # For assertion

__C.DATASETS.NO_CLASS = edict()
__C.DATASETS.NO_CLASS.ROOT_DIR = osp.join(__C.ROOT_DIR, 'no_class')
__C.DATASETS.NO_CLASS.TRAIN_DIR = osp.join(__C.DATASETS.NO_CLASS.ROOT_DIR, 'train')
__C.DATASETS.NO_CLASS.TEST_DIR = osp.join(__C.DATASETS.NO_CLASS.ROOT_DIR, 'test')
__C.DATASETS.NO_CLASS.NUM_CLASSES = 2         # For assertion

__C.DATASETS.RETRIEVAL = edict()
__C.DATASETS.RETRIEVAL.ROOT_DIR = osp.join(__C.ROOT_DIR, 'no_class')
__C.DATASETS.RETRIEVAL.TRAIN_DIR = osp.join(__C.DATASETS.RETRIEVAL.ROOT_DIR, 'train')
__C.DATASETS.RETRIEVAL.TEST_DIR = osp.join(__C.DATASETS.RETRIEVAL.ROOT_DIR, 'test')
# __C.DATASETS.RETRIEVAL.NUM_CLASSES = 2         # unknown



import logging

def create_logger(filename, logger_name='logger',
                  file_fmt='%(asctime)s %(levelname)-8s: %(message)s',
                  console_fmt='%(message)s',
                  file_level=logging.DEBUG, console_level=logging.DEBUG):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_fmt = logging.Formatter(file_fmt)
    log_file = logging.FileHandler(filename)
    log_file.setLevel(file_level)
    log_file.setFormatter(file_fmt)
    logger.addHandler(log_file)

    console_fmt = logging.Formatter(console_fmt)
    log_console = logging.StreamHandler()
    log_console.setLevel(logging.DEBUG)
    log_console.setFormatter(console_fmt)
    logger.addHandler(log_console)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
