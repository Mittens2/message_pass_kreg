import torch
import math
from torch.utils.data.sampler import Sampler
import timeit
import os
import matplotlib.pyplot as plt
from enum import Enum

FIGS_DIR = 'figs'

class GType(Enum):
    KR = 1
    ER = 2

def cycles_expected(n, c, gtype):
    if gtype == GType.ER:
        f = lambda i: math.pow(2 * math.log(i), i) / (2 * i)
    else:
        f = lambda i: math.pow(2, i) / (2 * i)
    return sum(map(f, range(2, c + 1)))

def logistic(x):
    return 1. / (1. + torch.exp(-x))

def wrapper(func, *args):
    def wrapped():
        return func(*args)
    return wrapped

def time(func, *args):
    wrapped = wrapper(func, *args)
    return timeit.timeit(wrapped, number=1)

def savefig(fname, verbose=True):
    path = os.path.join('.', FIGS_DIR, fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))

class subSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return iter(range((torch.nonzero(self.mask).shape[0])))

    def __len__(self):
        return len(self.mask)
