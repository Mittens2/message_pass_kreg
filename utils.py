import torch
import numpy as np
import math
import random
from torch.utils.data.sampler import Sampler
import timeit
import os
import matplotlib.pyplot as plt
from enum import Enum

FIGS_DIR = 'figs'

class GType(Enum):
    KR = 1
    ER = 2

def plotCycles(max_k, max_cyc):
    y = np.zeros(max_k - 3)
    x = np.arange(min, max)
    for k in x:
        G = nx.fast_gnp_random_graph(n=n, p=k/n, seed=42)
        lengths = np.fromiter((map(lambda x: len(x), nx.cycle_basis(G))), dtype=np.int)
        _, counts = np.unique(lengths, return_counts=True)
        y[k - min] = np.sum(counts[:max_cyc])
        print(y)
    plt.plot(x, y, '-o')
    plt.title("Cycles in graph of %d nodes" % (n))
    plt.xlabel("Degree (k)")
    plt.ylabel("Cycles of length <=%d" % (max_cyc))
    savefig("cycles_%d_ER.png" % (n))

def cycles_expected(n, k, c, gtype):
    if gtype == GType.ER:
        f = lambda i: math.pow(math.log(n), i) / (2 * i)
    else:
        f = lambda i: math.pow(k - 1, i) / (2 * i)
    return sum(map(f, range(2, c + 1))) / n

def logistic(x):
    return 1. / (1. + torch.exp(-x))

def wrapper(func, *args):
    def wrapped():
        return func(*args)
    return wrapped

def time(func, *args):
    wrapped = wrapper(func, *args)
    return timeit.timeit(wrapped, number=1)

def savefig(fname, gtype, verbose=True):
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
