import torch
import timeit
import os
import matplotlib.pyplot as plt

FIGS_DIR = 'figs'

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
