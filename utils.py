import torch
import timeit

def logistic(x):
    return 1. / (1. + torch.exp(-x))

def wrapper(func, *args):
    def wrapped():
        return func(*args)
    return wrapped

def time(func, *args):
    wrapped = wrapper(func, *args)
    return timeit.timeit(wrapped, number=1)
