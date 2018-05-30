import torch
import torchvision
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import scipy.sparse as sp
import networkx as nx
import numpy as np
from message_pass import SparseMP
from random import random


if __name__ == "__main__":
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if data does not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    n, k = 10000, 10
    G = nx.random_regular_graph(k, n, seed=42)
    sparse_adj = nx.adjacency_matrix(G)
    adj = sparse_adj.todense()
    _, col = sparse_adj.nonzero()
    adj_list = col.reshape(n, -1)
    smp = SparseMP(train_set=train_set, adj=adj, adj_list=adj_list)
