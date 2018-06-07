import torch
import torchvision
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import scipy.sparse as sp
import matplotlib.pyplot as plt
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

    n = 1
    plt.figure(figsize=(4.2, 4))
    for i in range(n):
        #d = np.random.randint(0, len(train_set))
        d = 0
        X0 = train_set[d][0].squeeze(0)
        plt.subplot(2, n, 2 * i + 1)
        plt.imshow(X0, cmap=plt.cm.gray_r,interpolation='nearest')
        plt.subplot(2, n, 2 * i + 2)
        plt.imshow(smp.expectation(X0), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        print("SMP: " + str(i) + " images generated.")
    plt.suptitle('Regenerated numbers', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
