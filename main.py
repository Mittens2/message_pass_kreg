import torch
import torchvision
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from message_pass import SparseMP
from random import random


if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    # Generate k-regular graph
    n, k = 20000, 10
    G = nx.random_regular_graph(k, n, seed=42)
    sparse_adj = nx.adjacency_matrix(G)
    _, col = sparse_adj.nonzero()
    adj_list = torch.from_numpy(col.reshape(n, -1)).type(torch.LongTensor)
    adj = torch.ones(n, k) * 0.5
    local = torch.rand(n) - 0.5

    trans = transforms.Compose([transforms.ToTensor()])
    # if data does not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    model = SparseMP(adj=adj, local=local, adj_list=adj_list, lr=0.01, epochs=100, batch_size=1, max_iters=1)
    model.train(train_set=train_set)

    # Generate n samples from graphical model
    n = 2
    X0, label = train_set[7]
    X0 = X0.squeeze(0)
    plt.figure(figsize=(4.2, 4))
    for i in range(1, n ** 2 + 1):
        plt.subplot(n, n, i)
        plt.imshow(model.gibbs(X0, 500), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        print("SMP: " + str(i) + " images generated.")
    plt.suptitle('Regenerated numbers', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
