import torch
import torchvision
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from message_pass import SparseMP
from random import random


if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    # Generate k-regular graph
    n, k = 2000, 3
    G = nx.random_regular_graph(k, n, seed=42)
    sparse_adj = nx.adjacency_matrix(G)
    adj = sparse_adj.todense()
    _, col = sparse_adj.nonzero()
    adj_list = col.reshape(n, -1)

    # transforms.Normalize((0.5,), (1.0,))
    trans = transforms.Compose([transforms.ToTensor()])
    # if data does not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    model = SparseMP(adj=adj, adj_list=adj_list)
    model.train(train_set=train_set)

    #Generate n ** 2 samples from graphical model
    n = 1
    plt.figure(figsize=(4.2, 4))
    X0 = train_set[0][0].squeeze(0)
    print(train_set[0][1])
    for i in range(1, n ** 2 + 1):
        plt.imshow(model.gibbs(X0, 100), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.subplot(n, n, i)
        plt.xticks(())
        plt.yticks(())
        print("SMP: " + str(i) + " images generated.")
    plt.suptitle('Regenerated numbers', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
