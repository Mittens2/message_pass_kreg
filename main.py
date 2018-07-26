import torch
import torchvision
import os
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import igraph
import numpy as np
import scipy.sparse as sp
from message_pass import SparseMP
from random import random
from utils import savefig

if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    # Generate k-regular graph
    n, k = 100000, 5
    G = nx.random_regular_graph(k, n, seed=42)
    #G = nx.fast_gnp_random_graph(n=n, p=5e-4)
    #G = igraph.Graph.Erdos_Renyi(n=n, p=5e-4)
    #G = igraph.Graph.K_Regular(n, k)
    # min, max = 3, 10
    # upper = 10
    # y = np.zeros((max - min + 1) // 2)
    # x = np.arange(min, max, 2)
    # for k in x:
    #     G = nx.random_regular_graph(k, n, seed=42)
    #     lengths = np.fromiter((map(lambda x: len(x), nx.cycle_basis(G))), dtype=np.int)
    #     _, counts = np.unique(lengths, return_counts=True)
    #     y[(k - min) // 2] = np.sum(counts[:upper + 1])
    #     print(y)
    # plt.plot(x, y, '-o')
    # plt.title("Cycles in graph of %d nodes" % (n))
    # plt.xlabel("Degree (k)")
    # plt.ylabel("Cycles of length <=%d" % (upper))
    # savefig("cycles_%d.png" % (n))
    # plt.show()
    sparse_adj = nx.adjacency_matrix(G)
    _, col = sparse_adj.nonzero()
    adj_list = torch.from_numpy(col.reshape(n, -1)).type(torch.LongTensor)
    if torch.cuda.is_available():
        adj_list = adj_list.cuda()
    adj = torch.zeros(n, k, device=device)
    local = torch.rand(n, device=device) - 0.5

    trans = transforms.Compose([transforms.ToTensor()])
    # if data does not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    model = SparseMP(adj=adj, local=local, adj_list=adj_list, eps=1e-6, lr=0.05, epochs=100, batch_size=1, max_iters=10, device=device)
    model.train(train_set=train_set)

    # Generate n samples from graphical model
    m = 3
    X0, _ = train_set[0]
    X0 = X0.squeeze(0)
    plt.figure(figsize=(4.2, 4))
    for i in range(1, m ** 2 + 1):
        plt.subplot(m, m, i)
        plt.imshow(model.gibbs(X0, 500), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        print("SMP: " + str(i) + " images generated.")
    plt.suptitle('Regenerated numbers', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    savefig("gibbs_%d_%d.png" % (n, k))
