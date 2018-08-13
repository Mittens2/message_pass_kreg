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
from functools import reduce
from random import random
from utils import *

if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    # Generate k-regular graph
    gtype = GType.KR
    #gtype = GType.ER
    n, k = int(5e6), 5
    cyc = 16
    print("     expected cycles of length <=%d per node: %.5f" % (cyc, cycles_expected(n, cyc, gtype) / n))
    G = nx.random_regular_graph(k, n, seed=42)
    # print("Graph connected? %r" % (nx.is_connected(G)))
    # G = nx.fast_gnp_random_graph(n=n, p=2*math.log(n)/2, seed=42)

    #G = igraph.Graph.Erdos_Renyi(n=n, p=2*math.log(n)/n)
    # print("Graph connected? %r" % (G.is_connected()))
    # G = igraph.Graph.K_Regular(n, k)
    # min, max = 3, 10
    # upper = 10
    # y = np.zeros((max - min + 1) // 2)
    # x = np.arange(min, max, 2)
    # for k in x:
    #     G = nx.fast_gnp_random_graph(n=n, p=k/n, seed=42)
    #     lengths = np.fromiter((map(lambda x: len(x), nx.cycle_basis(G))), dtype=np.int)
    #     _, counts = np.unique(lengths, return_counts=True)
    #     y[(k - min) // 2] = np.sum(counts[:upper + 1])
    #     print(y)
    # plt.plot(x, y, '-o')
    # plt.title("Cycles in graph of %d nodes" % (n))
    # plt.xlabel("Degree (k)")
    # plt.ylabel("Cycles of length <=%d" % (upper))
    # savefig("cycles_%d_ER.png" % (n))
    # plt.show()
    lr, eps, th, epochs, batch_size, max_iters = 0.1, 1e-10, 0.4, 1000, 10, 20
    sparse_adj = nx.adjacency_matrix(G)
    row, col = sparse_adj.nonzero()
    cr = dict(((row[x], col[x]), x) for x in range(col.shape[0]))
    r2c = np.array([cr[x] for x in list(zip(col, row))])
    r2c = torch.from_numpy(r2c).type(torch.LongTensor)
    row = torch.from_numpy(row).type(torch.LongTensor)
    col = torch.from_numpy(col).type(torch.LongTensor)
    # adj_list= G.get_adjlist()
    # col = torch.LongTensor([item for sublist in adj_list for item in sublist])
    # row_list = list(map(lambda x: x[0] * [x[1]], zip(list(map(lambda x: len(x), adj_list)), range(n))))
    # row = torch.LongTensor([item for sublist in row_list for item in sublist])
    if torch.cuda.is_available():
        row = row.cuda()
        col = col.cuda()
    adj = torch.zeros(row.shape[0], device=device)
    local = torch.rand(n, device=device) * th - th / 2

    trans = transforms.Compose([transforms.ToTensor()])
    # if data does not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    mask = (train_set.train_labels == 5) | (train_set.train_labels == 3)
    sampler = subSampler(mask)
    model = SparseMP(adj=adj, local=local, row=row, col=col, r2c=r2c, lr=lr, eps=eps, th=th, epochs=epochs, batch_size=batch_size, max_iters=max_iters, device=device)
    model.train(train_set=train_set, sampler=sampler)

    # Generate n samples from graphical model
    m = 3
    X0, _ = train_set[0]
    X0 = X0.squeeze(0)
    plt.figure(figsize=(4.2, 4))
    for i in range(1, m ** 2 + 1):
        plt.subplot(m, m, i)
        plt.imshow(model.gibbs(X0, 100), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        print("SMP: " + str(i) + " images generated.")
    plt.suptitle('Regenerated numbers', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    savefig("gibbs_er_%d_%d_%.2f_%d_%d.png" % (n, k, lr, epochs, batch_size))
