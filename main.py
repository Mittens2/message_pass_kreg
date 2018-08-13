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
from itertools import chain
from utils import *

if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    # Generate random graph
    gtype = GType.KR
    n, k = int(5e6), 5
    print("%s graph with %d nodes" % (gtype.name, n))
    lr, eps, th, epochs, batch_size, max_iters = 0.1, 1e-10, 0.4, 1000, 10, 20
    cyc = round(math.log(eps) / (math.log(th) + th - math.log(math.exp(th) + 1)))
    print("     expected cycles of length <=%d per node: %.5f" % (cyc, cycles_expected(n, cyc, gtype) / n))

    # Use networkx for k-regular
    G = nx.random_regular_graph(k, n)
    print("Graph connected? %r" % (nx.is_connected(G)))
    sparse_adj = nx.adjacency_matrix(G)
    row, col = sparse_adj.nonzero()
    cr = dict(map(lambda x: ((row[x], col[x]), x), range(col.shape[0])))
    r2c = torch.LongTensor(list(map(lambda x: cr[x], list(zip(col, row)))))
    row = torch.from_numpy(row).type(torch.LongTensor)
    col = torch.from_numpy(col).type(torch.LongTensor)

    # Use igraph for erdos-renyi
    # G = igraph.Graph.Erdos_Renyi(n=n, p=2*math.log(n)/n)
    # print("     Graph connected? %r" % (G.is_connected()))
    # adj_list= G.get_adjlist()
    # col_list = [item for sublist in adj_list for item in sublist]
    # row_list = list(chain.from_iterable(map(lambda x: x[0] * [x[1]], zip(list(map(lambda x: len(x), adj_list)), range(n)))))
    # cr = dict(map(lambda x: ((row_list[x], col_list[x]), x), range(len(col_list))))
    # r2c = torch.LongTensor([cr[x] for x in list(zip(col_list, row_list))])
    # r2c = torch.LongTensor(list(map(lambda x: cr[x], list(zip(col_list, row_list)))))
    # row = torch.LongTensor(row_list)
    # col = torch.LongTensor(col_list)

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
