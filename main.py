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

MODEL_DIR = 'data/model/'
TRAIN = True

if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor()])
    # if data does not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)

    # Generate random graph
    gtype = GType.ER
    n, k = int(1e6), 5
    print("%s graph with %d nodes" % (gtype.name, n))

    lr, eps, th, epochs, batch_size, max_iters = 0.08, 1e-6, 2.0, 100, 10, 100
    if TRAIN:
        # Use networkx for k-regular
        if gtype == GType.KR:
            G = nx.random_regular_graph(k, n)
            print("Graph connected? %r" % (nx.is_connected(G)))
            sparse_adj = nx.adjacency_matrix(G)
            row, col = sparse_adj.nonzero()
            cr = dict(map(lambda x: ((row[x], col[x]), x), range(col.shape[0])))
            r2c = torch.LongTensor(list(map(lambda x: cr[x], list(zip(col, row)))))
            row = torch.from_numpy(row).type(torch.LongTensor)
            col = torch.from_numpy(col).type(torch.LongTensor)
        else: # Use igraph for erdos-renyi
            G = igraph.Graph.Erdos_Renyi(n=n, p= 1.1 * math.log(n) / n)
            print("     Graph connected? %r" % (G.is_connected()))
            adj_list= G.get_adjlist()
            col_list = [item for sublist in adj_list for item in sublist]
            row_list = list(chain.from_iterable(map(lambda x: x[0] * [x[1]], zip(list(map(lambda x: len(x), adj_list)), range(n)))))
            cr = dict(map(lambda x: ((row_list[x], col_list[x]), x), range(len(col_list))))
            r2c = torch.LongTensor([cr[x] for x in list(zip(col_list, row_list))])
            r2c = torch.LongTensor(list(map(lambda x: cr[x], list(zip(col_list, row_list)))))
            row = torch.LongTensor(row_list)
            col = torch.LongTensor(col_list)

        # Initialize model
        if torch.cuda.is_available():
            row = row.cuda()
            col = col.cuda()
        adj = torch.zeros(row.shape[0], device=device)
        local = torch.rand(n, device=device) * th - th / 2
        model = SparseMP(adj=adj, local=local, row=row, col=col, r2c=r2c, lr=lr, eps=eps, th=th, epochs=epochs, batch_size=batch_size, max_iters=max_iters, device=device)

        # Cycle statistics
        # cyc = round(math.log(eps) / -math.log(math.exp(-th) + 1)) + 1
        cyc = 30
        print("     expected cycles of length <=%d per node: %.3f" % (cyc, cycles_expected(n, k, cyc, gtype)))

        # Train model with sub-sampler
        mask = (train_set.train_labels == 5) | (train_set.train_labels == 0)
        sampler = subSampler(mask)
        model.train(train_set=train_set, sampler=sampler, save=True)

    else: # Load parameters
        local, adj = torch.load(MODEL_DIR + "weights.pt")
        row, col, r2c = torch.load(MODEL_DIR + "adjacency.pt")
        model = SparseMP(adj=adj, local=local, row=row, col=col, r2c=r2c, device=device)

    # Generate n samples from graphical model
    m = 3
    X0, _ = train_set[0]
    X0 = X0.squeeze()
    x = torch.round(torch.rand(n, device=device))
    plt.figure(figsize=(4.2, 4))
    for i in range(1, m ** 2 + 1):
        plt.subplot(m, m, i)
        x = model.gibbs(100, 20, x)
        plt.imshow(x[:X0.view(-1).shape[0]].view(X0.shape[0], -1), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        print("SMP: " + str(i) + " images generated.")
    plt.suptitle('Regenerated numbers', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    fname = "kr/gibbs_*n=%d_k=%d_d=%d_lr=%.2f.png" % (n, k, epochs * batch_size, lr) if gtype == GType.KR else "er/gibbs_n=%d_d=%d_lr=%.2f.png" %  (n, epochs * batch_size, lr)
    savefig(fname, gtype)
