import torch
import torchvision
import os
import sys
sys.path.insert(0, './exact')
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from brute_force import BruteForceInference
from clique_tree import CliqueTreeInference
from message_pass import SparseMP
from itertools import chain
from ex_utils import *
from utils import *

MODEL_DIR = 'data/model/'

if __name__ == "__main__":
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)

    # These parameters should be taken in via command line
    gtype = GType.TR
    # numbers = list(range(10))
    numbers = [0]
    # n, k = int(1e4), 10
    n, k = 10, 2
    load, train, plot, exact = False, False, False, True

    print("%s graph with %d nodes" % (gtype.name, n))

    # Initialize model
    lr, lr_decay, eps, th, epochs, batch_size, max_iters = 0.2, 0.1, 1e-5, 5.0, 1, 3, 200
    if load:
        model = SparseMP(gtype=gtype, dims = (n, k), load=True, numbers=numbers,
            lr=lr, lr_decay=lr_decay, eps=eps, th=th, epochs=epochs, batch_size=batch_size, max_iters=max_iters, device=device)
    else:
        model = SparseMP(gtype=gtype, dims = (n, k), load=False, numbers=numbers,
            lr=lr, lr_decay=lr_decay, eps=eps, th=th, epochs=epochs, batch_size=batch_size, max_iters=max_iters, device=device)

    # Cycle statistics
    # print("     expected cycles of length <=%d per node: %.3f" % (cyc, cycles_expected(n, k, cyc, gtype)))

    # Train model with sub-sampler
    if train:
        mask = torch.ones(len(train_set.train_labels), dtype=torch.uint8)
        sampler = subSampler(numbers, train_set)
        pseudo_trend = model.train(train_set=train_set, save=True, sampler=sampler)
        # Plot pseudo likelihood
        plt.plot(np.arange(0, len(pseudo_trend)), pseudo_trend)
        plt.xlabel('epoch')
        plt.ylabel('pseudo likelihood')
        title = '(%d, %d, %d) Pseudo likelihood trend' % (len(numbers), n, k)
        plt.title(title)
        plt.legend()
        savefig(title, gtype)
        plt.show()
    # Generate m samples from model
    if plot:
        m = 2
        samples = 10000
        X0, _ = train_set[0]
        X0 = X0.squeeze()
        x = torch.round(torch.rand(n, device=device))
        plt.figure(figsize=(4.2, 4))
        for i in range(1, m ** 2 + 1):
            plt.subplot(m, m, i)
            # x = model.gibbs(samples)
            x = model.free_mp()
            plt.imshow(x[:X0.view(-1).shape[0]].view(X0.shape[0], -1), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
            print("SMP: " + str(i) + " images generated.")
        plt.suptitle('Regenerated numbers', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        savefig("(%d, %d, %d)gibbs" % (len(numbers), n, k), gtype)

    if exact:
        # Get marginals using message passing
        # Non-zero entries of adjacency matrix
        row = model.row.numpy()
        col = model.col.numpy()
        adj = np.zeros((n, n))
        print(row)
        print(col)
        adj[row, col] = 1
        cti = CliqueTreeInference(adj, verbosity=1)
        # bfi = BruteForceInference(adj, verbosity=2)
        marg_mp = model.free_mp()
        marg_ex = cti.get_marginal(list(range(0, n)))
        print(marg_mp)
        print(marg_ex)
        # Get marginals using exact inference
