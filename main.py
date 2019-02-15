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
import argparse
from ex_utils import *
from utils import *

MODEL_DIR = 'data/model/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Script Parameters
    parser.add_argument("-l", "--load", action="store_false")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-x", "--exact", action="store_true")
    parser.add_argument("-s", "--save", action="store_false")
    parser.add_argument("-d", '--numbers', nargs='+', type=int, default=[0])
    # Graph Parameters
    parser.add_argument("-g", "--graph_type", type=int, default=2)
    parser.add_argument("-n", "--nodes", type=int, default=int(1e4))
    # Hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-ld", "--lr_decay", type=float, default=0.1)
    parser.add_argument("-ep", "--epsilon", type=float, default=1e-4)
    parser.add_argument("-mi", "--max_iters", type=int, default=200)
    parser.add_argument("-th", "--threshold", type=float, default=10.0)
    parser.add_argument("-da", "--damping", type=float, default=0.5)
    args, _ = parser.parse_known_args()
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    print("%s graph with %d nodes" % (GType(args.graph_type).name, args.nodes))

    # Initialize model
    model = SparseMP(gtype=GType(args.graph_type), n=args.nodes, load=args.load, numbers=args.numbers,
        lr=args.learning_rate, lr_decay=args.lr_decay, damping=args.damping, eps=args.epsilon, th=args.threshold, max_iters=args.max_iters,
        device=device)

    # Train model with sub-sampler
    if args.train:
        mask = torch.ones(len(train_set.train_labels), dtype=torch.uint8)
        args.numbers = [int(x) for x in args.numbers]
        sampler = subSampler(args.numbers, train_set)
        pseudo_trend = model.train(train_set=train_set, epochs=args.epochs, batch_size=args.batch_size, save=args.save, sampler=sampler)
        # Plot pseudo likelihood
        plt.plot(np.arange(0, len(pseudo_trend)), pseudo_trend)
        plt.xlabel('iter')
        plt.ylabel('bethe free energy')
        title = '(%d, %d) bethe' % (len(args.numbers), args.nodes)
        plt.title(title)
        plt.legend()
        savefig(title, GType(args.graph_type))
        plt.show()
    # Generate m samples from model
    if args.plot:
        m = 4
        samples = 100000
        X0, _ = train_set[0]
        X0 = X0.squeeze()
        plt.figure(figsize=(4.2, 4))
        # x = torch.round(torch.rand(args.nodes, device=device))
        for i in range(1, m ** 2 + 1):
            plt.subplot(m, m, i)
            x = model.gibbs(samples)
            plt.imshow(x[:X0.view(-1).shape[0]].view(X0.shape[0], -1), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
            print("SMP: " + str(i) + " images generated.")
        plt.suptitle('Regenerated numbers', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        savefig("(%d, %d) gibbs" % (len(args.numbers), args.nodes), GType(args.graph_type))

    if args.exact:
        # Get marginals using message passing
        # Non-zero entries of adjacency matrix
        row = model.row.numpy()
        col = model.col.numpy()
        adj = np.zeros((args.nodes, args.nodes))
        print(row)
        print(col)
        adj[row, col] = 1
        cti = CliqueTreeInference(adj, verbosity=1)
        marg_mp = model.free_mp()
        marg_ex = cti.get_marginal(list(range(0, n)))
        print(marg_mp)
        print(marg_ex)
