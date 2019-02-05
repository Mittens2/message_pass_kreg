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
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-x", "--exact", action="store_true")
    parser.add_argument("-s", "--save", action="store_false")
    parser.add_argument("-d", "--numbers", type=list, default=[0])
    # Graph Parameters
    parser.add_argument("-g", "--graph_type", type=GType, default=GType.ER)
    parser.add_argument("-n", "--nodes", type=int, default=int(1e4))
    parser.add_argument("-k", "--degree", type=int, default=10)
    # Hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-ld", "--lr_decay", type=float, default=0.1)
    parser.add_argument("-ep", "--epsilon", type=float, default=1e-3)
    parser.add_argument("-mi", "--max_iters", type=int, default=200)
    parser.add_argument("-th", "--threshold", type=float, default=10.0)
    args, _ = parser.parse_known_args()
    print(args.epsilon)
    # Set torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # Load data
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    print("%s graph with %d nodes" % (args.graph_type.name, args.nodes))

    # Initialize model
    model = SparseMP(gtype=args.graph_type, dims = (args.nodes, args.degree), load=args.load, numbers=args.numbers,
        lr=args.learning_rate, lr_decay=args.lr_decay, eps=args.epsilon, th=args.threshold, max_iters=args.max_iters,
        device=device)

    # Train model with sub-sampler
    if args.train:
        mask = torch.ones(len(train_set.train_labels), dtype=torch.uint8)
        sampler = subSampler(args.numbers, train_set)
        pseudo_trend = model.train(train_set=train_set, epochs=args.epochs, batch_size=args.batch_size, save=args.save, sampler=sampler)
        # Plot pseudo likelihood
        plt.plot(np.arange(0, len(pseudo_trend)), pseudo_trend)
        plt.xlabel('iter')
        plt.ylabel('bethe free energy')
        title = '(%d, %d, %d) Bethe trend' % (len(args.numbers), args.nodes, args.degree)
        plt.title(title)
        plt.legend()
        savefig(title, args.graph_type)
        plt.show()
    # Generate m samples from model
    if args.plot:
        m = 3
        samples = 20000
        X0, _ = train_set[0]
        X0 = X0.squeeze()
        x = torch.round(torch.rand(args.nodes, device=device))
        plt.figure(figsize=(4.2, 4))
        for i in range(1, m ** 2 + 1):
            plt.subplot(m, m, i)
            x = model.gibbs(samples, x)
            plt.imshow(x[:X0.view(-1).shape[0]].view(X0.shape[0], -1), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
            print("SMP: " + str(i) + " images generated.")
        plt.suptitle('Regenerated numbers', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        savefig("(%d, %d, %d)gibbs" % (len(args.numbers), args.nodes, args.degree), args.graph_type)

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
