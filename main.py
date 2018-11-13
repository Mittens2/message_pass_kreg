import torch
import torchvision
import os
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from message_pass import SparseMP
from itertools import chain
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
    gtype = GType.KR
    numbers = [0, 1]
    n, k = int(5e4), 5
    load = False
    train = True
    print("%s graph with %d nodes" % (gtype.name, n))

    # Initialize model
    lr, eps, th, epochs, batch_size, max_iters = 1.0, 1e-6, 5.0, 200, 1, 200
    if load:
        model = SparseMP(gtype=gtype, dims = (n, k), load=True, numbers=numbers,
            lr=lr, eps=eps, th=th, epochs=epochs, batch_size=batch_size, max_iters=max_iters, device=device)
    else:
        model = SparseMP(gtype=gtype, dims = (n, k), load=False, numbers=numbers,
            lr=lr, eps=eps, th=th, epochs=epochs, batch_size=batch_size, max_iters=max_iters, device=device)

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
    # m = 3
    # samples = 100000
    # X0, _ = train_set[0]
    # X0 = X0.squeeze()
    # x = torch.round(torch.rand(n, device=device))
    # plt.figure(figsize=(4.2, 4))
    # for i in range(1, m ** 2 + 1):
    #     plt.subplot(m, m, i)
    #     x = model.gibbs(samples)
    #     plt.imshow(x[:X0.view(-1).shape[0]].view(X0.shape[0], -1), cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.xticks(())
    #     plt.yticks(())
    #     print("SMP: " + str(i) + " images generated.")
    # plt.suptitle('Regenerated numbers', fontsize=16)
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    # savefig("(%d, %d, %d)gibbs" % (len(numbers), n, k), gtype)
