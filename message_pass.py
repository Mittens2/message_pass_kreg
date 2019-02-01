import os
import torch
import numpy as np
import math
import networkx as nx
import igraph
from itertools import chain
from utils import logistic, time, GType
from torch.utils.data import DataLoader

MODEL_DIR = 'data/model/'

class SparseMP():

    def __init__(self, gtype, dims, numbers, seed=42, load=False,
                 lr=0.1, damping=0.5, eps=1e-16, th=0.2, max_iters=10, lr_decay=2, device=torch.device("cpu")):
        # Model parameters
        n, k = dims
        torch.manual_seed(seed)
        self.gtype = gtype
        self.dims = dims
        self.numbers = len(numbers)
        self.lr_decay = lr_decay
        self.best_pseudo = -float('inf')
        self.device = device

        # Load model data
        if load:
            self.load_params()
        # Initialize model data from scratch
        else:
            if gtype == GType.ER: # Use igraph for erdos-renyi
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
            else: # Use nx for tree (exact inference test) or K-regular
                if gtype == GType.KR:
                    G = nx.random_regular_graph(k, n)
                else:
                    G = nx.star_graph(n=n-1)
                print("Graph connected? %r" % (nx.is_connected(G)))
                sparse_adj = nx.adjacency_matrix(G)
                row, col = sparse_adj.nonzero()
                cr = dict(map(lambda x: ((row[x], col[x]), x), range(col.shape[0])))
                r2c = torch.LongTensor(list(map(lambda x: cr[x], list(zip(col, row)))))
                row = torch.from_numpy(row).type(torch.LongTensor)
                col = torch.from_numpy(col).type(torch.LongTensor)

            if torch.cuda.is_available():
                row = row.cuda()
                col = col.cuda()
                r2c = r2c.cuda()

            # Normally distributed weights (look at RBM literature)
            adj = torch.zeros(row.shape[0], device=device)
            local = torch.zeros(n, device=device)


            self.adj = adj
            self.local = local
            self.row = row
            self.col = col
            self.r2c = r2c

        # Model Hyperparameters
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        self.eps = eps
        self.th = th

    def load_params(self):
        """Load model parameters
        """
        n, k = self.dims
        numbers = self.numbers
        if self.gtype == GType.ER:
            fn_weights = os.path.join(MODEL_DIR, 'ER', '%d_%d_weights.pt' % (numbers, n))
            fn_adjecency = os.path.join(MODEL_DIR, 'ER', '%d_%d_adjecency.pt' % (numbers, n))
        else:
            fn_weights = os.path.join(MODEL_DIR, 'KR', '%d_%d_%d_weights.pt' % (numbers, n, k))
            fn_adjecency = os.path.join(MODEL_DIR, 'KR', '%d_%d_%d_adjecency.pt' % (numbers, n, k))
        local, adj = torch.load(fn_weights)
        row, col, r2c = torch.load(fn_adjecency)
        if torch.cuda.is_available():
            local = local.cuda()
            adj = adj.cuda()
            row = row.cuda()
            col = col.cuda()
            r2c = r2c.cuda()

        self.adj = adj
        self.local = local
        self.row = row
        self.col = col
        self.r2c = r2c


    def save_params(self):
        """Save model parameters
        """
        n, k = self.dims
        numbers = self.numbers
        local = self.local
        adj = self.adj
        row = self.row
        col = self.col
        r2c = self.r2c
        if torch.cuda.is_available():
             local = local.to(torch.device("cpu"))
             adj = adj.to(torch.device("cpu"))
             row = row.to(torch.device("cpu"))
             col = col.to(torch.device("cpu"))
             r2c = r2c.to(torch.device("cpu"))

        if self.gtype == GType.ER:
            fn_weights = os.path.join(MODEL_DIR, 'ER', '%d_%d_weights.pt' % (numbers, n))
            fn_adjecency = os.path.join(MODEL_DIR, 'ER', '%d_%d_adjecency.pt' % (numbers, n))
        else:
            fn_weights = os.path.join(MODEL_DIR, 'KR', '%d_%d_%d_weights.pt' % (numbers, n, k))
            fn_adjecency = os.path.join(MODEL_DIR, 'KR', '%d_%d_%d_adjecency.pt' % (numbers, n, k))

        torch.save((local, adj), fn_weights)
        torch.save((row, col, r2c), fn_adjecency)

    def train(self, train_set, epochs, batch_size, sampler=None, save=True):
        """ Perform training using message passing for graphical model.
        """
        if torch.cuda.is_available():
            num_workers = 4
        else:
            num_workers = 0
        train_loader = iter(DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers))
        i = 0
        pseudo_trend = []
        for i in range(epochs):
            pseudo_epoch = []
            print("epoch %d" %(i))
            for j, (data, label) in enumerate(train_loader):
            # for j in range(300):
                print("batch %d" %(j))
                # data, label = next(train_loader)
                data = data.squeeze()
                data = torch.round(torch.transpose(data.view(-1, data.shape[1] ** 2), 0, 1))
                if torch.cuda.is_available():
                    data = data.cuda()
                self.free_mp()
                self.clamp_mp(data)
                pseudo_batch = self.update(data)
                pseudo_epoch += [pseudo_batch]
                print("     epoch pseudo-like: %.3g \n" % (sum(pseudo_epoch) / (j + 1)))
                # if pseudo_batch > -55:
                #     break
            pseudo_like = sum(pseudo_epoch) / len(train_set)
            if pseudo_like > self.best_pseudo:
                self.best_pseudo = pseudo_like
            self.save_params()
            self.lr *= self.lr_decay
            pseudo_trend += pseudo_epoch
        print(self.local[:data.shape[1]])
        return pseudo_trend

    def free_mp(self):
        """ Unclamped message-passing for one mini-batch.
        """
        adj = self.adj
        local = self.local
        row = self.row
        col= self.col
        n = self.dims[0]
        m = adj.shape[0]
        
        # Initialize to zeros because using log ratio
        message_old = torch.zeros(m, device=self.device)
        message_new = torch.zeros(m, device=self.device)
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_new.clone()
            message = (torch.zeros(n, device=self.device).index_add_(0, row, message_old)[row] - message_old)[self.r2c]
            message_new = local[row]
            mask = message < 10
            message_new[mask != 1] += adj[mask != 1]
            message_new[mask] += torch.log(torch.exp(local[col][mask] + adj[mask] + message[mask]) + 1)
            message_new[mask] -= torch.log(torch.exp(local[col][mask] + message[mask]) + 1)
            # For some reason samples from model never explode
            # message_new += torch.log(torch.exp(local[col] + adj + message) + 1)
            # message_new -= torch.log(torch.exp(local[col] + message) + 1)
            # message_new = message_new * self.damping + message_old * (1 - self.damping)
            iters+=1
        print("     %d iterations until convergence, minimum difference %.3e" % (iters,  torch.max(torch.abs(message_new - message_old))))
        self.message_free = message_new
        return logistic(torch.zeros(n, device=self.device).index_add_(0, row, message_new) + local)

    def clamp_mp(self, data):
        """ Clamped message-passing for one mini-batch.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data for current epoch, used to clamp visible units.
        """
        clamp = data.shape[0]
        batch_size = data.shape[1]
        adj = self.adj.clone().unsqueeze(1).expand(-1, batch_size)
        local = self.local.unsqueeze(1).expand(-1, batch_size)
        row = self.row
        col = self.col
        n = local.shape[0]

        # Add interaction of clamped units to local biases based on data vectors
        local.index_add_(0, col[row < clamp], adj[row < clamp] * data[row[row < clamp]])
        # Set interaction of clamped units to 0
        adj[row < clamp] = 0
        adj[col < clamp] = 0

        message_old = self.message_free.unsqueeze(1).repeat(1, batch_size)
        message_new = message_old.clone()
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_new.clone()
            message = (torch.zeros((n, batch_size), device=self.device).index_add_(0, row, message_old)[row] - message_old)[self.r2c]
            message_new = local[row]
            mask = message < 10
            message_new[mask != 1] += adj[mask != 1]
            message_new[mask] += torch.log(torch.exp(local[col][mask] + adj[mask] + message[mask]) + 1)
            message_new[mask] -= torch.log(torch.exp(local[col][mask] + message[mask]) + 1)
            # Alternative way of ensuring no nan
            # message = torch.clamp(message, max=10)
            # message_new += torch.log(torch.exp(local[col] + adj + message) + 1)
            # message_new -= torch.log(torch.exp(local[col] + message) + 1)
            message_new = message_new * self.damping + message_old * (1 - self.damping)
            iters+=1
        print("     %d iterations until convergence, minimum difference %.3e" % (iters,  torch.max(torch.abs(message_new - message_old))))
        self.message_clamp = message_new

    def update(self, data):
        """ Update of model parameters.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data for current epoch, needed for updating gradient.
        """
        adj = self.adj
        local = self.local
        row = self.row
        col = self.col
        r2c = self.r2c
        clamp = data.shape[0]
        batch_size = data.shape[1]
        message_free = self.message_free
        message_clamp = self.message_clamp
        n = self.dims[0]

        #Old without external
        msg_free = self.message_free
        sum_free = (torch.zeros(n, device=self.device).index_add_(0, row, msg_free)[row] - msg_free)
        p_ij_marg = 1 / (torch.exp(-(sum_free + sum_free[r2c] + local[row] + local[col] + adj)) \
        + torch.exp(-(sum_free + local[row] + adj)) + torch.exp(-(sum_free[r2c] + local[col] + adj)) + 1)

        #Calculate conditional joint probability
        msg_clamp = self.message_clamp
        adj = self.adj.unsqueeze(1).expand(-1, batch_size)
        local = local.unsqueeze(1).expand(-1, batch_size)
        # Add unit that sends signal of +-100 based on data
        msg_sum = torch.zeros((n, batch_size), device=self.device).index_add_(0, row, msg_clamp)
        msg_sum[:clamp] += (data * 2 - 1) * 10
        sum_clamp = msg_sum[row] - msg_clamp
        p_ij_cond = 1 / (torch.exp(-(sum_clamp + sum_clamp[r2c] + local[row] + local[col] + adj)) \
        + torch.exp(-(sum_clamp + local[row] + adj)) + torch.exp(-(sum_clamp[r2c] + local[col] + adj)) + 1)

        #Calculate marginal and condtional probabilities
        p_i_marg = logistic(torch.zeros(n, device=self.device).index_add_(0, row, message_free) + self.local)
        p_i_cond = logistic(torch.zeros((n, batch_size), device=self.device).index_add_(0, row, message_clamp) + local)

        # Average over batchs
        p_ij_cond = torch.sum(p_ij_cond, 1) / batch_size
        p_i_cond = torch.sum(p_i_cond, 1) / batch_size

        # Calcluate pseudo_likelihood
        exp_arg = torch.zeros(n, device=self.device).index_add_(0, row, self.adj) + self.local
        logZ = torch.log(1 + torch.exp(exp_arg))
        pos = (p_i_cond * self.local).index_add_(0, row, p_ij_cond * self.adj)
        pseudo_like = torch.sum((pos - logZ)[:clamp], 0)

        #Update model parameters
        th = self.th
        self.adj = torch.clamp(self.adj + self.lr * (p_ij_cond / batch_size - p_ij_marg), max=th)
        self.local = torch.clamp(self.local + self.lr * (p_i_cond - p_i_marg), min=-th, max=th)
        print("     avg weight: %.3g" % (torch.sum(self.adj) / self.adj.shape[0]))
        print("     max weight: %.3g" % (torch.max(self.adj)))
        print("     min weight: %.3g" % (torch.min(self.adj)))
        print("     avg bias: %.3g" % (torch.sum(self.local) / self.local.shape[0]))
        print("     max bias: %.3g" % (torch.max(self.local)))
        print("     min bias: %.3g" % (torch.min(self.local)))
        print("     batch pseudo-like: %.3g" % pseudo_like)

        return pseudo_like

    def gibbs(self, iters, x=None):
        """ Perform iters iterations of gibbs sampling, and return result.
        Parameters
        ----------
        x : array-like, shape (n)
            Sample from model
        """
        adj = self.adj
        local = self.local
        row = self.row
        col = self.col
        n = self.dims[0]
        # Initialize units to random weights
        if x is None:
            x = torch.round(torch.rand(n, device=self.device))
        # Sample iter times
        i = 0
        while i < iters:
            prob = logistic(x * (torch.zeros(n, device=self.device).index_add_(0, row, adj * x[col]) + local))
            sample = torch.rand(n, device=self.device)
            x[prob > sample] = 0
            x[prob < sample] = 1
            # x = torch.round(prob)
            i += 1
        # Return the state of the visible units
        return x
