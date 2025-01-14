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

    def __init__(self, gtype, n, k, numbers, seed=42, load=False,
                 lr=0.1, damping=0.2, eps=1e-5, th=10.0, max_iters=200, lr_decay=0.1, device=torch.device("cpu")):
        # Model parameters
        torch.manual_seed(seed)
        self.gtype = gtype
        self.n = n
        self.k = k
        self.numbers = len(numbers)
        self.lr_decay = lr_decay
        self.best_bethe = -float('inf')
        self.device = device

        # Load model data
        if load:
            self.load_params()
        else:
            if gtype == GType.ER: # Use igraph for erdos-renyi
                G = igraph.Graph.Erdos_Renyi(n=n, p=2 * math.log(n) / n)
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
            adj = (torch.rand(row.shape[0], device=device) - 0.5) * 0.01
            local = (torch.rand(n, device=device) - 0.5) * 0.01
            # local = (torch.zeros(n, device=device))


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
        n = self.n
        k = self.k
        numbers = self.numbers
        prefix = '%d_%d_' % (numbers, n)
        if self.gtype == GType.KR:
            prefix += '%d_' % k
        fn_weights = os.path.join(MODEL_DIR, self.gtype.name, prefix + 'weights.pt')
        fn_adjecency = os.path.join(MODEL_DIR, self.gtype.name,  prefix + 'adjecency.pt')

        if not os.path.exists(fn_weights):
            return False
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
        return True


    def save_params(self):
        """Save model parameters
        """
        n = self.n
        k = self.k
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
        prefix = '%d_%d_' % (numbers, n)
        if self.gtype == GType.KR:
            prefix += '%d_' % k
        fn_weights = os.path.join(MODEL_DIR, self.gtype.name, prefix + 'weights.pt')
        fn_adjecency = os.path.join(MODEL_DIR, self.gtype.name,  prefix + 'adjecency.pt')

        torch.save((local, adj), fn_weights)
        torch.save((row, col, r2c), fn_adjecency)

    def train(self, train_set, epochs, batch_size, sampler=None, save=True):
        """ Perform training using message passing for graphical model.
        """
        if torch.cuda.is_available():
            num_workers = 4
        else:
            num_workers = 0
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        bethe_trend = []
        # lr_decay_epoch = [3]
        for i in range(epochs):
            bethe_epoch = []
            print("epoch %d" %(i))
            for j, (data, label) in enumerate(train_loader):
                print("batch %d" %(j))
                data = data.squeeze()
                data = torch.round(torch.transpose(data.view(-1, data.shape[1] ** 2), 0, 1))
                if torch.cuda.is_available():
                    data = data.cuda()
                self.free_mp()
                self.clamp_mp(data)
                bethe_batch = self.update(data)
                bethe_epoch += [bethe_batch]
                print("     epoch bethe: %.3g \n" % (sum(bethe_epoch) / (j + 1)))
            self.save_params()
            # if i in lr_decay_epoch:
            #     self.lr *= 0.1
            #     self.eps *= 0.1
                # self.damping += 0.2
            bethe_trend += bethe_epoch
        print("     local weights")
        print(self.local[:data.shape[0]])
        return bethe_trend

    def free_mp(self):
        """ Unclamped message-passing for one mini-batch.
        """
        adj = self.adj
        local = self.local
        row = self.row
        col= self.col
        n = self.n
        m = adj.shape[0]

        # Initialize to zeros because using log ratio
        message_old = torch.zeros(m, device=self.device)
        message_new = torch.zeros(m, device=self.device)
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_new.clone()
            message = (torch.zeros(n, device=self.device).index_add_(0, row, message_old)[row] - message_old)[self.r2c]
            message_new = local[row]
            # For some reason samples from model never explode
            message_new += torch.log(torch.exp(local[col] + adj + message) + 1)
            message_new -= torch.log(torch.exp(local[col] + message) + 1)
            message_new[torch.isinf(message_new) + torch.isnan(message_new)] = adj[torch.isinf(message_new) + torch.isnan(message_new)].clone()
            message_new = message_new * (1 - self.damping) + message_old * self.damping
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
        adj = self.adj.clone().unsqueeze(1).repeat(1, batch_size)
        local = self.local.clone().unsqueeze(1).repeat(1, batch_size)
        row = self.row
        col = self.col
        n = local.shape[0]

        # Add interaction of clamped units to local biases based on data vectors
        local.index_add_(0, col[row < clamp], adj[row < clamp] * data[row[row < clamp]])
        local[:clamp] = 0
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
            message_new += torch.log(torch.exp(local[col] + adj + message) + 1)
            message_new -= torch.log(torch.exp(local[col] + message) + 1)
            message_new[torch.isinf(message_new) + torch.isnan(message_new)] = adj[torch.isinf(message_new) + torch.isnan(message_new)]
            message_new = message_new * (1 - self.damping) + message_old * self.damping
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
        n = self.n

        #Old without external
        msg_free = self.message_free
        sum_free = (torch.zeros(n, device=self.device).index_add_(0, row, msg_free)[row] - msg_free)
        p_ij_marg = 1 / (torch.exp(-(sum_free + sum_free[r2c] + local[row] + local[col] + adj)) \
        + torch.exp(-(sum_free + local[row] + adj)) + torch.exp(-(sum_free[r2c] + local[col] + adj)) + 1)

        #Calculate conditional joint probability
        msg_clamp = self.message_clamp
        adj = adj.unsqueeze(1).expand(-1, batch_size)
        local = local.unsqueeze(1).expand(-1, batch_size)
        # Add unit that sends signal of +-100 based on data
        msg_sum = torch.zeros((n, batch_size), device=self.device).index_add_(0, row, msg_clamp)
        msg_sum[:clamp] += (data * 2 - 1) * 100
        sum_clamp = msg_sum[row] - msg_clamp
        p_ij_cond = 1 / (torch.exp(-(sum_clamp + sum_clamp[r2c] + local[row] + local[col] + adj)) \
        + torch.exp(-(sum_clamp + local[row] + adj)) + torch.exp(-(sum_clamp[r2c] + local[col] + adj)) + 1)

        #Calculate marginal and condtional probabilities
        message_clamp[:clamp] += (data * 2 - 1) * 100
        p_i_marg = logistic(torch.zeros(n, device=self.device).index_add_(0, row, message_free) + self.local)
        p_i_cond = logistic(torch.zeros((n, batch_size), device=self.device).index_add_(0, row, message_clamp) + local)

        # Average over batchs
        p_ij_cond = torch.sum(p_ij_cond, 1) / batch_size
        p_i_cond = torch.sum(p_i_cond, 1) / batch_size

        mask = (row >= clamp) * (col >= clamp) * (1 - torch.isinf(torch.log(p_ij_cond)))
        neg_clamp = torch.sum(torch.log(p_ij_cond)[mask] * p_ij_cond[mask]) / 2
        const_clamp = torch.zeros(n, device=self.device).index_add_(0, row[col >= clamp], torch.ones(row[col >= clamp].shape[0], device=self.device)) - 1
        mask = (p_i_cond != 0)
        pos_clamp = torch.sum(const_clamp[mask] * (p_i_cond[mask] * torch.log(p_i_cond[mask])))
        entropy_clamp = pos_clamp - neg_clamp
        energy = torch.sum((p_ij_cond - p_ij_marg) * self.adj) / 2 + torch.sum((p_i_cond - p_i_marg) * self.local)

        neg = torch.sum(torch.log(p_ij_marg) * p_ij_marg) / 2
        const = torch.zeros(n, device=self.device).index_add_(0, row, torch.ones(row.shape[0], device=self.device)) - 1
        pos = torch.sum(const * (p_i_marg * torch.log(p_i_marg)))
        entropy = pos - neg

        bethe = (entropy_clamp - entropy) + energy

        delta_loc = torch.norm(p_i_cond - p_i_marg)
        delta_adj = torch.norm(p_ij_cond - p_ij_marg)

        #Update model parameters
        th = self.th
        self.adj = torch.clamp(self.adj + self.lr * (p_ij_cond - p_ij_marg), max=th)
        self.local = torch.clamp(self.local + self.lr * (p_i_cond - p_i_marg), max=th)
        print("     avg weight: %.3g" % (torch.sum(self.adj) / self.adj.shape[0]))
        print("     max weight: %.3g" % (torch.max(self.adj)))
        print("     min weight: %.3g" % (torch.min(self.adj)))
        print("     avg bias: %.3g" % (torch.sum(self.local) / self.local.shape[0]))
        print("     max bias: %.3g" % (torch.max(self.local)))
        print("     min bias: %.3g" % (torch.min(self.local)))
        print("     delta local: %.3g" % delta_loc)
        print("     delta adj: %.3g" % delta_adj)
        print("     batch bethe: %.3g" % bethe)

        return bethe

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
        n = self.n
        # Initialize units to random weights
        if x is None:
            x = torch.round(torch.rand(n, device=self.device))
        # Sample iter times
        i = 0
        while i < iters:
            prob = logistic(x * (torch.zeros(n, device=self.device).index_add_(0, row, adj * x[col]) + local))
            sample = torch.rand(n, device=self.device)
            x[prob > sample] = 1
            x[prob < sample] = 0
            # x = torch.round(prob)
            i += 1
        # Return the state of the visible units
        return x
