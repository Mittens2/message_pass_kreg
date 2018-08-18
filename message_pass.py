import torch
import numpy as np
import math
from utils import logistic, time
from torch.utils.data import DataLoader

MODEL_DIR = 'data/model/'

class SparseMP():

    def __init__(self, adj, local, row, col, r2c, lr=0.1, damping=0.5, eps=1e-16, th=0.2, epochs=100, max_iters=10, batch_size=1, device=torch.device("cpu")):
        #torch.manual_seed(0)
        self.device = device
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        self.eps = eps
        self.th = th
        self.epochs = epochs
        self.batch_size = batch_size
        self.adj = adj
        self.local = local
        self.row = row
        self.col = col
        self.r2c = r2c


    def train(self, train_set, sampler=None,save=False):
        """ Perform training for k-regular graph using loopy bp.
        """
        train_loader = iter(DataLoader(train_set, batch_size=self.batch_size, sampler=sampler, num_workers=1))
        i = 0
        while i < self.epochs:
            print("epoch %d" %(i))
            # data, label = train_set[0]
            # data = torch.round(data.view(-1).unsqueeze(1))
            data, label = next(train_loader)
            data = data.squeeze()
            data = torch.round(torch.transpose(data.view(-1, data.shape[1] ** 2), 0, 1))
            if torch.cuda.is_available():
                data = data.cuda()
            self.free_mp()
            self.clamp_mp(data)
            self.update(data)
            i += 1
        if save:
            torch.save((self.local, self.adj), MODEL_DIR + "weights.pt")
            torch.save((self.row, self.col, self.r2c), MODEL_DIR + "adjacency.pt")

    def free_mp(self):
        """ Message-passing for unclamped graphical model.
        """
        adj = self.adj
        local = self.local
        row = self.row
        col= self.col
        n = local.shape[0]
        m = adj.shape[0]

        # Initialize to zeros because using log ratio
        message_old = torch.zeros(m, device=self.device)
        message_new = torch.zeros(m, device=self.device)
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_new.clone()
            message = (torch.zeros(n, device=self.device).index_add_(0, row, message_old)[row] - message_old)[self.r2c]
            message_new = torch.log((torch.exp(local[col] + adj + message) + 1) / (torch.exp(local[col] + message) + 1))
            message_new = message_new * self.damping + message_old * (1 - self.damping)
            iters+=1
        print("     %d iterations until convergence, minimum difference %.3e" % (iters,  torch.max(torch.abs(message_new - message_old))))
        self.message_free = message_new

    def clamp_mp(self, data):
        """ Clamped message-passing for one mini-batch.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data to use for training.
        """
        adj = self.adj.unsqueeze(1).expand(-1, self.batch_size)
        local = self.local.unsqueeze(1).expand(-1, self.batch_size)
        row = self.row
        col = self.col
        n = local.shape[0]
        clamp = data.shape[0]

        # Add interaction of clamped units to local biases based on data vectors
        local.index_add_(0, col[row < clamp], adj[row < clamp] * data[row[row < clamp]])
        # Set interaction of clamped units to 0
        adj[row < clamp] = 0
        adj[col < clamp] = 0

        message_old = self.message_free.unsqueeze(1).repeat(1, self.batch_size)
        message_new = message_old.clone()
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_new.clone()
            message = (torch.zeros((n, self.batch_size), device=self.device).index_add_(0, row, message_old)[row] - message_old)[self.r2c]
            message_new = torch.log((torch.exp(local[col] + adj + message) + 1) / (torch.exp(local[col] + message) + 1))
            message_new = message_new * self.damping + message_old * (1 - self.damping)
            iters+=1
        print("     %d iterations until convergence, minimum difference %.3e" % (iters,  torch.max(torch.abs(message_new - message_old))))
        self.message_clamp = message_new

    def update(self, data):
        """ Update of model parameters.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data to use for training.
        """
        adj = self.adj
        local = self.local
        row = self.row
        col = self.col
        r2c = self.r2c
        n = local.shape[0]
        clamp = data.shape[0]

        # Calculate marginal joint probability
        msg_free = self.message_free
        sum_free = (torch.zeros(n, device=self.device).index_add_(0, row, msg_free)[row] - msg_free)
        p_ij_marg = self.batch_size * 1 / (torch.exp(-(sum_free + sum_free[r2c] + local[row] + local[col] + adj)) \
        + torch.exp(-(sum_free + local[row] + adj)) + torch.exp(-(sum_free[r2c] + local[col] + adj)) + 1)

        # Calculate conditional joint probability
        msg_clamp = self.message_clamp
        adj = self.adj.clone().unsqueeze(1).expand(-1, self.batch_size)
        local = local.unsqueeze(1).expand(-1, self.batch_size)
        # Add unit that sends signal of +-100 based on data
        msg_sum = torch.zeros((n, self.batch_size), device=self.device).index_add_(0, row, msg_clamp)
        msg_sum[:clamp] += (data * 2 - 1) * 100
        sum_clamp = msg_sum[row] - msg_clamp
        p_ij_cond = 1 / (torch.exp(-(sum_clamp + sum_clamp[r2c] + local[row] + local[col] + adj)) \
        + torch.exp(-(sum_clamp + local[row] + adj)) + torch.exp(-(sum_clamp[r2c] + local[col] + adj)) + 1)
        p_ij_cond = torch.sum(p_ij_cond, 1)

        # Calculate marginal and condtional probabilities
        p_i_marg = self.batch_size * logistic(torch.zeros(n, device=self.device).index_add_(0, row, msg_free))
        p_i_cond = torch.sum(logistic(torch.zeros((n, self.batch_size), device=self.device).index_add_(0, row, msg_clamp)), 1)

        # Update model parameters
        th = self.th
        self.adj = torch.clamp(self.adj + self.lr * (p_ij_cond - p_ij_marg), max=th)
        self.local = torch.clamp(self.local + self.lr * (p_i_cond - p_i_marg), max=th)
        print("     avg weight: %.3e" % (torch.sum(self.adj) / self.adj.shape[0]))
        print("     max weight: %.3e" % (torch.max(self.adj)))
        print("     min weight: %.3e" % (torch.min(self.adj)))

    def gibbs(self, iters, avg_over, x=None):
        """ Perform iters iterations of gibbs sampling, and return result.
        Parameters
        ----------
        x : array-like, shape (n_adj)
            The data to use for training.
        """
        adj = self.adj.clone()
        local = self.local
        row = self.row
        col = self.col
        n = local.shape[0]
        # Initialize units to random weights
        if x is None:
            x = torch.round(torch.rand(n, device=self.device))
        y = torch.zeros(n, device=self.device)
        # Sample iter times
        i = 0
        while i < iters:
            prob = logistic(x * (torch.zeros(n, device=self.device).index_add_(0, row, adj * x[col]) + local))
            # sample = torch.rand(n, device=self.device)
            # x[prob > sample] = 1
            # x[prob < sample] = 0
            x = torch.round(prob)
            if iters - i <= avg_over:
                y += x
            i += 1
        # Return the state of the visible units
        # return x[:data.view(-1).shape[0]].view(data.shape[0], -1)
        # return torch.round(y[:data.view(-1).shape[0]].view(data.shape[0], -1) / iters)
        return torch.round(y / avg_over)
