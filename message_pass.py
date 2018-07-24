import torch
import numpy as np
import math
from utils import logistic, time
from torch.utils.data import DataLoader

class SparseMP():

    def __init__(self, adj, local, adj_list, lr=0.1, damping=1., eps=1e-5, epochs=100, max_iters=10, batch_size=1):
        #torch.manual_seed(0)
        n = adj.shape[0]
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        self.eps = eps
        self.epochs = epochs
        self.batch_size = batch_size
        # Use upper triangular matrix
        if torch.cuda.is_available():
            self.local = local.cuda()
            self.adj = adj.cuda()
        else:
            self.adj = adj
            self.local = local
        self.adj_list = adj_list

    def train(self, train_set):
        """ Perform training for k-regular graph using loopy bp.
        """
        train_loader = iter(DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=1))
        i = 0
        while i < self.epochs:
            # data, label = train_set[0]
            # data = data.view(-1).unsqueeze(1)
            data, label = next(train_loader)
            data = data.squeeze()
            data = torch.round(torch.transpose(data.view(-1, data.shape[1] ** 2), 0, 1))
            self.free_mp()
            self.clamp_mp(data)
            self.update(data)
            print(i)
            i += 1

    def in_to_out(self, message_in, adj_list):
        n = adj_list.shape[0]
        k = adj_list.shape[1]
        range = torch.arange(n, dtype=torch.long).unsqueeze(1).expand(-1, k).contiguous().view(-1).unsqueeze(1)
        mask = (adj_list.index_select(0, adj_list.view(-1)) == range)
        return message_in.index_select(0, adj_list.view(-1))[mask]

    def free_mp(self):
        """ Message-passing for unclamped graphical model.
        """
        adj = self.adj
        n = adj.shape[0]
        k = adj.shape[1]
        local = self.local.unsqueeze(1).expand(-1, k)
        adj_list = self.adj_list
        # Initialize to zeros because using log ratio
        message_old = torch.zeros(n, k)
        message_new = torch.zeros(n, k)
        if torch.cuda.is_available():
            message_old = message_old.cuda()
            message_new = message_new.cuda()
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_new.clone()
            message = torch.sum(message_old, 1).unsqueeze(1).expand(-1, k) - message_old
            message_new = self.in_to_out(torch.log((torch.exp(local + adj + message) + 1) / (torch.exp(local + message) + 1)), adj_list).view(n, -1)
            message_new = message_new * self.damping + message_old * (1 - self.damping)
            iters+=1
        print(iters)
        self.message_free = message_new

    def clamp_mp(self, data):
        """ Clamped message-passing for one mini-batch.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data to use for training.
        """
        n = self.adj.shape[0]
        adj = self.adj.unsqueeze(2).expand(n, -1, self.batch_size)
        k = adj.shape[1]
        local = self.local.unsqueeze(1).expand(-1, self.batch_size)
        adj_list = self.adj_list
        clamp = data.shape[0]
        # Add interaction of clamped units to local biases based on data vectors
        data_adj = adj.clone()
        data_adj[adj_list < clamp] *= data[adj_list[adj_list < clamp]]
        data_adj[adj_list >= clamp] = 0
        local += torch.sum(data_adj, 1)
        local = torch.transpose(local.unsqueeze(2).expand(n, -1, k), 1, 2)
        # Set interaction of clamped units to 0
        adj[:clamp, :, :] = 0
        adj[adj_list < clamp] = 0

        message_old = self.message_free.clone().unsqueeze(2).repeat(1, 1, self.batch_size)
        message_new = message_old.clone()

        iters = 0
        while iters == 0 or iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps:
            message_old = message_new.clone()
            message = torch.transpose(torch.sum(message_old, 1).unsqueeze(2).expand(n, -1, k), 1, 2) - message_old
            message_new = self.in_to_out(torch.log((torch.exp(local + adj + message) + 1) / (torch.exp(local + message) + 1)), adj_list=adj_list).view(n, -1, self.batch_size)
            message_new = message_new * self.damping + message_old * (1 - self.damping)
            iters+=1

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
        adj_list = self.adj_list.clone()
        n = adj.shape[0]
        k = adj_list.shape[1]
        clamp = data.shape[0]

        # Calculate marginal joint probability
        message_free = self.message_free
        local_i = local.unsqueeze(1).expand(-1, k)
        local_j = local.unsqueeze(0).expand(n, -1).gather(1, adj_list)
        msg_free_i = torch.sum(message_free, 1).unsqueeze(1).expand(-1, k) - message_free
        msg_free_j = self.in_to_out(msg_free_i, adj_list=adj_list).view(n, -1)
        p_ij_marg = 1 / (torch.exp(-(msg_free_i + msg_free_j + local_i + local_j + adj)) \
        + torch.exp(-(msg_free_i + local_i + adj)) + torch.exp(-(msg_free_j + local_j + adj)) + 1)

        # Calculate conditional joint probability
        message_clamp = self.message_clamp
        adj = self.adj.unsqueeze(2).expand(n, -1, self.batch_size)
        local_i = local_i.unsqueeze(2).expand(n, -1, self.batch_size)
        local_j = local_j.unsqueeze(2).expand(n, -1, self.batch_size)
        # Add unit that sends signal of +-100 based on data
        msg_sum = torch.sum(message_clamp, 1)
        msg_sum[:clamp] += (data * 2 - 1) * 100
        msg_clamp_i = torch.transpose(msg_sum.unsqueeze(2).expand(n, -1, k), 1, 2) - message_clamp
        msg_clamp_j = self.in_to_out(msg_clamp_i, adj_list=adj_list).view(n, -1, self.batch_size)
        p_ij_cond = 1 / (torch.exp(-(msg_clamp_i + msg_clamp_j + local_i + local_j + adj)) \
        + torch.exp(-(msg_clamp_i + local_i + adj)) + torch.exp(-(msg_clamp_j + local_j + adj)) + 1)
        p_ij_cond = torch.sum(p_ij_cond, 2) / self.batch_size

        # Calculate marginal and condtional probabilities
        p_i_marg = logistic(torch.sum(message_free, 1))
        p_i_cond = torch.sum(logistic(torch.sum(message_clamp, 1)), 1) / self.batch_size

        # Update model parameters
        self.adj = torch.clamp(self.adj + self.lr * self.batch_size * (p_ij_cond - p_ij_marg), -0.95, 0.95)
        print(self.adj[data.shape[0] - 1])
        print(self.adj[data.shape[0]])
        self.local = torch.clamp(self.local + self.lr * self.batch_size * (p_i_cond - p_i_marg), -0.95, 0.95)

    def gibbs(self, data, iters):
        """ Perform n iterations of gibbs sampling, and return result.
        Parameters
        ----------
        data : array-like, shape (n_data, n_data)
            The data to use for training.
        """
        adj = self.adj.clone()
        local = self.local
        adj_list = self.adj_list
        n = adj.shape[0]
        k = adj_list.shape[1]
        # Initialize units to random weights
        x = torch.round(torch.rand(n))
        x_new = (torch.rand(n) < logistic(torch.sum(adj * x.unsqueeze(0).expand(n, -1).gather(1, adj_list), 1) + local)).type(torch.FloatTensor)
        # Sample until convergence
        i = 0
        while not torch.eq(x, x_new).all() and i < iters:
            x = x_new.clone()
            x_new = (torch.rand(n) < logistic(torch.sum(adj * x.unsqueeze(0).expand(n, -1).gather(1, adj_list), 1) + local)).type(torch.FloatTensor)
            i += 1
        # Return the state of the visible units
        return x[:data.view(-1).shape[0]].view(data.shape[0], -1)
