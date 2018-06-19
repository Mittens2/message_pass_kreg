import torch
import numpy as np
import math
from utils import logistic, time
from torch.utils.data import DataLoader

class SparseMP():

    def __init__(self, adj, adj_list=None, lr=0.3, damping=0.2, eps=1e-16, epochs=10, max_iters=20, batch_size=1):
        #torch.manual_seed(0)
        n = adj.shape[0]
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        self.eps = eps
        self.epochs = epochs
        self.batch_size = batch_size
        # TODO: since adjacency matrix is being passed, only allowing for interraction terms
        # Use upper triangular matrix
        adj = (torch.from_numpy(adj).type(torch.FloatTensor) * (torch.rand(n , n) * 2 - 1))
        self.adj = torch.triu(adj)
        self.adj_list = torch.from_numpy(adj_list).type(torch.LongTensor)

    def train(self, train_set):
        """ Perform training for k-regular graph using loopy bp.
        """
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=1)
        i = 0
        while i < self.epochs:
            data, label = train_set[0]
            data = data.view(-1).unsqueeze(1)
            #data, label = next(iter(self.train_loader))
            #data = data.squeeze()
            #data = torch.transpose(data.view(-1, data.shape[1] ** 2), 0, 1)
            self.free_mp()
            self.clamp_mp(data)
            self.update(data)
            print(i)
            i += 1

    def free_mp(self):
        """ Message-passing for unclamped graphical model.
        """
        adj = self.adj
        adj_list = self.adj_list
        n = adj.shape[0]
        k = adj_list.shape[1]
        # Initialize to zeros because using log ratio
        message_old = torch.zeros(n, n)
        message_new = torch.zeros(n, n)
        iters = 0
        while iters == 0 or (iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old.scatter_(1, adj_list, message_new.gather(1, adj_list))

            message = torch.transpose(torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0)
            .unsqueeze(0).expand(k, -1) - message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
            local = torch.diag(adj).unsqueeze(1).expand(-1, k)
            # Gather pattern for upper triangular
            interaction = adj.gather(1, adj_list) + torch.transpose(adj.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)

            message_new.scatter_(1 , adj_list, torch.log((torch.exp(local + interaction + message) + 1) / (torch.exp(local + message) + 1)))

            message_new.scatter_(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters+=1

        self.message_old = message_new

    def clamp_mp(self, data):
        """ Clamped message-passing for one mini-batch.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data to use for training.
        """
        adj = self.adj.clone()
        n = adj.shape[0]
        clamp = data.size()[0]
        adj = adj.unsqueeze(2).expand(n, -1, self.batch_size)
        adj_list = self.adj_list.unsqueeze(2).expand(n, -1, self.batch_size)
        k = adj_list.shape[1]
        # Clamp visible units
        adj[:-clamp, :-clamp, :].scatter_(1, torch.arange(n - clamp).unsqueeze(1).unsqueeze(2)
        .expand(-1, n - clamp, self.batch_size).type(torch.LongTensor), \
        torch.sum(adj[:-clamp, -clamp:, :] * data.unsqueeze(0).expand(n - clamp, -1, self.batch_size), 1)
        .unsqueeze(0).expand(n - clamp, -1, self.batch_size))
        # Set adjacency matrix to 0 where clamped to facilitate gather/scatter pattern
        adj[:, -clamp:, :] = 0
        adj[-clamp:, :, :] = 0

        message_old = self.message_old.clone().unsqueeze(2).expand(n, n, self.batch_size)
        message_old[:, :, -clamp:] = 0
        message_old[:, -clamp:, :] = 0
        message_new = message_old.clone()

        iters = 0
        while iters == 0 or iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps:
            message_old.scatter_(1, adj_list, message_new.gather(1, adj_list))

            message = torch.transpose(torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0)
            .unsqueeze(0).expand(k, -1, self.batch_size) - message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
            local = torch.transpose(adj[torch.eye(n) == 1].unsqueeze(2), 1, 2).expand(-1, k, self.batch_size)
            interaction = adj.gather(1, adj_list) + torch.transpose(adj.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)

            message_new.scatter_(1 , adj_list, torch.log((torch.exp(local + interaction + message) + 1) / (torch.exp(local + message) + 1)))

            message_new.scatter_(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters += 1

        self.message_new = message_new

    def update(self, data):
        """ Update of model weights.
        Parameters
        ----------
        data : array-like, shape (n_data ** 2, n_batch)
            The data to use for training.
        """
        adj = self.adj
        adj_list = self.adj_list.clone()
        n = adj.shape[0]
        k = adj_list.shape[1]
        message_old = self.message_old
        message_new = self.message_new

        # Update interaction terms
        local_i = torch.diag(adj).unsqueeze(1).expand(-1, k)
        local_j = torch.diag(adj).unsqueeze(0).expand(n, -1).gather(1, adj_list)
        interaction = adj.gather(1, adj_list) + torch.transpose(adj.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
        msg_joint_free_i = torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0).unsqueeze(1).expand(-1, k) - torch.transpose(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
        msg_joint_free_j = torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0).unsqueeze(0).expand(n, -1).gather(1, adj_list) - message_old.gather(1, adj_list)
        prob_free = 1 / (torch.exp(-(msg_joint_free_i + msg_joint_free_j + local_i + local_j + interaction)) + torch.exp(-(msg_joint_free_i + local_i + interaction)) + torch.exp(-(msg_joint_free_j + local_j + interaction)) + 1)

        # Add a fake unit that sends a message of positive or negative infinity if unit is clamped on/off
        adj_clamp = adj_list.clone()
        adj_clamp = torch.cat((adj_list, torch.ones(n, dtype=torch.long).unsqueeze(1)), 1)
        adj_clamp = adj_clamp.unsqueeze(2).expand(n, -1, self.batch_size)
        adj_list = adj_list.unsqueeze(2).expand(n, -1, self.batch_size)
        message_new = torch.cat((message_new, torch.cat((torch.zeros(n - data.shape[0], self.batch_size), (data * 2 - 1) * 100), 0).unsqueeze(0)), 0)

        local_i = local_i.unsqueeze(2).expand(n, -1, self.batch_size)
        local_j = local_j.unsqueeze(2).expand(n, -1, self.batch_size)
        interaction = interaction.unsqueeze(2).expand(n, -1, self.batch_size)
        msg_joint_clamp_i = torch.transpose(torch.sum(message_new.gather(0, torch.transpose(adj_clamp, 0, 1)), 0).unsqueeze(2), 1, 2).expand(-1, k, self.batch_size)
        - torch.transpose(message_new.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
        msg_joint_clamp_j = torch.sum(message_new.gather(0, torch.transpose(adj_clamp, 0, 1)), 0).unsqueeze(0).expand(n, -1, self.batch_size).gather(1, adj_list)
        - message_new[:-1, :].gather(1, adj_list)

        prob_clamp = 1 / (torch.exp(-(msg_joint_clamp_i + msg_joint_clamp_j + local_i + local_j + interaction)) + torch.exp(-(msg_joint_clamp_i + local_i + interaction)) + torch.exp(-(msg_joint_clamp_j + local_j + interaction)) + 1)
        prob_clamp = torch.sum(prob_clamp, 2) / self.batch_size

        adj_list = self.adj_list
        adj.scatter_add_(1, adj_list, self.lr * (prob_clamp - prob_free))
        adj.scatter_add_(0, torch.transpose(adj_list, 0, 1), torch.transpose(self.lr * (prob_clamp - prob_free), 0, 1))
        adj = torch.triu(adj)

        # Update local terms
        marginals = logistic(torch.sum(torch.gather(message_old, 0, torch.transpose(adj_list, 0, 1)), 0))
        conditionals = torch.sum(logistic(torch.sum(message_new.gather(0, torch.transpose(adj_clamp, 0, 1)), 0)), 1) / self.batch_size
        adj[(torch.eye(n) == 1)] += self.lr * (conditionals - marginals)
        eq = torch.eq(adj, torch.transpose(adj, 0, 1))
        print(adj)

    def gibbs(self, data, n):
        """ Perform n iterations of gibbs sampling, and return result.
        Parameters
        ----------
        data : array-like, shape (n_data, n_data)
            The data to use for training.
        """
        adj = self.adj.clone()
        adj_list = self.adj_list
        n = adj.shape[0]
        k = adj_list.shape[1]
        # Get locals, set diagonal of weights to 0
        local = torch.diag(adj).unsqueeze(1)
        adj[(torch.eye(n) == 1)] = 0
        # Initialize units to random weights
        x = torch.round(torch.rand(n)).unsqueeze(1)
        x_new = torch.round(logistic(torch.mm(adj, x) + local))
        # Sample until convergence
        i = 0
        while not torch.eq(x, x_new).all() and i < n:
            x = x_new.clone()
            #print(x[-data.view(-1).shape[0]:].view(data.shape[0], -1))
            x_new = torch.round(logistic(torch.mm(adj, x) + local))
            i += 1
        x = x.squeeze(1)
        # Return the state of the visible units
        return x[-data.view(-1).shape[0]:].view(data.shape[0], -1)
