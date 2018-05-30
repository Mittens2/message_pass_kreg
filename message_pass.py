import torch
import numpy as np
from utils import logistic, time

class SparseMP():

    def __init__(self, train_set, adj, adj_list=None, lr=0.01, damping=0.1, eps=1e-16, epochs=100, max_iters=10):
        torch.manual_seed(42)
        self.train_set = train_set
        n = adj.shape[0]
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        # TODO: since adjacency matrix is being passed, only allowing for interraction terms
        self.adj = 2 * (torch.from_numpy(adj).type(torch.FloatTensor) * (torch.rand(n , n) - 0.5))
        # TODO: implement adjacency list without having it passed
        if adj_list is None:
            pass
        self.adj_list = torch.from_numpy(adj_list).type(torch.LongTensor)
        self.inference()

    def inference(self):
        #for data, label in self.train_set:
        data = self.train_set[0][0]
        self.forward()
        self.backward(data.view(-1))

    def forward(self):
        adj = self.adj
        adj_list = self.adj_list
        n = adj.shape[0]
        k = adj_list.shape[1]
        # Initialize to zeros because using log ratio
        message_old = torch.zeros(n, n)
        message_new = torch.zeros(n, n)
        iters = 0
        while iters == 0 or iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            # Old way of calculating marginals
            # message_new[mask] = (torch.sum(torch.gather(message_old, 1, adj_list), 1)
            # .unsqueeze(1).repeat(1, n) - message_old)[mask]
            # local = torch.zeros(n, n).masked_scatter(mask, torch.diag(torch.diag(adj)))
            # message_new += torch.log((torch.exp(local + adj) + 1) / (torch.exp(local) + 1)) + local
            # message_new = message_new * (self.damping) + (1 - self.damping) * message_old
            # message_old = message_new.clone()
            message_old = message_old.scatter(1, adj_list, message_new.gather(1, adj_list))

            message_new = message_new.scatter(1, adj_list, torch.sum(message_old.gather(1, adj_list), 1)
            .unsqueeze(1).expand(-1, k) - message_old.gather(1, adj_list))

            local = torch.diag(adj).unsqueeze(0).expand(n, -1)
            message_new.scatter_add_(1, adj_list, torch.log((torch.exp(adj.gather(1, adj_list)
            + local.gather(1, adj_list)) + 1) / (torch.exp(local.gather(1, adj_list)) + 1)))

            message_new = message_new.scatter(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters+=1

        self.marginals = logistic(torch.sum(torch.gather(message_new, 1, adj_list), 1))
        self.message_old = message_new

    def backward(self, data):
        n = self.adj.shape[0]
        # Condition on next data point
        clamp = data.size()[0]
        adj_list = self.adj_list
        adj = self.adj.clone()
        adj[:-clamp, :-clamp] += torch.diag(torch.sum(adj[:-clamp, -clamp:], 1))
        # Set adjacency matrix to 0 where clamped to facilitate gather/scatter pattern
        adj[:, -clamp:] = 0
        adj[-clamp:, :] = 0

        message_old = self.message_old
        message_old[:, -clamp:] = 0
        message_old[-clamp:, :] = 0
        message_new = message_old.clone()

        iters = 0
        while iters == 0 or iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps):
            message_old = message_old.scatter(1, adj_list, message_new.gather(1, adj_list))

            message_new = message_new.scatter(1, adj_list, torch.sum(message_old.gather(1, adj_list), 1)
            .unsqueeze(1).expand(-1, n).gather(1, adj_list) - message_old.gather(1, adj_list))

            local = torch.cat((torch.diag(adj).unsqueeze(0).expand(n - clamp, -1), torch.zeros(clamp, n)), 0)
            message_new.scatter_add_(1, adj_list, torch.log((torch.exp(adj.gather(1, adj_list)
            + local.gather(1, adj_list)) + 1) / (torch.exp(local.gather(1, adj_list)) + 1)))

            message_new = message_new.scatter(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters += 1

        conditionals = logistic(torch.sum(message_old, 1)).unsqueeze(1)
        marginals = self.marginals.unsqueeze(1)
        # update model parameters
        # TODO: Maybe for interaction terms should use joint probability, not marginals?
        update = torch.mm(conditionals, torch.transpose(conditionals, 0, 1)).gather(1, adj_list)
        - torch.mm(marginals, torch.transpose(marginals, 0, 1)).gather(1, adj_list)
        adj.scatter_add_(1, adj_list, self.lr * update)
        adj[(torch.eye(n) != 0)] += self.lr * (conditionals.squeeze(1) - marginals.squeeze(1))
