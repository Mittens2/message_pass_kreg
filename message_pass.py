import torch
import numpy as np
from utils import logistic, time

class SparseMP():

    def __init__(self, train_set, adj, adj_list=None, lr=0.1, damping=0.1, eps=1e-16, epochs=100, max_iters=10asaass):
        #torch.manual_seed(0)
        self.train_set = train_set
        n = adj.shape[0]
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        self.eps = eps
        self.epochs = epochs
        # TODO: since adjacency matrix is being passed, only allowing for interraction terms
        adj = 2 * (torch.from_numpy(adj).type(torch.FloatTensor) * (torch.rand(n , n) * 0.1 - 0.5))
        self.adj = adj * torch.transpose(adj, 0, 1)
        # TODO: implement adjacency list without having it passed
        if adj_list is None:
            pass
        self.adj_list = torch.from_numpy(adj_list).type(torch.LongTensor)
        self.inference()

    def inference(self):
        i = 0
        while i < self.epochs:
            data, label = self.train_set[0]
            self.forward()
            self.backward(data.view(-1))
            self.update()
            print(i)
            i += 1

    def forward(self):
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
            interaction = adj.gather(1, adj_list)

            message_new.scatter_(1 , adj_list, torch.log((torch.exp(local + interaction + message) + 1) / (torch.exp(local + message) + 1)))
            # Damping
            message_new.scatter_(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters+=1

        self.marginals = logistic(torch.sum(torch.gather(message_new, 0, torch.transpose(adj_list, 0, 1)), 0))
        self.message_old = message_new

    def backward(self, data):
        clamp = data.size()[0]
        adj_list = self.adj_list
        adj = self.adj.clone()
        n = adj.shape[0]
        k = adj_list.shape[1]
        # Clamp visible units
        adj[:-clamp, :-clamp].scatter_(1, torch.arange(n - clamp).unsqueeze(1).type(torch.LongTensor), \
        torch.sum(adj[:-clamp, -clamp:] * data.unsqueeze(0).expand(n - clamp, -1), 1).unsqueeze(0).expand(n - clamp, -1))
        # Set adjacency matrix to 0 where clamped to facilitate gather/scatter pattern
        adj[:, -clamp:] = 0
        adj[-clamp:, :] = 0

        message_old = self.message_old
        message_old[:, -clamp:] = 0
        message_old[-clamp:, :] = 0
        message_new = message_old.clone()

        iters = 0
        while iters == 0 or iters < self.max_iters and torch.max(torch.abs(message_new - message_old)) > self.eps:
            message_old.scatter_(1, adj_list, message_new.gather(1, adj_list))

            message = torch.transpose(torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0)
            .unsqueeze(0).expand(k, -1) - message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
            local = torch.diag(adj).unsqueeze(1).expand(-1, k)
            interaction = adj.gather(1, adj_list)

            message_new.scatter_(1 , adj_list, torch.log((torch.exp(local + interaction + message) + 1) / (torch.exp(local + message) + 1)))
            #print((torch.exp(local + interaction + message) + 1) / (torch.exp(local + message) + 1))
            message_new.scatter_(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters += 1

        self.conditionals = logistic(torch.sum(message_new.gather(0, torch.transpose(adj_list, 0, 1)), 0))
        self.conditionals[-clamp:] = 0
        self.marginals[-clamp:] = 0
        self.message_old = torch.zeros(n, n)

    def update(self):
        adj = self.adj
        adj_list = self.adj_list
        n = adj.shape[0]
        marginals = self.marginals.unsqueeze(1)
        conditionals = self.conditionals.unsqueeze(1)
        # update model parameters
        update = torch.mm(conditionals, torch.transpose(conditionals, 0, 1)).gather(1, adj_list)
        - torch.mm(marginals, torch.transpose(marginals, 0, 1)).gather(1, adj_list)
        adj.scatter_add_(1, adj_list, self.lr * update)
        adj[(torch.eye(n) != 0)] += self.lr * (conditionals.squeeze(1) - marginals.squeeze(1))

    def expectation(self, data):
        dim = data.shape[0]
        data = data.view(-1)
        mid = data.shape[0] // 2
        self.backward(data[mid:])
        conditionals = self.conditionals[-(mid * 2):-mid]
        print(conditionals)
        return torch.cat((conditionals, data[mid:]), 0).view(dim, dim)
