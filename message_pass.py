import torch
import numpy as np
from utils import logistic, time

class SparseMP():

    def __init__(self, train_set, adj, adj_list=None, lr=0.3, damping=0.1, eps=1e-16, epochs=10, max_iters=20):
        torch.manual_seed(0)
        self.train_set = train_set
        n = adj.shape[0]
        self.max_iters = max_iters
        self.lr = lr
        self.damping = damping
        self.eps = eps
        self.epochs = epochs
        # TODO: since adjacency matrix is being passed, only allowing for interraction terms
        adj = (torch.from_numpy(adj).type(torch.FloatTensor) * (torch.rand(n , n) - 0.5))
        self.adj = adj * torch.transpose(adj, 0, 1)
        self.adj_list = torch.from_numpy(adj_list).type(torch.LongTensor)
        self.inference()

    def inference(self):
        i = 0
        while i < self.epochs:
            data, label = self.train_set[0]
            self.forward()
            self.backward(data.view(-1))
            self.update(data.view(-1))
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

            message_new.scatter_(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters+=1

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

        message_old = self.message_old.clone()
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

            message_new.scatter_(1, adj_list, message_new.gather(1, adj_list) * self.damping
            + (1 - self.damping) * message_old.gather(1, adj_list))

            iters += 1

        self.message_new = message_new

    def update(self, data):
        adj = self.adj
        adj_list = self.adj_list
        n = adj.shape[0]
        k = adj_list.shape[1]
        message_old = self.message_old
        message_new = self.message_new

        # Update interaction terms
        local_i = torch.diag(adj).unsqueeze(1).expand(-1, k)
        local_j = torch.diag(adj).expand(n, -1).gather(1, adj_list)
        interaction = adj.gather(1, adj_list)
        msg_joint_free_i = torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0).unsqueeze(1).expand(-1, k) - torch.transpose(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
        msg_joint_free_j = torch.sum(message_old.gather(0, torch.transpose(adj_list, 0, 1)), 0).unsqueeze(0).expand(n, -1).gather(1, adj_list) - message_old.gather(1, adj_list)
        prob_free = torch.exp(local_i + local_j + interaction + msg_joint_free_i + msg_joint_free_j) / (torch.exp(local_i + local_j + interaction + msg_joint_free_i + msg_joint_free_j) + torch.exp(local_i + msg_joint_free_i) + torch.exp(local_j + msg_joint_free_j) + 1)

        # Add a fake unit that sends a message of positive or negative infinity if unit is clamped on/off
        adj_clamp = adj_list.clone()
        adj_clamp = torch.cat((adj_list, torch.ones(n, dtype=torch.long).unsqueeze(1)), 1)
        message_new = torch.cat((message_new, torch.cat((torch.zeros(n - data.shape[0]), (data * 2 - 1) * 100)).unsqueeze(0)), 0)

        msg_joint_clamp_i = torch.sum(message_new.gather(0, torch.transpose(adj_clamp, 0, 1)), 0).unsqueeze(1).expand(-1, k)
        - torch.transpose(message_new.gather(0, torch.transpose(adj_list, 0, 1)), 0, 1)
        msg_joint_clamp_j = torch.sum(message_new.gather(0, torch.transpose(adj_clamp, 0, 1)), 0).unsqueeze(0).expand(n, -1).gather(1, adj_list)
        - message_new[:-1, :].gather(1, adj_list)
        prob_clamp = torch.exp(local_i + local_j + interaction + msg_joint_clamp_i + msg_joint_clamp_j) / (torch.exp(local_i + local_j + interaction + msg_joint_clamp_i + msg_joint_clamp_j) + torch.exp(local_i + msg_joint_clamp_i) + torch.exp(local_j + msg_joint_clamp_j) + 1)

        adj.scatter_add_(1, adj_list, self.lr * (prob_clamp - prob_free))

        # Update local terms
        marginals = logistic(torch.sum(torch.gather(message_old, 0, torch.transpose(adj_list, 0, 1)), 0))
        conditionals = logistic(torch.sum(message_new.gather(0, torch.transpose(adj_clamp, 0, 1)), 0))
        adj[(torch.eye(n) == 1)] += self.lr * (conditionals - marginals)

    def gibbs(self, data):
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
        while not torch.eq(x, x_new).all() and i < 100:
            x = x_new.clone()
            #print(x[-data.view(-1).shape[0]:].view(data.shape[0], -1))
            x_new = torch.round(logistic(torch.mm(adj, x) + local))
            i += 1
            print(i)
        x = x.squeeze(1)
        # Return the state of the visible units
        return x[-data.view(-1).shape[0]:].view(data.shape[0], -1)
