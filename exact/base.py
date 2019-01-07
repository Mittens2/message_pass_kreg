import numpy as np
import pdb
LARGE_NUMBER = 100

class Inference():
    """
        superclass of inference procedures for an Ising model.
        the model is represented using an adjacency matrix adj:
        n x n symmetric adjacency matrix, the diagonal elements are the local potentials:
        p(x) \propto exp(+\sum_{i,j<i} x_i x_j adj[i,j] + \sum_{i} x_i adj[i,i]) for  x_i \in {-1,+1}
    """

    def __init__(self, adj,
                 verbosity=1  # for more info, use higher verbosity level
    ):
        assert np.all(adj == np.transpose(
            adj)), "the adjacency matrix is not symmetric"
        self.adj = adj
        self._verbosity = verbosity

    def get_marginal(
            target=None
    ):
        """
        return the marginal prob. of xi = 1
        for the list of target variables
        --i.e., a vector of marginals p(xi=1)
        """
        pass

    def update_adjacency(self, new_adj):
        self.adj = new_adj

    def incorporate_evidence(self, e):
        """
        evidence is a dictionary of assignments to a subset of variables;
        for example e = {0:+1, 2:-1, 6:-1} means x0 <- 1, x2 <- -1 and x6 <- -1
        this method updates the adjacency matrix to incorporate evidence
        """
        adj = self.adj.astype(float)
        n = adj.shape[0]
        eps=1e-16
        """ your code goes here and calculates a new nxn adjacency matrix """
        for key in e:
            # Set interaction to 0 and add to local bias
            row, col = np.diag_indices(n)
            adj[row, col] += adj[key] * e[key]
            adj[key][adj[key] != 0] = eps
            adj[:, key][adj[key] != 0] = eps
            # Set local of evidence to some high value
            adj[key, key] = 100 * e[key]

        self.update_adjacency(adj)
