import numpy as np
import pdb
from base import Inference
from ex_utils import *

class BruteForceInference(Inference):
    """
    represents bruete-force inference by
    producing the high-dimensional joint tensor
    """
    def __init__(self, adj, verbosity=1):
        Inference.__init__(self, adj, verbosity)
        self.joint = self._get_joint()

    def _get_joint(self):
        """
        builds the n-dim tensor joint (size 2 x 2 x ... x 2)
        returns the joint probability tensor that has n-dimensions
        """
        adj = self.adj
        n = adj.shape[0]
        if self._verbosity > 0:
            print("producing the joint dist over %d variables" % (n), flush=True)
        pairwise = np.zeros([2, 2])
        local = np.zeros([2])
        joint = np.ones(n*[2], dtype=np.float)
        for i, j in np.transpose(np.nonzero(adj)):
            if i < j:
                pairwise[0, 0] = np.exp(adj[i, j])
                pairwise[1, 1] = pairwise[0, 0]
                # pairwise[0, 1] = np.exp(-adj[i, j])
                pairwise[0, 1] = 1
                pairwise[1, 0] = pairwise[0, 1]
                joint = tensor_mult(joint, pairwise, [i, j], [0, 1])
            elif i == j:
                # local[0] = np.exp(-adj[i, j])
                local[0] = 1
                local[1] = np.exp(+adj[i, j])
                joint = tensor_mult(joint, local, [i], [0])
        return joint

    def get_marginal_multivar(self, target):
        """
        target: a list of length 1 \leq k \leq n
        returns a k-dim tensor 'marg' of marginals over the target,
        where marg[1,0,...,1] = \tilde{p}(target[0]=1, target[1]=0,...,target[k]=1)
        """
        joint = self.joint
        n = joint.ndim
        if self._verbosity > 0:
            print("producing the joint marginals over %d target variables" % (len(target)), flush=True)
        assert np.all([i < n for i in target]), "targets are invalid"
        marg_inds = [i for i in range(n) if i not in target]
        marg = joint.sum(axis=tuple(marg_inds))
        order = np.unique(target, return_index=True)[1]
        marg = np.transpose(marg, order)
        return marg

    def get_marginal(self, target):
        marg = self.get_marginal_multivar(target)
        if self._verbosity > 0:
            print("reducing the joint marginals to univar. marginals", flush=True)
        marg /= np.sum(marg)
        p1 = []
        target_set = set(target)
        all_inds = set(range(len(target)))
        for v in range(len(target)):
            p1.append(np.sum(marg, axis=tuple(all_inds-{v}))[1])
        return np.array(p1)

    def update_adjacency(self, new_adj):
        self.adj = new_adj
        self.joint = self._get_joint()

    def get_normalization_constant(self):
        """ return the normalization constant"""
        if self._verbosity > 0:
            print("summing the joint distribution", flush=True)
        return np.sum(self.joint)
