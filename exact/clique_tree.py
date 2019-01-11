import numpy as np
from ex_utils import tensor_mult, draw_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import pdb
from base import Inference
from itertools import product
from math import factorial


class CliqueTreeInference(Inference):
    """
    implements exact inference using clique-tree
    """

    def __init__(self, adj,
                 verbosity=1,
                 #normalizing messages helps with overflow in large models,
                 normalize_messages=True,

    ):
        Inference.__init__(self, adj, verbosity=verbosity)
        self._normalize_messages = normalize_messages
        self.cliques = None  # a list of lists
        # a 0-1 matrix wehre ctree[i,j] > 0 means cliques[i] and cliques[j] are connected in the clique-tree
        self.ctree_adj = None
        self.chordal = None  # the 0-1 chordal graph
        self.order = None  # the elimination order
        # dictionary identifying the parent of each node in the ctree with root=0
        self.parents = None
        # dictionary identifying the children of each node in the ctree with root=0
        self.children = None
        # dictionary of messages sent from clique i->j (i,j):msg, where the variables in the sepset appear in their natural order in the msg tensor
        self.messages = {}
        # a list of tensors, corresponding to clique potentials (i.e., product of all factors associated with a clique)
        self.clique_potentials = None
        # a dict of tensors, corresponding to the marginal beliefs over cliques
        self.clique_beliefs = {}
        # indicates whether the message updates have been sent or not
        self._calibrated = False
        # build the clique-tree from the adjacency matrix
        self._build_clique_tree()


    def update_adjacency(self, new_adj):
        self.adj = new_adj
        self.ctree_adj = None
        self.order = None
        self.chordal_adj = None
        self.messages = {}
        self.parents = None
        self.children = None
        self.cliques = None
        self.clique_potentials = None
        self.clique_beliefs = {}
        self._calibrated = False
        self._build_clique_tree()

    def _min_fill(self):
        """
        return the elimination order (a list of indices) as well
        as the resulting chordal graph baded on min_fill heuristic
        chordal_adj[i,j] = 1 iff (i,j) are connected.
        The diagonal of chordal_adj should be zero.
        """
        adj = self.adj
        n = adj.shape[0]
        chordal_adj = np.copy(adj).astype(float)
        np.fill_diagonal(chordal_adj, 0)
        chordal_adj[chordal_adj != 0] = 1
        order = list()
        vertices = list(range(n))
        while len(vertices) > 0:
            # Only care about rows and columns of remaining vertices
            elim_adj = chordal_adj[np.ix_(vertices, vertices)]
            # Want to create a fill edge between every pair of nodes that are not connected, but have a neighbouring node in common
            fill_graph = elim_adj * (elim_adj@(1 - elim_adj))
            v = np.argmin(np.sum(fill_graph, axis=1))
            fill_edges = np.where(fill_graph[v] != 0)[0]
            elim_adj[np.ix_(fill_edges, fill_edges)] = 1
            chordal_adj[np.ix_(vertices, vertices)] = elim_adj
            np.fill_diagonal(chordal_adj, 0)
            order.append(vertices[v])
            vertices.remove(vertices[v])

        return order, chordal_adj

    def _max_cardinality_search(self, mask):
        """
        mask is the adjacency matrix for 0-1 chordal graph
        this method returns a list of lists: the set of maximal cliques
        we can also return sep-sets here, but we do that using max-spanning-tree later
        """
        n = mask.shape[0]
        cliques = [[]]  # maintains the list of cliques
        last_mark = -1  # number of marked neighbors for prev. node
        marks = [[] for i in range(n)]  # a set tracking the marked neighbors of each node
        mark_size = np.zeros(n)  # number of marked neighbors for each node
        remaining = list(range(n))
        for _ in reversed(range(n)):
            node = remaining[np.argmax(mark_size[remaining])]
            if mark_size[node] <= last_mark:  # moving into a new clique
                cliques.append(marks[node] + [node])
            else:  # add it to the last clique
                cliques[-1].append(node)
            nb_node = np.nonzero(mask[node,:])[0]  # neighbors of node
            for nb in nb_node:  # update the marks for neighbors
                marks[nb].append(node)
                mark_size[nb] += 1
            last_mark = mark_size[node]
            remaining.remove(node)
        sorted_cliques = [sorted(c) for c in cliques]
        return sorted_cliques

    def _get_directed_tree(self, adj, root=0):
        """
        produce a directed tree from the adjacency matrix, with the given root
        return a dictionary of children and parents for each node
        """
        visited = set()
        to_visit = set([0])
        n = adj.shape[0]
        rest = set(range(1, n))
        parents = {root:None}
        children = {}
        while len(to_visit) > 0:
            current = to_visit.pop()
            nexts = set(np.nonzero(adj[current, :])[0]).intersection(rest)
            for j in nexts:
                parents[j] = current
            children[current] = frozenset(nexts)
            to_visit.update(nexts)
            rest.difference_update(nexts)
            visited.add(current)
        assert len(rest) == 0, "the clique tree is disconnected!"
        return parents, children

    def _calc_clique_potentials(self, cliques):
        """
        calculate the potential/factor for each clique
        as the product of factors associated with it.
        Note that each local and pairwise factor is counted within
        a single clique (family-preserving property)
        the method receives a list of lists of variable indices, that define the cliques.
        It should return a list of tensors. if cliques[2].ndim == 3,
        then the clique_potentials[2]
        should be a 3-dimensional (2 x 2 x 2) tensor.
        """
        adj = self.adj
        n = adj.shape[0]
        local = np.copy(np.diagonal(adj))
        inter = np.copy(adj).astype(float)
        np.fill_diagonal(inter, 0)
        c = len(cliques)
        clique_potentials = list()
        x = np.zeros(n)
        # Iterate over all possible -1, 1 assignments to variables in clique to get potentials
        for clique in cliques:
            potential = np.zeros(tuple([2]*len(clique)))
            for assign in product({0, 1}, repeat=len(clique)):
                x[clique] = assign
                ex = 0.5 * x@inter@x.T + local@x
                assign = [max(0, a) for a in assign]
                potential[tuple(assign)] = np.exp(ex)
                x[clique] = 0
            clique_potentials.append(potential)

        return clique_potentials

    def _build_clique_tree(self):
        """
        builds the clique-tree from the adjacency matrix by
        1. triangulating the graph to get a chordal graph
        2. find the maximal cliques in the chordal graph
        3. calculating the clique-potentials
        4. selecting the sepsets using max-spanning tree
        5. selecting a root node and building a directed tree
        this method does not calibrate the tree -- i.e., it doesn't
        send the messages upward and downward in the tree
        """
        graph='path'
        if self._verbosity > 0:
            print("building the clique-tree ...", flush=True)
        order, chordal_adj = self._min_fill()
        if self._verbosity > 1:
            print("\t found the elimination order {}".format(order), flush=True)
            chordal_viz = chordal_adj + (self.adj != 0)  # so that the color of the chords is different
            np.fill_diagonal(chordal_viz, 0)
            draw_graph(chordal_viz, draw_edge_color = True, title='2_chordal_' + graph)
        cliques = self._max_cardinality_search(chordal_adj)
        if self._verbosity > 1:
            print("\t number of maximal cliques: {} with max. size: {}".format(len(cliques),
                max([len(c) for c in cliques])), flush=True)
            labels = [[c for c in range(len(cliques)) if i in cliques[c]] for i in range(self.adj.shape[0])]
            draw_graph(chordal_viz, draw_edge_color = True, title='2_nodes_' + graph, node_labels= labels)
        if self._verbosity > 1:
            print("\t calculating clique potentials")
        # assign each factor (pairwise or local) to a clique and calculate the clique-potentials
        clique_potentials = self._calc_clique_potentials(cliques)
        # find the size of septsets between all cliques and use max-spanning tree to build the clique-tree
        sepset_size = np.zeros((len(cliques), len(cliques)))
        for i, cl1 in enumerate(cliques):
            for j, cl2 in enumerate(cliques):
                if i != j:
                    sepset_size[i, j] = max(len(set(cl1).intersection(cl2)), .1)
        if self._verbosity > 1:
            print("\t finding the max-spanning tree", flush=True)
        # use scipy for max-spanning-tree
        ctree = minimum_spanning_tree(
            csr_matrix(-sepset_size)).toarray().astype(int)
        # make it symmetric
        ctree_adj = (np.maximum(-np.transpose(ctree.copy()), -ctree) > 0)
        if self._verbosity > 1:
            draw_graph(ctree_adj, title='2__clique-tree_' + graph, node_size=600, node_labels=cliques)
        # set the first cluster to be the root and build the directed tree
        root = 0
        parents, children = self._get_directed_tree(ctree_adj, root)
        self.parents = parents
        self.children = children
        self.chordal_adj = chordal_adj
        self.cliques = cliques
        self.ctree_adj = ctree_adj
        self.clique_potentials = clique_potentials
        if self._verbosity > 0:
            print("... done!", flush=True)

    def _calc_message(self,
                      src_node,  # source of the message
                      dst_set,  # a set of destinations,
                      upward,
    ):
        """
        if the downward, the message to all destinations is calculated by first
        obtaining the belief and dividing out the corresponding incoming messages
        This assumes that the distribution is positive and therefore messages are never zero.
        during the upward pass there is only a single destination and the message is obtained directly
        should also work when the dst_set is empty (producing belief over the leaf nodes)
        """
        # incoming messages are from these clusters
        incoming = set(self.children[src_node])
        if self.parents[src_node] is not None:
            incoming.add(self.parents[src_node])
        if upward:
            incoming.difference_update(dst_set)  # only has one destination
            assert len(dst_set) == 1, "should have a single receiver in the upward pass!"
        factor = self.clique_potentials[src_node].copy()
        clique_vars = self.cliques[src_node]
        for r in incoming:
            sepset = list(set(self.cliques[r]).intersection(set(clique_vars)))
            # find the index of sepset in the clique potential
            inds = sorted([clique_vars.index(i) for i in sepset])
            # multiply with the incoming message from the child
            factor = tensor_mult(factor, self.messages[(r,src_node)], inds, list(range(len(sepset))))
        for dst_node in dst_set:
            tmp_factor = factor.copy()
            if not upward:  # divide out the incoming message to produce the outgoing message
                sepset = set(self.cliques[dst_node]).intersection(set(clique_vars))
                # find the index of sepset in the clique potential
                inds = sorted([clique_vars.index(i) for i in sepset])
                # multiply with the incoming message from the child
                tmp_factor = tensor_mult(tmp_factor, 1./self.messages[(dst_node,src_node)], inds, list(range(len(sepset))))
            outgoing_vars = set(clique_vars).intersection(set(self.cliques[dst_node]))
            sum_over_vars = set(clique_vars) - set(outgoing_vars)
            sum_over_vars_inds = sorted([clique_vars.index(i) for i in sum_over_vars])
            msg = np.sum(tmp_factor, axis=tuple(sum_over_vars_inds))
            if self._normalize_messages:
                msg /= np.sum(msg)
            self.messages[(src_node,dst_node)] = msg
            if self._verbosity > 2:
                print("{} -> ({})-> {}".format(clique_vars, outgoing_vars ,self.cliques[dst_node]), flush=True)
        return factor  # is used to set the clique-marginals in the downward pass

    def _upward(self, root=0):
        """
        send the message from leaf nodes towards the root
        each node sends its message as soon as received messages from its children
        """
        if self._verbosity > 0:
            print("sending messages towards the root node", end="", flush=True)
        # leaf nodes
        ready_to_send = set([node for node, kids in self.children.items() if len(kids) == 0])
        #until root receives all its incoming messages
        while root not in ready_to_send:
            if self._verbosity > 0:
                print(".", end="", flush=True)
            current = ready_to_send.pop()
            # send the message to the parent
            parent = self.parents[current]
            self._calc_message(current, {parent}, True)
            #if the parent has received all its incoming messages, add it to ready_to_send
            parent_is_ready = np.all([((ch,parent) in self.messages.keys()) for ch in self.children[parent]])
            if parent_is_ready: ready_to_send.add(parent)
        if self._verbosity > 0:
            print("", end="\n", flush=True)

    def _downward(self, root=0):
        """
        send the messages downward from the root
        each node sends its message to its children as soon as received messages from its parent
        """

        if self._verbosity > 0:
            print("sending messages towards the leaf nodes", end="", flush=True)
        ready_to_send = set([root])
        while len(ready_to_send) > 0:
            current = ready_to_send.pop()
            self.clique_beliefs[current] = self._calc_message(current, self.children[current], False)
            ready_to_send.update(self.children[current])
            if self._verbosity > 0:
                print(".", end="", flush=True)
        if self._verbosity > 0:
            print("", end="\n", flush=True)

    def get_marginal(self, target):
        """
        return the marginal prob. of xi = 1
        for the list of target variables
        --i.e., a vector of marginals p(xi=1)
        these are calculated using clique_beliefs
        """
        if not self._calibrated:  # we haven't sent the messages yet
            self._upward()
            self._downward()
            self._calibrated = True

        if self._verbosity > 0:
            print("calculating the marginals for {} target variables".format(len(target)), flush=True)

        target_set = set(target)
        p1 = {}
        for c, clique in enumerate(self.cliques):
            cl_var_set = set(clique)
            for v in target_set.intersection(cl_var_set):
                v_ind = clique.index(v)
                summation_inds = list(set(range(len(cl_var_set))).difference({v_ind}))
                mrg = np.sum(self.clique_beliefs[c], axis=tuple(summation_inds))
                mrg /= np.sum(mrg)
                p1[v] = mrg[1]
        p1_arr = np.array([p1[v] for v in target])
        return p1_arr
