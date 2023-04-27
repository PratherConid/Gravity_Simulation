from physics.base import *
from mathematics.base import tally_sum


# Force described by a hypergraph. Vertices are objects.
class GraphPositional(Physics):

    # force_f(post : tuple(dim * N), propt : tuple(property)) -> force : tuple(dim * N)
    #   The graph is a k-hypergraph
    #   post = position(s) of particle 1 in the edge, ..., position(s) of particle k in the edge
    #   propt = properties of particle 1 in the edge, ..., properties of particle k in the edge
    #   returns : force(s) experienced by particle 1 in the edge, ..., force(s) experienced by particle k in the edge
    # adj : E * k
    #   E is the number of hyperedges
    #   The graph is a k-hypergraph
    # There is no key-r
    def __init__(self, key_p : str, force_f, adj : torch.Tensor):
        super().__init__('a')
        self.key_p = key_p
        self.force_f = force_f
        self.adj = adj

    def a(self, sim):
        dim      : int          = sim.dim
        pos      : torch.Tensor = sim.pos
        prop     : torch.Tensor = sim.pr[self.key_p]
        adj      : torch.Tensor = self.adj.to(torch.int64)
        graph_k  : int          = adj.shape[1]
        post     : tuple = ()
        propt    : tuple = ()
        for i in range(graph_k):
            post = post + (pos[:, adj[:, i]],)
            propt = propt + (prop[adj[:, i]],)
        forces = self.force_f(post, propt)
        all_forces = torch.concat(forces, 1).transpose(0, 1)
        actee = adj.transpose(0, 1).flatten()
        indice, summed = tally_sum(actee, all_forces)
        ret      : torch.tensor = torch.zeros((dim, sim.N)).cuda()
        ret[:, indice] = summed.transpose(0, 1)
        return ret

    def dv(self, sim):
        return self.a(sim) * sim.dt
    
# Force described by a hypergraph. Vertices are objects.
class GraphPosVel(Physics):

    # force_f(post : tuple(dim * N), velt : tuple(dim * N), propt : tuple(property)) -> force : tuple(dim * N)
    #   The graph is a k-hypergraph
    #   post = position(s) of particle 1 in the edge, ..., position(s) of particle k in the edge
    #   velt = velocity(s) of particle 1 in the edge, ..., velocity(s) of particle k in the edge
    #   propt = properties of particle 1 in the edge, ..., properties of particle k in the edge
    #   returns : force(s) experienced by particle 1 in the edge, ..., force(s) experienced by particle k in the edge
    # adj : E * k
    #   E is the number of hyperedges
    #   The graph is a k-hypergraph
    # There is no key-r
    def __init__(self, key_p : str, force_f, adj : torch.Tensor):
        super().__init__('a')
        self.key_p = key_p
        self.force_f = force_f
        self.adj = adj

    def a(self, sim):
        dim      : int          = sim.dim
        pos      : torch.Tensor = sim.pos
        vel      : torch.Tensor = sim.vel
        prop     : torch.Tensor = sim.pr[self.key_p]
        adj      : torch.Tensor = self.adj.to(torch.int64)
        graph_k  : int          = adj.shape[1]
        post     : tuple = ()
        velt     : tuple = ()
        propt    : tuple = ()
        for i in range(graph_k):
            post = post + (pos[:, adj[:, i]],)
            velt = velt + (vel[:, adj[:, i]],)
            propt = propt + (prop[adj[:, i]],)
        forces = self.force_f(post, velt, propt)
        all_forces = torch.concat(forces, 1).transpose(0, 1)
        actee = adj.transpose(0, 1).flatten()
        indice, summed = tally_sum(actee, all_forces)
        ret      : torch.tensor = torch.zeros((dim, sim.N), dtype=all_forces.dtype).cuda()
        ret[:, indice] = summed.transpose(0, 1)
        return ret

    def dv(self, sim):
        return self.a(sim) * sim.dt