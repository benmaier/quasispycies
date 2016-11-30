import numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.sparse.linalg as LA
from networkprops import networkprops as nprops

class quasispecies(nprops):

    def __init__(self,G,fitness,strain_length=None,use_giant_component=True,normed_by_degree=False,maxiter=None,tol=0):

        self.nprops = nprops(G,use_giant_component = use_giant_component)
        self.G = self.nprops.G
        self.A = self.nprops.get_adjacency_matrix()
        self.N = self.nprops.N
        self.m = self.nprops.m

        self.row_sums = np.array(self.A.sum(axis=1)).T[0]
        self.normed_by_degree = normed_by_degree

        if strain_length is None:
            self.strain_length = self.nprops.N + 1
        else:
            self.strain_length = strain_length

        self.fitness = fitness
        self.redraw_fitness()

        self.maxiter = maxiter
        self.tol = tol

    def redraw_fitness(self):
        if hasattr(self.fitness,"__len__"):
            self.f = np.array(self.fitness)
        else:
            self.f = np.array( self.fitness(self.N) )
        self.F = sprs.diags([self.f],[0])

    def get_operator(self,mu):

        if self.normed_by_degree:
            Q = sprs.diags([mu/self.row_sums],[0]).dot(self.A) + sprs.diags([np.array([1.-mu]*self.N)],[0])
            factor = 1.
            print "normed by degree"
        else:
            Q = self.A * mu + sprs.diags([np.array([self.strain_length*(1.-mu)]*self.N)],[0])
            factor = 1./self.strain_length
            print "normed with strain length"

        W = self.F.dot(Q) * factor

        return W

    def get_quasispecies(self,mu):

        W = self.get_operator(mu)
        val,vec = LA.eigs(W.T,k=1,maxiter=self.maxiter,tol=self.tol)
        ndx = np.argmax(val)
        self.phi = np.real(val[ndx])
        self.x = np.real(vec[:,ndx])/np.sum(np.real(vec[:,ndx]))
        self.L = self.x.dot(self.x)

        return self.phi, self.x, self.L
        

if __name__=="__main__":
    import mhrn
    import pylab as pl
    import seaborn as sns
    sns.set_style("whitegrid")

    B = 8
    L = 5
    k = 10
    xi = 0.4
    
    mu = 0.9

    N = B**L
    strain_length = np.log(N)/np.log(2.)
    
    G = mhrn.fast_mhr_graph(B,L,k,xi)

    fitness = lambda N: np.random.gamma(1,1,N)

    fig, ax = pl.subplots(2,1)

    QS = quasispecies(G,fitness,strain_length=1.,tol=1e-6,maxiter=100*B**L)
    phi,x,L = QS.get_quasispecies(mu)
    ax[0].plot(np.arange(QS.N),QS.f)
    ax[1].plot(np.arange(QS.N),x)

    print phi,x,L

    QS = quasispecies(G,fitness=QS.f,normed_by_degree=True,tol=1e-6,maxiter=100*B**L)
    phi,x,L = QS.get_quasispecies(mu)
    ax[0].plot(np.arange(QS.N),QS.f)
    ax[1].plot(np.arange(QS.N),x)


    print phi,x,L
    

    pl.show()
