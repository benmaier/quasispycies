import numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.sparse.linalg as LA
from networkprops import networkprops as nprops

class quasispecies():

    def __init__(self,
                    G,
                    fitness,
                    strain_length = None,
                    use_giant_component = True,
                    normed_by_degree = False,
                    maxiter = None,
                    tol = 0,
                 ):
        """
        Construct the basic necessities to compute the equilibrium quasispecies.


        Parameters
        ----------
            G : networkx.Graph or matrix in a scipy.sparse format
                The mutation network. If G is a sparse matrix, it should
                already consist of a single giant component.
            fitness : numpy.ndarray or function with one argument
                The nodes' fitness. If this a numpy array, it should be of
                the length of number of nodes. Every node i in Graph G
                is associated with ``fitness[i]``.
                If ``fitness`` is a function, it should take the number of nodes
                as an argument and return an array that is the nodes' fitnesses
                (e.g. lambda N: random.gamma(1,1,N)).
            strain_length : float, optional, default: None
                if this argument is passed, the mutation matrix is scaled
                by ``strain_length``. Becomes obsolete if ``normed_by_degree``
                is ``True``.
            use_giant_component : bool, default: True
                Only relevant if ``G`` is a networkx.Graph instance. If True,
                the computation will only be done on the giant component of G.
            normed_by_degree : bool, default: False
                if True, the outgoing entries of the mutation matrix will
                be normed by the degree of the current node.
            maxiter : int, default: None
                maximal number of iterations when computing the maximum 
                eigenvalue and corresponding eigenvector.
            tol : float, default: 0.
                numerical tolerance when computing the maximum 
                eigenvalue and corresponding eigenvector.

        Attributes
        ----------
        A : sprs.csr_matrix
            Adjacency matrix
        N : int
            number of nodes
        m : int
            number of edges
        row_sums : numpy.ndarray
            node degrees
        f : numpy.ndarray
            current fitness vector
        F : scipy.sparse format
            corresponding fitness matrix

        Further attributes once a computation has been done
        ---------------------------------------------------
        phi : float
            quasispecies average fitness
        x : numpy.ndarray of length self.N
            quasispecies strain distribution. Normed as
            sum(x) = 1.
        L : float
            Localization (or 'peakiness') of strain distribution.
            L = x.dot(x)
        """

        if sprs.issparse(G):
            if use_giant_component:
                self.A = self._delete_zero_rows_and_cols(G)
            else:
                self.A = G
            self.N = self.A.shape[0]
            self.m = len(self.A.data) / 2
        else:
            self.nprops = nprops(G,use_giant_component = use_giant_component)
            self.G = self.nprops.G
            self.A = self.nprops.get_adjacency_matrix()
            self.N = self.nprops.N
            self.m = self.nprops.m

        self.row_sums = np.array(self.A.sum(axis=1)).T[0]
        self.normed_by_degree = normed_by_degree

        if strain_length is None:
            # assume N = 2^L nodes with L being the strain length
            self.strain_length = np.log(self.N) / np.log(2.)
        else:
            self.strain_length = strain_length

        self.fitness = fitness
        self.redraw_fitness()

        self.maxiter = maxiter
        self.tol = tol

    def redraw_fitness(self):
        """ 
        Resets the fitness to the initial values.
        If self.fitness is a function, evaluates that 
        function once again.
        """
        if hasattr(self.fitness,"__len__"):
            self.f = np.array(self.fitness)
        else:
            self.f = np.array( self.fitness(self.N) )
        self.F = sprs.diags([self.f],[0])

    def get_mutation_operator(self,mu):
        """ 
        Returns the mutation operator given
        the mutation rate 0 <= mu <= 1.
        """

        if self.normed_by_degree:
            Q = sprs.diags([mu/self.row_sums],[0]).dot(self.A) + sprs.diags([np.array([1.-mu]*self.N)],[0])
        else:
            Q = (self.A * mu + sprs.diags([np.array([self.strain_length*(1.-mu)]*self.N)],[0])) / self.strain_length

        return Q

    def get_mutation_selection_operator(self,mu):
        """ 
        Returns the mutation selection operator given
        the mutation rate 0 <= mu <= 1.
        """

        Q = self.get_mutation_operator(mu)
        W = self.F.dot(Q)

        return W

    def is_fittest_most_abundant(self):
        """ 
        Returns whether or not the fittest strain is the most
        abundant in the last computed quasispecies.
        """
        fittest = np.argmax(self.f)
        most_abundant = np.argmax(self.x)
        return fittest == most_abundant

    def get_quasispecies(self,mu):
        """ 
        Returns the quasispecies given
        the mutation rate 0 <= mu <= 1.

        Returns
        -------
        phi : float
            quasispecies average fitness
        x : numpy.ndarray of length self.N
            quasispecies strain distribution. Normed as
            sum(x) = 1.
        L : float
            Localization (or 'peakiness') of strain distribution.
            L = x.dot(x)
        """

        W = self.get_mutation_selection_operator(mu)
        val,vec = LA.eigs(W.T,k=1,maxiter=self.maxiter,tol=self.tol)
        ndx = np.argmax(val)
        self.phi = np.real(val[ndx])
        self.x = np.real(vec[:,ndx])/np.sum(np.real(vec[:,ndx]))
        self.L = self.x.dot(self.x)

        return self.phi, self.x, self.L
        
    def _delete_zero_rows_and_cols(self,M):
        """
        Given a matrix M, removes all columns and rows
        that only contain zeros and returns the result.
        """
        M = M[M.getnnz(1)>0][:,M.getnnz(0)>0]
        return M


if __name__=="__main__":
    import mhrn
    import pylab as pl
    import seaborn as sns
    sns.set_style("whitegrid")

    B = 8
    L = 5
    k = 10
    xi = 0.4
    
    mu = 0.3

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
