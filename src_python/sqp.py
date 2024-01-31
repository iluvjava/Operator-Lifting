import numpy as np
import sys
import scipy
import scipy.sparse

import itertools
import collections
import numbers



class ProblemModelingSQP:
    """

    Conduct the following: 
    - Maximal likelihood estiamtion for the probability transition matrix given discrete transition states encoded in 
    integers 0, ..., n. it stores this. 
    - Formulate the coefficients lambda*|p_hat(i) - p_hat(j)| for the constraints, it stores this. 
    - Compute necessary modeling informatoin for the Scipy optimize API and stores them. 
        - objective function and gradient for the smooth objective
        - Sparse constraints matrix and the Jacobi of the constraint system. 
    
    """    
    def __init__(self, observed_sequence, lmbd=1):
        if not isinstance(lmbd, numbers.Number): 
            raise TypeError(f"lmbd passed to __init__ for {type(self)} should be int.")
        if lmbd < 0: 
            raise ValueError(f"lmbd passed to __int__ for {type(self)} should be nonegative. ")
        self.lmbd = lmbd
        Tcount, self.TransitionMatrix, self.states = empirical_mle_transmatrix(observed_sequence)
        self.TransitionCounts = Tcount.reshape((-1, 1)) # To column vector. 
        self.ConstraintMatrixC = make_matrix(self.TransitionMatrix, lmbd).todense()
        return None
    
    def mirror_transmatrix(self):
        """
        Borrow the resource for the transition matrix. Called "mirror" due shallow copying in python. 
        """
        return self.TransitionMatrix

    def mirror_constraint_matrix(self):
        return self.ConstraintMatrixC

    def mirror_grad_fxn(self):
        
        pass
    
    def idx2state(self):
        pass

    def state2idx(self):
        pass


def make_matrix(CoeffMtx, regl=1.0):
    """
    Psuedo code for making a numpy sparse matrix. 
    This is for making the constraints matrix for the non-smooth part with the one-norm. 

    parameters
    ---
    CoeffMtx: A matrix that can be indexed by as [i, j], it should be the estimated transition probability matrix via maximum likelihood without the penalization term. 

    """
    if not(type(CoeffMtx) == np.ndarray and CoeffMtx.ndim == 2):
        raise ValueError("Gives MtxCoeff as a 2 Dim numpy.ndarray")
    if np.size(CoeffMtx, 0) != np.size(CoeffMtx, 1): 
        raise ValueError("Expect MtxCoeff as a squared numpy.ndarray matrix ")
    regl = float(regl) # type stablization. 
    if regl < 0: 
        raise ValueError(f"Expect regularization term to be > 0 but get regularization={regl}")
    csr_matrix = scipy.sparse.csr_matrix
    n = np.size(CoeffMtx, 0)
    row, col, data = [], [], []
    C = CoeffMtx.reshape((-1,)) # flatten the coefficient matrix into a vector. 
    for k, (i, j) in enumerate(itertools.combinations(range(n**2), 2)): 
        if C[i] - C[j] == 0: 
            # still tells the sparse matrix parser that there is a zero row there. 
            row.append(k); col.append(0); data.append(0)
        else:
            row.append(k); col.append(i); data.append(regl/abs(C[i] - C[j]))
            row.append(k); col.append(j); data.append(-regl/abs(C[i] - C[j]))
        pass
    return csr_matrix((data, (row, col)), dtype=float)



def empirical_mle_transmatrix(observed): 
    """
    Compute the empirical MLE transition matrix given the total number of availaible states, say n
    and the observed sequence encoded by integers from 0 to n - 1

    parameters
    -----
    observed: Iterable
        An iterable list of observable items. Hashable so they can be compared by python internally. 

    Returns: tuple
    ----
    - (TransitionMatrix, ObservedStates)
        - Transition matrix is the empirical estimated transition matrix estimated from the observed data. 
        - Observed states are all sates that appeared in the data. 
            Their order corresponds to the same index as they were in the matrix. 
            E.g, the first element is the first states, which would be the first element in the array and first row in the retured transition matrix. 

    """
    if issubclass(type(observed), collections.abc.Iterable): 
        observed = list(observed)    
        ObservedStates = list(set(observed))
        States2Idx = dict([(state, idx) for idx, state in enumerate(ObservedStates)])
        ObservedIdx = [States2Idx[state] for state in observed]
        # compute states and make matrix. 
        n = len(ObservedStates)
        TransitionCountMatrix = np.zeros((n, n))
        for (pre, cur) in zip(ObservedIdx[:-1], ObservedIdx[1:]):
            TransitionCountMatrix[pre, cur] += 1
        # normalize by row sum. 
        RowCounts = np.sum(TransitionCountMatrix, axis=0).reshape((1, -1))
        return TransitionCountMatrix, TransitionCountMatrix/RowCounts, ObservedStates
    else:
        raise TypeError("Argument states/observed is not itertable. ")



def make_objective_fxngrad(ncounts):
    """
    ### Description: 

    This function prepare callable functions that are the objective function and the gradient of the objective function for the problem. 
    These callable functions will be used for the SQP interface of scipy. 
    ---

    ### parameters
    ---
    - mtx: The probability transition matrix estimated via simple MLE. 
    - ncounts: a vector of length n^2 denote the total count of i->j transiiton for states. 

    ### returns : (objfxn:Callable, objfxngrad:Callable)
    ---
    - objfxn: A callable function that takes in a numpy array of length: n^2 + Combinatorics(n^2, 2). 
        It returns the objective value of the function. 
    - objfxngrad: A callable function that takes in a numpy array of length: n^2 + Combinatorics(n^2, 2)
        It returns the gradient of the objective function. 
    """
    n = len(ncounts)
    def ObjectiveMake(x): 
        # x here is literally the transition matrix (in vector form) we are trying to optimize. 
        # This function should get called by scipy.optimize internal code! 
        p = x[0:n**2]
        u = x[n**2 + 1:]
        prd = -np.log(p.reshape((-1,))*ncounts.reshape((-1,)))
        return prd + np.sum(u)
        

    def ObjectiveGrad(x):
        # x here is literally the transition matrix (in vector form) we are trying to optimize. 
        # This function should get called by scipy.optimize internal code! 
        p = x[0:n**2]
        u = x[n**2 + 1:]
        return np.vcat(ncounts/p, np.ones_like(u))
        
    return ObjectiveMake, ObjectiveGrad



def make_eqcon_fxnjac(n:int):
    """
    returns : (:Callable, :Callable)
    """

    pass



def make_ineqcon_fxnjac(conmtx):
    """
    returns : (:Callable, :Callable)
    """

    pass



def SQP_solve():
    """
    Given a list of eta_i for the maximal likelihood, and the MLE estimated transition probability matrix P that is n by n, the function formulate the problems for scipy.optimize. 
    """
    pass




def main(): 
    global TESTSTRING
    TESTSTRING = "ABCCBA"
    global pbm
    pbm = ProblemModelingSQP(TESTSTRING)
    return None


if __name__ == "__main__":
    print(f"Running file at: {__file__}")
    main()