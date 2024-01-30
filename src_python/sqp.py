import numpy as np
import sys
import scipy
import scipy.sparse

import itertools
import collections.abc


def main(): 
    make_matrix()
    pass




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
    def __init__(self, n, observed_sequence, lmbd=1):
        self.lmbd = 1
        
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
        row.append(k); col.append(i); data.append(regl/abs(C[i] - C[j]))
        row.append(k); col.append(j); data.append(-regl/abs(C[i] - C[j]))
        pass
    return csr_matrix((data, (row, col)), dtype=float)


"""
Compute the empirical MLE transition matrix given the total number of availaible states, say n
and the observed sequence encoded by integers from 0 to n - 1

parameters
-----
n: integers. 
"""
def empirical_mle_transmatrix(n, observed): 
    if not isinstance(n, int): 
        raise TypeError("argument n is not an instance of integers. ")
    if n <= 0: 
        raise ValueError(f"the argument n should be strictly larger than zero but n={n}")
    if issubclass(type(observed), collections.abc.Iterable): 
        TransitionMatrix = np.array((n, n))
        

        pass
    else:
        raise TypeError("Argument states/observed is not itertable. ")


"""
Given a list of eta_i for the maximal likelihood, and the MLE estimated transition probability matrix P that is n by n, the function formulate the problems for scipy.optimize. 
"""
def formulate_SQP():
    pass


if __name__ == "__main__":
    print(f"Running file at: {__file__}")
    main()