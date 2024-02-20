
import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import itertools
import collections


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
        # if C[i] - C[j] == 0: 
        #     # still tells the sparse matrix parser that there is a zero row there. 
        #     row.append(k); col.append(i); data.append(0)
        #     row.append(k); col.append(j); data.append(0)
        # else:
        #     row.append(k); col.append(i); data.append(regl/abs(C[i] - C[j]))
        #     row.append(k); col.append(j); data.append(-regl/abs(C[i] - C[j]))
        # row.append(k); col.append(i); data.append(regl/(abs(C[i] - C[j]) + 1))
        # row.append(k); col.append(j); data.append(-regl/(abs(C[i] - C[j]) + 1))
        row.append(k); col.append(i); data.append(regl)
        row.append(k); col.append(j); data.append(-regl)
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
        RowCounts = np.sum(TransitionCountMatrix, axis=1).reshape((-1, 1))
        return TransitionCountMatrix/np.sum(RowCounts), TransitionCountMatrix/RowCounts, ObservedStates
    else:
        raise TypeError("Argument states/observed is not itertable. ")

