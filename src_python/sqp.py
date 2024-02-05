import numpy as np
import sys
import scipy
import scipy.sparse
import scipy.optimize


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
    
    ---
    ### Attributes 
    - TransitionMatrix
    - States
    - TrstCntsVec
    - ConstraintMatrixC
    - CRowsCount
    - ObjFxn
    - ObjGrad
    - EqnCon
    - EqnJac
    - IneqCon
    - IneqJac
    """    
    
    def __init__(self, observed_sequence, lmbd=1):
        """_summary_

        Args:
            observed_sequence (iterable): some type of iteratble in python. 
            lmbd (int, optional): the regularization panalty term. Defaults to 1.

        Raises:
            TypeError: error is thrown if lambd is not an instance of a number. 
            ValueError: error is thrown when the lmbd is a negative value. 

        Returns:
            None: null. 
        """
        if not isinstance(lmbd, numbers.Number): 
            raise TypeError(f"lmbd passed to __init__ for {type(self)} should be int.")
        if lmbd < 0: 
            raise ValueError(f"lmbd passed to __int__ for {type(self)} should be nonegative. ")
        self.lmbd = lmbd

        transFreqVec, self.TransitionMatrix, self.States = empirical_mle_transmatrix(observed_sequence)
        n = len(self.States)
        self.TransFreqasVec = transFreqVec.reshape(-1)                            # Transition matrix to column vector. 
        C = make_matrix(self.TransitionMatrix, lmbd).toarray()
        self.ConstraintMatrixC = C
        self.CRowsCount = np.size(C, 0)
        self.ObjFxn, self.ObjGrad = make_objective_fxngrad(n, self.TransFreqasVec)
        self.EqnCon, self.EqnJac = make_eqcon_fxnjac(n, self.ConstraintMatrixC)
        self.IneqCon, self.IneqJac = make_ineqcon_fxnjac(C)
        

        return None
    
    def mirror_transmatrix(self):
        """
        Borrow the resource for the transition matrix. Called "mirror" due shallow copying in python. 
        """
        return self.TransitionMatrix

    def mirror_constraint_matrix(self):
        return self.ConstraintMatrixC

    def grad_fxn(self):
        """
        
        Returns:
            callable : the objective function as a callable, it's a shallow borrow. 
        """
        return self.ObjFxn
    
    def idx2state(self):
        return self.States

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




### ================== Optimization Related Implementations =======================
def make_objective_fxngrad(n, trans_freq_as_vec):
    """
    ### Description: 

    This function prepare callable functions that are the objective function and the gradient of the objective function for the problem. 
    These callable functions will be used for the SQP interface of scipy. 

    ---
    ### parameters
    - n: The number of observed states involved in the problem. 
    
    ---
    ### returns : (objfxn:Callable, objfxngrad:Callable)
    - objfxn: A callable function that takes in a numpy array of length: n^2 + Combinatorics(n^2, 2). 
        It returns the objective value of the function. 
    - objfxngrad: A callable function that takes in a numpy array of length: n^2 + Combinatorics(n^2, 2)
        It returns the gradient of the objective function. 
    """
    p_hat = trans_freq_as_vec
    # TODO: Add Expected Length checks here. 
    def ObjectiveMake(x): 
        # x here is literally the transition matrix (in vector form) we are trying to optimize. 
        # This function should get called by scipy.optimize internal code! 
        p = x[0:n**2]
        u = x[n**2 + 1:]
        prd = -p_hat*np.log(p)
        return np.sum(prd) + np.sum(u)
        

    def ObjectiveGrad(x):
        # x here is literally the transition matrix (in vector form) we are trying to optimize. 
        # This function should get called by scipy.optimize internal code! 
        p = x[0:n**2]
        u = x[n**2 + 1:]
        grad = np.hstack((p_hat/p, np.ones_like(u)))
        
        return grad
        
    return ObjectiveMake, ObjectiveGrad



def make_eqcon_fxnjac(n:int, conmtx):
    """
    Make the equality constraints function and the Jacobi of the equality constraints function. 
    
    ---
    ### returns : (:Callable, :Callable)


    """
    C = np.kron(np.eye(n), np.ones(n))
    m = np.size(conmtx, 0) # length of the u decision variables. 
    def Obj(x):
        return np.dot(C, x[:n**2]) - 1
    
    def Jacb(x):
        return np.hstack((C, np.zeros((n, m))))
    
    return Obj, Jacb


def make_ineqcon_fxnjac(conmtx):
    """
    returns : (:Callable, :Callable)
    """
    m, _ = np.shape(conmtx)
    rBlock1st = np.hstack((-conmtx, np.eye(m))) # Pass in as tuples! 
    rBlock2nd = np.hstack((conmtx, np.eye(m)))
    G = np.vstack((rBlock1st, rBlock2nd))

    def Obj(x): 
        return np.dot(G, x)
    def jac(x): 
        return G
    return Obj, jac
    




def main(): 
    global TESTSTRING
    TESTSTRING = "AABB"
    global pbm
    pbm = ProblemModelingSQP(TESTSTRING, lmbd=0.5)
    global ineq_cons
    # Functional representation for constraints are fine 
    ineq_cons = {'type': 'ineq',
        'fun' : pbm.IneqCon,
        'jac' : pbm.IneqJac
        }
    global eq_cons
    eq_cons = {'type': 'eq',
            'fun' : pbm.EqnCon,
            'jac' : pbm.EqnJac
            }
    l = pbm.CRowsCount + len(pbm.TransFreqasVec)
    global bounds
    bounds = scipy.optimize.Bounds(
        np.zeros(l), np.full(l, np.inf)
    )
    # the objective function and gradient 
    objfxn = pbm.ObjFxn
    objgrad = pbm.ObjGrad
    # initial Guess 
    global x0
    C = pbm.ConstraintMatrixC
    x0 = np.eye(len(pbm.States)).reshape(-1)
    x0 = np.hstack((x0, np.dot(C, x0)))

    global res 
    res = scipy.optimize.minimize(
        objfxn, x0, method='SLSQP', jac=objgrad,
        constraints=[eq_cons, ineq_cons], 
        options={'ftol': 1e-9, 'disp': True},
        bounds=bounds
        )
    
    return None


if __name__ == "__main__":
    print(f"Running file at: {__file__}")
    main()