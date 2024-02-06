import numpy as np
import numpy.linalg
import sys
import scipy
import scipy.sparse
import scipy.optimize

import itertools
import collections
import numbers
from datetime import datetime



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
            row.append(k); col.append(i); data.append(0)
            row.append(k); col.append(j); data.append(0)
        else:
            row.append(k); col.append(i); data.append(regl/abs(C[i] - C[j]))
            row.append(k); col.append(j); data.append(-regl/abs(C[i] - C[j]))
        # row.append(k); col.append(i); data.append(regl/(abs(C[i] - C[j]) + 1))
        # row.append(k); col.append(j); data.append(-regl/(abs(C[i] - C[j]) + 1))
        # row.append(k); col.append(i); data.append(regl)
        # row.append(k); col.append(j); data.append(-regl)
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
    NumTol = 1e-10
    # helps computing x1*log(x2), handles things like 0*log(0) = 0
    valueErrorMessage = """
        View source code please. 
        The error is hard to describe. 
        we expect some conditiosn and it didn't happen.
    """
    def LogProdHelper(x1, x2): 
        if (not x1 == 0) and (x2 <= -NumTol): # not in domain! 
            raise ValueError(valueErrorMessage)
        # Remaining cases: 
        # - x1 != 0 and x2 != 0
        # - x1 == 0 and x2 == 0 
        # - x1 == 0 and x2 != 0
        if x1 == 0 and x2 == 0: 
            return 0
        return x1*np.log(x2)
        
    # helps computing derivative of x1*log(x2) wrt x2. 
    def GradDivHelper(x1, x2): 
        if (not x1 == 0) and (x2 <= - NumTol): # not in domain! 
            raise ValueError(valueErrorMessage)
        
        if x1 == 0 and x2 == 0:
            return 0
        
        return x1/x2
        
    # TODO: Add Expected Length checks here. 
    
    def ObjectiveMake(x): 
        # x here is literally the transition matrix (in vector form) we are trying to optimize. 
        # This function should get called by scipy.optimize internal code! 
        p = x[0:n**2]
        u = x[n**2:]
        objFxnVal = np.sum(
            - np.array(
                    list(map(LogProdHelper, p_hat, p))
                )
            ) + np.sum(u)
        # print(f"[{datetime.utcnow().strftime('%F %T.%f')[:-3]}] Objective Fxn Value = {objFxnVal}")

        return objFxnVal
        
    def ObjectiveGrad(x):
        # x here is literally the transition matrix (in vector form) we are trying to optimize. 
        # This function should get called by scipy.optimize internal code! 
        p = x[0:n**2]
        u = x[n**2:]
        grad = np.hstack(
                (
                    - np.array(
                        list(map(GradDivHelper, p, p_hat))
                    ),
                    np.ones_like(u)
                )
            )
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
        result = np.dot(C, x[:n**2]) - 1
        return result
    
    def Jacb(x):
        result = np.hstack((C, np.zeros((n, m))))
        return result
    
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
        result = np.dot(G, x)
        return result
    
    def jac(x): 
        return G
    
    return Obj, jac
    

def sqp():
    pass


def main(): 
    global TESTSTRING
    TESTSTRING = TESTSTRING = "02002102000101000000102002110000021102201100000121101100102200102212002011112010100202020220021011110001020002102020211001102210020111001100000102100022100110201210022100020000101002210202100021000220220211202211110002221011010211000211202201021002102200012201101110222110002022012210202020020102100202211110202001122020000110020222220022110010020002102120002010010000211002021102102121210202221122000110202101020002020022200021000211020211022210200121022200010211002201101110220220110202110202210020212102102120002210002202112110210020001010002002000202102121222022121022201210211202020022100222101102112100221202021001010211020210102110202211200202000000000022102020000021111220012110201121010002002020000120200222022110202011002101002110010002120221100011000002100220222202021110222102200022001101011122021021111120021100010210222100222110202210102002221000021202020210200201101001120002211121011000212002000122022200121011120000210111011111020112221002002202"
    global pbm
    pbm = ProblemModelingSQP(TESTSTRING, lmbd=1)
    n = len(pbm.States)
    global ineq_cons
    # Functional representation of Constraints =================================
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
        np.zeros(l) + 1e-10, np.full(l, np.inf)
    )
    # the objective function and gradient ======================================
    objfxn = pbm.ObjFxn
    objgrad = pbm.ObjGrad
    # initial Guess 
    global x0
    C = pbm.ConstraintMatrixC
    x0 = pbm.TransitionMatrix.reshape(-1)
    x0 = np.hstack((x0, np.dot(C, x0) + 1)).copy()

    global res 
    res = scipy.optimize.minimize(
            objfxn, x0, method='SLSQP', jac=objgrad,
            constraints=[eq_cons, ineq_cons], 
            options={'ftol': 1e-14, 'disp': True, 'maxiter': 40},
            bounds=bounds, 
        )
    print("The best estimate of the solution matrix is:")
    global M
    M = res.x[:n**2].reshape((n, n))
    print(M)
    return M


if __name__ == "__main__":
    print(f"Running file at: {__file__}")
    main()