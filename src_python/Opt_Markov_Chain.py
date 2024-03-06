import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import Counter
from itertools import product
from itertools import combinations
np.random.seed(10086)

# Markov Chain's states
M = 3

# Length of Sequence
N = 1000
LAMBDA = 1.2
GAMMA = 1


# Set the true p_ij
p_ij_true = np.array([
    [0.4, 0.2, 0.4],
    [0.5, 0.37, 0.13],
    [0.4, 0.34, 0.26]
])

# McSequence function
def McSimulate(p_ij, length):
    data = [np.random.choice([0, 1, 2], size=1, p=[1 / 3, 1 / 3, 1 / 3])[0]]
    for i in range(length - 1):
        data.append(np.random.choice([0, 1, 2], size=1, p=p_ij[data[i]])[0])
    return data

# TransitionCounts function
def TransitionCounts(T, M):
    pairs: list[str] = [str(T[i]) + str(T[i + 1]) for i in range(len(T) - 1)]
    pairs_order = [str(i) + str(j) for i, j in product(range(M), repeat=2)]
    iter = Counter(pairs)
    iter = [iter[pair] for pair in pairs_order]
    return iter

# MLE (p_ij_hat_matrix) function
def mle(N):
    p_ij_hat_matrix = np.round(N / N.sum(axis=1, keepdims=True),3)
    return p_ij_hat_matrix

## Transition Probability Difference function
def prob_diff(z):
    pairs= list(combinations(z,2))
    w=[]
    for pair in pairs:
        diff = abs(pair[0]-pair[1])
        w.append(diff)
    return np.array(w)

# Penalty fraction function
def penalty_function(numerator,denomiator):
    pelt = numerator/denomiator
    pelt[denomiator==0]=0 ## IF the denomiator is 0, then the whole penalty term force to 0
    return pelt

### Find Pair of transition that p_ij_hat equal Function
def equal_pij_hat(gap):
    pairs = list(combinations(range(M**2), 2))
    pairs = pd.DataFrame(pairs, columns=["V1", "V2"], index=None)
    pairs['diff'] = gap
    pairs = pairs[pairs['diff'] == 0]
    equal_pair = pd.DataFrame({'V1': pairs['V1'], 'V2': pairs['V2']})
    E1 = np.array(equal_pair['V1'])
    E2 = np.array(equal_pair['V2'])
    return E1, E2


####### Data Generation

## Simulate sequence
DATA = McSimulate(p_ij_true, N) 

## Compute Transitions counts
n =TransitionCounts(DATA,M)

## convert Transitions counts as matrix
n_ij_matrix = np.reshape(n,(M,M))

## compute MLE p_ij_hat_matrix
p_ij_hat_matrix = mle(n_ij_matrix)
p_ij_hat = p_ij_hat_matrix.flatten()


## denominator for penality term which is computed by MLE (p_ij_hat)
w_deno = prob_diff(p_ij_hat) 

# Objective function+
def obj_f(x):
    w_num  = prob_diff(x) 
    pelt = penalty_function(w_num,w_deno)
    return -np.sum(n * np.log(x)) + LAMBDA*np.sum(pelt)
    

# Equality constraints (Brute force constraints)
def eval_g_eq(x):
    E1, E2 = equal_pij_hat(w_deno)
    constraints = [
        x[0] + x[1] + x[2] - 1,
        x[3] + x[4] + x[5] - 1,
        x[6] + x[7] + x[8] - 1
    ]
    if E1.size > 0 and E2.size > 0:
        constraints += list(x[E1] - x[E2])
    return constraints

res = []

# Optimization results
res = minimize(obj_f, p_ij_hat, ## Initial value set to the MLE results for further optimize
    constraints={'type': 'eq', 'fun': eval_g_eq}, 
    bounds=[(0, 1)] * 9, 
    method='SLSQP', 
    options={'maxiter': 160000}
    )


## Export solutions
res.x = np.round(res.x,3)
p_tilde = np.reshape(res.x, (M,M))
p_tilde
p_tilde.sum(axis=1) ## Ensure sum of each row is 1
p_ij_true










