import numpy as np

def calcAFromBZP(B, Z, P):

    return B

def lgkBidiag(A, P, Z, b, cols):
    #Initialization step
    print(cols)
    alphas = np.zeros(cols-1)
    betas  = np.zeros(cols)
    betas[0], P[:,0] = factorCoefficientAndVector(b)

    #Computation of alphas and remaining betas
    for i in range(cols-1):
        if i == 0:
            alphas[i], Z[:, i] = factorCoefficientAndVector(np.matmul(A.T, P[:, i]))
        else:
            alphas[i], Z[:,i] = factorCoefficientAndVector(np.matmul(A.T, P[:,i]) - betas[i]*Z[:,i])

        betas[i+1], P[:, i+1] = factorCoefficientAndVector(np.matmul(A, Z[:,i]) - alphas[i]*P[:,i])

    return alphas, betas, P, Z

def extractbFromInputMatrix(M):
    return np.copy(M[:, 0])

def factorCoefficientAndVector(x):

    coef = 1/np.linalg.norm(x)
    vec  = (coef) * x

    return coef, vec
