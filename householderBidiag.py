import numpy as np
import numpy.linalg as nla

np.set_printoptions(suppress=True)

def computeHouseVec(x):
    dot_1on = x[1:].dot(x[1:])

    # v is our return vector; we hack on v[0]
    v = np.copy(x)
    v[0] = 1.0

    if dot_1on < np.finfo(float).eps:
        beta = 0.0
    else:
        # apply Parlett's formula (G&vL page 210) for safe v_0 = x_0 - norm(X)
        norm_x = np.sqrt(x[0] ** 2 + dot_1on)
        if x[0] <= 0:
            v[0] = x[0] - norm_x
        else:
            v[0] = -dot_1on / (x[0] + norm_x)
        beta = 2 * v[0] ** 2 / (dot_1on + v[0] ** 2)
        v = v / v[0]
    return v, beta


def blockMatrixEmbedHouseVec(n, col, v, beta):
    ''' for size n, embed a Householder vector v in the lower right block of
        a identity matrix to get a full-sized matrix with a smaller Householder matrix block'''
    full = np.eye(n)
    full[col:, col:] -= beta * np.outer(v, v)
    return full


# G&VL Algo. 5.4.2 with explicit reflections
def houseBidiag(A):
    m, n = A.shape
    assert m >= n
    P, Wt = np.eye(m), np.eye(n)

    for col in range(n):
        v, beta = computeHouseVec(A[col:, col])
        A[col:, col:] = (np.eye(m - col) - beta * np.outer(v, v)).dot(A[col:, col:])
        Q = blockMatrixEmbedHouseVec(m, col, v, beta)
        P = P.dot(Q)

        if col <= n - 2:
            # transpose here, reflection for zeros above diagonal in A
            # col+1 keeps us off the super diagonal
            v, beta = computeHouseVec(A[col, col + 1:].T)
            A[col:, col + 1:] = A[col:, col + 1:].dot(np.eye(n - (col + 1)) - beta * np.outer(v, v))
            Pt = blockMatrixEmbedHouseVec(n, col + 1, v, beta)
            Wt = Pt.dot(Wt)

        step = col+1
        #print(f"-------------------------------------------- Step No.{step} --------------------------------------------")
        #print("P:\n", P)
        #print("C:\n", A)
        #print("Wt:\n", Wt)
        #print("\n")
    return P, A, Wt

def getBidiagElems(C):
    m, n = C.shape
    betas, alphas = [], []
    for i in range(n):
        betas.append(C[i,i])
        if i == n-1:
            return np.array(betas), np.array(alphas)
        else:
            alphas.append(C[i, i+1])