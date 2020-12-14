import numpy as np
np.set_printoptions(linewidth=300)
def printInputMatrix(C):
    print("-------------------------------------------- Input Matrix --------------------------------------------")
    print("C:\n", C)
    print("\n")

def printCalcResults(P, C, Wt, origMatirx):
    #print("-------------------------------------------- Final Result --------------------------------------------")
    print("P:\n", P)
    print("B:\n", C)
    print("Wt:\n", Wt)
    print("\n")
    print("C == P*B*Wt:", np.allclose(P@C@Wt, origMatirx))
    print("\n")

def initRandomMatrix(rows , cols):
    A = np.random.randn(rows, cols)
    for row in range(rows):
        for elem in range(cols):
            if A[row, elem] < 0:
                A[row, elem] = abs(A[row, elem])
    return A

def getBlockFromMatrix(row, col, M):
    return M[row:, col:]

def removeZeroBottomFromMatrix(numOfColsNotFullZero, M):
    return M[0:numOfColsNotFullZero, 0:]

def removeFirstColFromMatrix(M):
    return M[0:, 1:]