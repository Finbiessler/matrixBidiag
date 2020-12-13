import householderBidiag as hb
import projUtilities as util
import numpy as np

if __name__ == '__main__':

    # Init matrices and needed variables
    C = util.initRandomMatrix(5, 4)
    C_test = np.copy(C)

    print("-------------------------------------------- Householder-Bidiagonalization --------------------------------------------")
    # Print initial matrix
    util.printInputMatrix(C)

    # Calculate householder bidiagonalization of initial matirx A
    P, C, W = hb.houseBidiag(C)

    # Display the calculation results
    util.printCalcResults(P, C, W, C_test)