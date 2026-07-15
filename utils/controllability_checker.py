import numpy as np

def controllabilty(A:np.ndarray, B:np.ndarray):
    #verifying the inputs A and B
    A=np.asarray(A) # state matrix, dim: nxn
    B=np.asarray(B) # input matrix, dim: nxm

    n=A.shape[0]

    #constructing the controllability matrix

    cols=[B]
    for i in range(1,n):
        cols.append(A@cols[-1])

    cols=np.hstack(cols)

    # check the rank of the matrix
    rank=np.linalg.matrix_rank(cols)
    
    
    