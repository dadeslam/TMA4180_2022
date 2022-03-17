import numpy as np

def CG(A, b, x_0, tol = 1e-15, max_it = 100):
    #The algorithm solves the linear system Ax=b with CG method
    #max_it is the maximum number of iterations allowed
    #x_0 is the initial guess

    x = x_0 #initial guess
    r = A@x - b #initial residual
    p = -r #initial conjugate direction

    it = 0

    while np.linalg.norm(p)>tol and it<max_it:
        alpha = np.dot(r,r)/np.dot(p,A@p)
        x = x + alpha * p
        rnew = r + alpha * A@p
        beta = np.dot(rnew,rnew)/np.dot(r,r)
        r = rnew
        p = -r + beta*p
        it += 1

    return x , it

