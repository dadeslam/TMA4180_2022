import numpy as np
from backtracking import *

def gradientDescent(f, grad_f, x_0, tol = 1e-3, max_it = 1000):

    #Input:
    #f : function to minimize
    #grad_f : its gradient
    #x_0 : initial guess
    #tol : stopping criterion is |xnew - x|<tol
    #max_it : maximum number of allowed iterations

    #Output:
    #x : approximation of the minimum
    #it : number of iterations done

    res = 1
    x = x_0
    p = -grad_f(x)
    alpha = chooseStep(x, p, f, grad_f, rho=0.5, alpha_bar = 1, c_1 = 1/4)
    it = 0
    while res>tol and it<max_it:
        x = x + alpha * p
        p = -grad_f(x)
        alpha = chooseStep(x, p, f, grad_f, rho=0.5, alpha_bar = 1, c_1 = 1/4)
        res = np.linalg.norm(grad_f(x))
        it+=1
    return x, it