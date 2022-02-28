import numpy
from backtracking import *

def updateH(H,rho_k,s_k,y_k):
    I = np.eye(len(H))
    mat1 = I-rho_k*np.outer(s_k,y_k)
    mat2 = rho_k * np.outer(s_k,s_k)
    return mat1@H@mat1 + mat2

def exactLS(x,p,Q,grad_f):
    num = -np.dot(p,grad_f(x))
    den = np.dot(p,Q@p)
    return num/den

def BFGS(f, grad_f, Q, H_0, x_0, exact=False, tol = 1e-6, max_it = 10000):
    #exact : is a boolean value that says if we are using exact or inexact line search
    sol = x_0
    it = 0
    res = tol + 1
    H = H_0
    while it<max_it and res>tol:
        p = -H@grad_f(sol)
        if exact:
            alpha = exactLS(sol,p,Q,grad_f)
        else:
            alpha = chooseStep(sol, p, f, grad_f)
        supp = sol + alpha * p
        s = supp - sol  
        y = grad_f(supp)-grad_f(sol)
        sol = supp
        rho = 1/np.dot(s,y)
        H=updateH(H, rho, s, y)
        it+=1
        res = np.linalg.norm(grad_f(sol),2)
    return sol, it