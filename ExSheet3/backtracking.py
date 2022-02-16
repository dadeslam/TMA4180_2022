import numpy as np

def chooseStep(x_k, p_k, f, grad_f, rho=0.5, alpha_bar = 1, c_1 = 1/4):

    #Input:
    #x_k : current position
    #p_k : current direction
    #f : function to minimize
    #grad_f : its gradient

    #Output:
    #alpha : chosen step-size
    
    alpha = alpha_bar
    while f(x_k + alpha * p_k) > f(x_k) + c_1 * alpha * np.dot(grad_f(x_k),p_k):
        alpha *= rho
    return alpha

