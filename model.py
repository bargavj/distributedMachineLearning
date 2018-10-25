import numpy as np
import math
from sklearn import linear_model
from sklearn import svm
import apgpy as apg


class model:
    #### Baseline 1 ####
    def baseline1(X, y, lambda2, epsilon): # Chaudhuri et al. 2011 - Output Perturbation
        n, d = X.shape[0], X.shape[1]
        #clf = linear_model.Ridge(alpha = lambda2).fit(X, y)
        clf = linear_model.LogisticRegression(C = 1. / lambda2, penalty = 'l2', solver = 'lbfgs').fit(X, y)
        beta = clf.coef_.ravel()
        beta += np.random.laplace(0, 2. / (n * lambda2 * epsilon), d)
        return beta, clf.intercept_.ravel()

    def baseline11(X, y, lambda2, epsilon):
        n, d = X.shape[0], X.shape[1]        
        #y = np.matrix(y).reshape((n, 1))
        #X = np.array(np.multiply(X, y))
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
 
        def log_reg_grad(v):
            l = np.exp(np.dot(X, v))
            return -np.dot(X.T, 1. / (1 + l)) / n + lambda2 * v

        def lin_reg_grad(v):
            return (np.dot(XtX, v) - Xty) / n + lambda2 * v

        beta = apg.solve(lin_reg_grad, {}, np.zeros(d), eps=1e-9, quiet=True, step_size=1., fixed_step_size=True, max_iters=100, use_gra=True) 
        beta += np.random.laplace(0, 2. / (n * lambda2 * epsilon), d)
        return beta


    #### Baseline 2 ####
    def baseline2(X, y, lambda2, epsilon): # Chaudhuri et al. 2011 - Objective Perturbation
        n, d = X.shape[0], X.shape[1]
        epsilon2 = epsilon - 2 * math.log(1 / (4 * n * lambda2)) # was log(1 + 1/(4...))
        if epsilon2 > 0:
            Delta = 0.
        else:
            Delta = 1 / (4 * n * (math.exp(epsilon / 4.) - 1)) - lambda2
            epsilon2 = epsilon / 2.
        
        #y = np.matrix(y).reshape((n, 1))
        #X = np.array(np.multiply(X, y))
        b = np.random.laplace(0, 2. / epsilon2, d)
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
 
        def log_reg_grad(v):
            l = np.exp(np.dot(X, v))
            return -np.dot(X.T, 1. / (1 + l)) / n + (Delta + lambda2) * v + b / n # divided l(.) by n
        
        def lin_reg_grad(v):
            return (np.dot(XtX, v) - Xty) / n + (Delta + lambda2) * v + b / n

        beta = apg.solve(lin_reg_grad, {}, np.zeros(d), eps=1e-9, quiet=True, step_size=1., fixed_step_size=True, max_iters=100, use_gra=True)
        return beta


    #### non-private ####
    def non_private(X, y, lambda2):
        #clf = linear_model.Ridge(alpha = lambda2).fit(X, y)
        clf = linear_model.LogisticRegression(C = 1. / lambda2, penalty = 'l2', solver = 'lbfgs').fit(X, y)
        return clf.coef_.ravel(), clf.intercept_.ravel()

    def non_private1(X, y, lambda2):
        n, d = X.shape[0], X.shape[1]        
        #y = np.matrix(y).reshape((n, 1))
        #X = np.array(np.multiply(X, y))
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
 
        def log_reg_grad(v):
            l = np.exp(np.dot(X, v))
            return -np.dot(X.T, 1. / (1 + l)) / n + lambda2 * v
            
        def lin_reg_grad(v):
            return (np.dot(XtX, v) - Xty) / n + lambda2 * v

        beta = apg.solve(lin_reg_grad, {}, np.zeros(d), eps=1e-9, quiet=True, step_size=1., fixed_step_size=True, max_iters=100, use_gra=True)
        return beta
    ####################
    
    
    #### gradient_pert ####
    def gradient_pert(v, X, y, lambda2):
        n, d = X.shape[0], X.shape[1]        
        #y = np.matrix(y).reshape((n, 1))
        #X = np.array(np.multiply(X, y))
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)

        #l = np.exp(np.dot(X, v))
        #return -np.dot(X.T, 1. / (1 + l)) / n + lambda2 * v
        return (np.dot(XtX, v) - Xty) / n + lambda2 * v

