import time
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from utility import secure_aggregate_laplace, secure_aggregate_gaussian
import matplotlib.pyplot as plt

####################
# chunk_size = 300 (adult), 10,000 (kdd)
# train_size = 30,000 (adult), 1,000,000 (kdd)
# test_size = ~15,000 (adult), ~3,000,000 (kdd)
####################

epsilon    = 0.5
eta        = 1.
lambda2    = 0.001
delta      = 0.001
M          = 100
chunk      = 500
T          = 1500
SCALE      = 1000000000


#### gradients ####

def log_reg_grad(v, X, y, lambda2):
    n, d = X.shape[0], X.shape[1]        
    y = np.matrix(y).reshape((n, 1))
    X = np.array(np.multiply(X, y))
    l = np.exp(np.dot(X, v))
    return -np.dot(X.T, 1. / (1 + l)) / n + lambda2 * v
            
def lin_reg_grad(v, X, y, lambda2):
    n, d = X.shape[0], X.shape[1]        
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    return (np.dot(XtX, v) - Xty) / n + lambda2 * v

def gradient(v, X, y, lambda2):
    return log_reg_grad(v, X, y, lambda2)

#### loss values ####

def log_reg_loss(v, X, y, lambda2):
    n, d = X.shape[0], X.shape[1]        
    y = np.matrix(y).reshape((n, 1))
    X = np.array(np.multiply(X, y))
    return np.sum(np.log(1 + np.exp(np.dot(X, v)))) / n + lambda2 * (np.linalg.norm(v) ** 2)
            
def lin_reg_loss(v, X, y, lambda2):
    n, d = X.shape[0], X.shape[1]
    return np.sum((np.dot(X, v) - y)**2) / n + lambda2 * (np.linalg.norm(v) ** 2)

def optimality_gap(beta, X, y, lambda2, T, m, epsilon):
    return np.abs(log_reg_loss(beta, X, y, lambda2) - log_reg_loss(beta_ref, X, y, lambda2))

#### models ####

def centralized_non_private(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    beta = np.zeros(d)
    
    for t in range(T):
        beta -= eta * gradient(beta, X, y, lambda2)
    return beta


def distributed_non_private(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    local_betas = np.zeros((m, d))
    acc, loss = [], []
    
    for t in range(T):
        for j in range(m):
            local_betas[j] -= eta * gradient(local_betas[j], X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2)
        beta = np.sum(local_betas, axis=0) / m
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta


#### Chaudhuri objective perturbation ####
def local_objective_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    local_betas = np.zeros((m, d))
    acc, loss = [], []
    
    epsilon2 = epsilon - 2 * np.log(1. / (4 * chunk * lambda2))
    if epsilon2 > 0:
        Delta = 0.
    else:
        Delta = 1. / (4 * chunk * (np.exp(epsilon / 4.) - 1)) - lambda2
        epsilon2 = epsilon / 2.
            
    b = np.random.laplace(0, 2. / epsilon2, d)
   
    for t in range(T):
        for j in range(m):
            local_betas[j] -= eta * ( gradient(local_betas[j], X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2) + Delta * local_betas[j] + b / chunk )
        beta = np.sum(local_betas, axis=0) / m
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta
    

#### Chaudhuri output perturbation ####
def local_output_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    local_betas = np.zeros((m, d))
    acc, loss = [], []
    
    for t in range(T):
        for j in range(m):
            local_betas[j] -= eta * gradient(local_betas[j], X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2)
        beta = np.sum(local_betas, axis=0) / m + np.random.laplace(0, 2. / (chunk * lambda2 * epsilon), d) / np.sqrt(m)
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta


#### Pathak output perturbation ####
def distributed_output_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    local_betas = np.zeros((m, d))
    acc, loss = [], []
    
    for t in range(T):
        for j in range(m):
            local_betas[j] -= eta * gradient(local_betas[j], X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2)
        beta = np.sum(local_betas, axis=0) / m + np.random.laplace(0, 2. / (chunk * lambda2 * epsilon), d)
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta
    

def centralized_gradient_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    beta = np.zeros(d)
    acc, loss = [], []
    
    for t in range(T):
        beta -= eta * ( gradient(beta, X, y, lambda2) + np.random.normal(0, np.sqrt(2. * T) / (n * ( np.sqrt(np.log(1. / delta) + epsilon) - np.sqrt(np.log(1. / delta)) ) ), d) )
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta
    

#### Shokri gradient perturbation ####
def local_gradient_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    beta = np.zeros(d)
    acc, loss = [], []
    
    for t in range(T):
        grad = np.zeros(d)
        for j in range(m):
            grad += gradient(beta, X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2)
        beta -= eta * ( grad / m + np.random.normal(0, np.sqrt(2. * T) / (chunk * ( np.sqrt(np.log(1. / delta) + epsilon) - np.sqrt(np.log(1. / delta)) ) ), d) / np.sqrt(m) )
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta


#### Rajkumar and Arun objective perturbation ####
def centralized_objective_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    local_betas = np.zeros((m, d))
    beta = np.zeros(d)
    acc, loss = [], []
    
    epsilon2 = epsilon - 2 * np.log(1. / (4 * chunk * lambda2))
    if epsilon2 > 0:
        Delta = 0.
    else:
        Delta = 1. / (4 * chunk * (np.exp(epsilon / 4.) - 1)) - lambda2
        epsilon2 = epsilon / 2.
            
    b = np.random.normal(0, 2. * np.sqrt(2 * np.log(1.25 / delta)) / epsilon2, d)
   
    for t in range(T):
        grad = np.zeros(d)
        for j in range(m):
            grad += gradient(beta, X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2)
        beta -= eta * ( grad / m + Delta * beta / (m * chunk) + b / (m * chunk) + np.random.laplace(0, 2. / (m * chunk * epsilon), d) )
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta


#### Proposed Method 1: Output Perturbation ####
def proposed_output_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    local_betas = np.zeros((m, d))
    acc, loss = [], []
    
    for t in range(T):
        for j in range(m):
            local_betas[j] -= eta * gradient(local_betas[j], X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2)
        # Note: set useMPC=True to run the secure MPC code
        beta = secure_aggregate_laplace(local_betas, 2. / (m * chunk * lambda2 * epsilon), useMPC=False)
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta
        

#### Proposed Method 2: Gradient Perturbation ####
def proposed_gradient_pert(X, y, lambda2, T, m, epsilon):
    n, d = X.shape[0], X.shape[1]
    beta = np.zeros(d)
    acc, loss = [], []
    
    for t in range(T):
        grads = [gradient(beta, X[j * chunk : (j + 1) * chunk], y[j * chunk : (j + 1) * chunk], lambda2) for j in range(m)]
        # Note: set useMPC=True to run the secure MPC code
        grad = secure_aggregate_gaussian(np.array(grads), np.sqrt(2. * T) / (m * chunk * (np.sqrt(np.log(1. / delta) + epsilon) - np.sqrt(np.log(1. / delta)))), useMPC=False)
        beta -= eta * grad
        loss.append(optimality_gap(beta, X, y, lambda2, T, m, epsilon))
        #acc.append(testModel(xtest, ytest, beta))
    plt.plot(loss)
    plt.show()
    return beta


####################

def testModel(X, y, beta):
    ypred = 2 * (np.dot(X, beta) > 0) - 1
    ypred_ref = 2 * (np.dot(X, beta_ref) > 0) - 1
    return np.abs(metrics.accuracy_score(y, ypred) - metrics.accuracy_score(y, ypred_ref))
    #return np.abs(metrics.mean_squared_error(y, np.dot(X, beta)) - metrics.mean_squared_error(y, np.dot(X, beta_ref)))


####################


def crossValidate(X, y, modelName, m):
    switch = {
        "centralized_non_private"    : centralized_non_private,
        "distributed_non_private"    : distributed_non_private,
        "local_output_pert"          : local_output_pert,
        "distributed_output_pert"    : distributed_output_pert,
        "proposed_output_pert"       : proposed_output_pert,
        "local_objective_pert"       : local_objective_pert,
        "centralized_gradient_pert"  : centralized_gradient_pert,
        "local_gradient_pert"        : local_gradient_pert,
        "proposed_gradient_pert"     : proposed_gradient_pert,
        "centralized_objective_pert" : centralized_objective_pert
    }
    fun = switch.get(modelName)
          
    for i in [-10., -7., -3.5, -2.5, -2., -1.5]:
        kf = KFold(n_splits=5)
        acc = []
        param = 10 ** i
        for train_index, test_index in kf.split(X):
            xtrain, ytrain = X[train_index], y[train_index]
            xtest, ytest = X[test_index], y[test_index]
            n, d = xtrain.shape[0], xtrain.shape[1]
            for runs in range(5):
                beta = fun(xtrain, ytrain, param, T, m, epsilon)
                acc.append(testModel(xtest, ytest, beta))
        print(np.mean(acc))


def trainAggregateModel(xtrain, ytrain, xtest, ytest, modelName, m):
    switch = {
        "centralized_non_private"    : centralized_non_private,
        "distributed_non_private"    : distributed_non_private,
        "local_output_pert"          : local_output_pert,
        "distributed_output_pert"    : distributed_output_pert,
        "proposed_output_pert"       : proposed_output_pert,
        "local_objective_pert"       : local_objective_pert,
        "centralized_gradient_pert"  : centralized_gradient_pert,
        "local_gradient_pert"        : local_gradient_pert,
        "proposed_gradient_pert"     : proposed_gradient_pert,
        "centralized_objective_pert" : centralized_objective_pert
    }
    fun = switch.get(modelName)
    for epsilon in [0.5]:#0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        acc, gap = [], []
        n, d = xtrain.shape[0], xtrain.shape[1]
        for runs in range(1):           
            beta = fun(xtrain, ytrain, lambda2, T, m, epsilon)
            acc.append(testModel(xtest, ytest, beta))
            gap.append(optimality_gap(beta, xtrain, ytrain, lambda2, T, m, epsilon))
        print("Relative Error is : " + str(np.mean(acc)))
        print("Optimality Gap is : " + str(np.mean(gap)))

            
####################

#X, y = pickle.load(open('Dataset/adult_data.p', 'rb'))
#X, y = pickle.load(open('Dataset/kddcup98_data_70k.p', 'rb'))
X, y = pickle.load(open('Dataset/kddcup99_data_70k.p', 'rb'))

X, y = shuffle(X, y, random_state = 0)
print(X.shape, y.shape)

modelName = 'centralized_objective_pert'
print(modelName)
print('##############')
beta_ref = centralized_non_private(X[:50000], y[:50000], lambda2, 1500, M, epsilon)

xtest, ytest = X[50000:], y[50000:]

#crossValidate(X[:50000], y[:50000], modelName, 80)

t0 = time.time()
trainAggregateModel(X[:50000], y[:50000], X[50000:], y[50000:], modelName, M)
t1 = time.time()
print('Runtime : ' + str(t1-t0))
