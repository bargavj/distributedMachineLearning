import pickle
import numpy as np
from model import model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

####################
# chunk_size = 300 (adult), 10,000 (kdd)
# train_size = 30,000 (adult), 1,000,000 (kdd)
# test_size = ~15,000 (adult), ~3,000,000 (kdd)
####################

epsilon = 0.1
chunk_size = 300
SCALE = 1000000000

####################

def testModel(X, y, beta, intercept):
    #ypred = 2 * (np.dot(X, beta) + intercept > 0) - 1
    #return metrics.accuracy_score(y, ypred)
    return metrics.mean_squared_error(np.clip(y, -1, 1), np.clip(np.dot(X, beta) + intercept, -1, 1))

####################

def crossValidate(X, y, modelName, m):
    for i in [-10., -7., -4., -3.5, -3., -2.5, -2., -1.5]:
        kf = KFold(n_splits=5)
        acc = []
        param = 10 ** i
        for train_index, test_index in kf.split(X):
            xtrain, ytrain = X[train_index], y[train_index]
            xtest, ytest = X[test_index], y[test_index]
            n, d = xtrain.shape[0], xtrain.shape[1]
            for runs in range(10):
                beta = np.zeros(d)
                intercept = 0
                if modelName == "ours":
                    for j in range(m):
                        a, b = model.non_private(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], param)
                        beta += a
                        intercept += b
                    beta /= m
                    intercept /= m
                    beta += np.random.laplace(0, 2. / (m * chunk_size * param * epsilon), d)
                elif modelName == "baseline1":
                    for j in range(m):
                        a, b = model.baseline1(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], param, epsilon)
                        beta += a
                        intercept += b
                    beta /= m
                    intercept /= m
                elif modelName == "baseline2":
                    for j in range(m):
                        beta += model.baseline2(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], param, epsilon)
                    beta /= m
                elif modelName == "baseline3":
                    for j in range(m):
                        a, b = model.non_private(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], param)
                        beta += a
                        intercept += b
                    beta /= m
                    intercept /= m
                    beta += np.random.laplace(0, 2. / (chunk_size * param * epsilon), d)
                elif modelName == "non_private":
                    for j in range(m):
                        a, b = model.non_private(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], param)
                        beta += a
                        intercept += b
                    beta /= m
                    intercept /= m
                acc.append(testModel(xtest, ytest, beta, intercept))
        print(np.mean(acc))

####################

def trainAggregateModel(xtrain, ytrain, xtest, ytest, modelName, m):
    for epsilon in [0.5]:#0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        acc = []
        fp1 = open('Inputs/beta1.txt', 'a')
        fp2 = open('Inputs/beta2.txt', 'a')
        fp3 = open('Inputs/random_vals.txt', 'a')
        fp4 = open('Inputs/intercepts.txt', 'a')
        n, d = xtrain.shape[0], xtrain.shape[1]
        for runs in range(1):
            beta = np.zeros(d)
            intercept = 0
            if modelName == "ours":
                for j in range(m):
                    a, b = model.non_private(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], 10 ** -2.5)
                    beta1 = np.array([int(SCALE*val) for val in a])
                    beta2 = np.random.randint(-SCALE, SCALE, d)
                    beta1 ^= beta2
                    fp1.write(str(beta1))
                    fp2.write(str(beta2))
                    fp3.write(str(np.random.randint(-SCALE, SCALE, d)))
                    beta += a
                    intercept += b
                beta /= m
                intercept /= m
                print(beta)
                fp4.write(str(intercept))
                beta += np.random.laplace(0, 2. / (m * chunk_size * 10 ** -2.5 * epsilon), d)
            elif modelName == "baseline1":
                for j in range(m):
                    a, b = model.baseline1(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], 10 ** -1.5, epsilon)
                    beta += a
                    intercept += b
                beta /= m
                intercept /= m
            elif modelName == "baseline2":
                for j in range(m):
                    beta += model.baseline2(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], 10 ** -1.5, epsilon)
                beta /= m
            elif modelName == "baseline3":
                for j in range(m):
                    a, b = model.non_private(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], 10 ** -1.5)
                    beta += a
                    intercept += b
                beta /= m
                intercept /= m
                beta += np.random.laplace(0, 2. / (chunk_size * 10 ** -1.5 * epsilon), d)
            elif modelName == "non_private":
                for j in range(m):
                    a, b = model.non_private(xtrain[j * chunk_size : (j + 1) * chunk_size], ytrain[j * chunk_size : (j + 1) * chunk_size], 10 ** -1.5)
                    beta += a
                    intercept += b
                beta /= m
                intercept /= m
            acc.append(testModel(xtest, ytest, beta, intercept))
        fp1.close()
        fp2.close()
        fp3.close()
        fp4.close()
        print(np.mean(acc))

####################

#X, y = pickle.load(open('adult_data.p', 'rb'))
X, y = pickle.load(open('kddcup98_data_70k.p', 'rb'))
#X, y = pickle.load(open('kddcup99_data_70k.p', 'rb'))
#xtest, ytest = pickle.load(open('kddcup99_data_train.p', 'rb'))

X, y = shuffle(X, y, random_state = 0)
print(X.shape, y.shape)

#crossValidate(X[:50000], y[:50000], 'baseline3', 80)

#trainAggregateModel(X[:50000], y[:50000], X[50000:], y[50000:], 'ours', 100)
#trainAggregateModel(X, y, xtest, ytest, 'ours', 100)


fp1 = open('Output/beta_avg.txt', 'r')
fp2 = open('Inputs/intercepts.txt', 'r')

for line in fp1:
    beta = [float(val) for val in line.split(' ')[:-1]]
for line in fp2:
    intercept = float(line.split()[1][:-1])
fp1.close()
fp2.close()

print(beta, len(beta), intercept)
print(testModel(X[50000:], y[50000:], beta, intercept))

'''
for param in [10 ** i for i in [-10., -7., -4., -3.5, -3., -2.5, -2., -1.5]]:
    print(param)
    #clf = linear_model.LogisticRegression(C = 1. / param, penalty = 'l2', solver = 'lbfgs')
    clf = svm.LinearSVC(C = 1. / param, penalty = 'l2')
    #clf = linear_model.SGDClassifier(loss = 'modified_huber', penalty = 'l2', alpha = param)

    scores = cross_val_score(clf, X, y, cv = 10)
    print(np.mean(scores))

clf.fit(XTrain, yTrain)

ypred = clf.predict(XTest) 
print(metrics.accuracy_score(yTest, ypred))
print(metrics.classification_report(yTest, ypred))'''
