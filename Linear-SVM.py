# -*- encoding: utf-8 -*-
# Written by Zibo

import numpy as np
import pandas as pd
import cvxopt
import cvxopt.solvers

def linear_svm(X, y):
    n_samples, n_features = X.shape

    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = np.dot(X[i], X[j])

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1,n_samples))
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    h = cvxopt.matrix(np.zeros(n_samples))
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    alpha = np.ravel(solution['x'])
    sv = alpha > 1e-5     # return a list with bool values
    index = np.arange(len(alpha))[sv]  # sv's index
    alpha = alpha[sv]
    sv_X = X[sv]  # sv's data
    sv_y = y[sv]  # sv's labels
    print("%d support vectors out of %d points" % (len(alpha), n_samples))

    w = np.zeros(n_features)
    for n in range(len(alpha)):
        w += alpha[n] * sv_y[n] * sv_X[n]

    b = sv_y[0] - np.dot(w, sv_X[0])

    return w, b


if __name__ == '__main__':
    df = pd.read_csv('bezdekIris.data', sep=',', header=None)
    df = df.iloc[0:100, 0:4] # 取前100个数据和前4列属性
    X = df.values 
    y = np.array([1.0] * 50 + [-1.0] * 50) # 前50个数据为一类，后50个数据为另一类
    weight, bias = linear_svm(X, y)
    print 'Weight is: ', weight
    print 'Bias is: ', bias
