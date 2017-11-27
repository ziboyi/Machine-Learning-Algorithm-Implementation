# -*- encoding: utf-8 -*-
# Written by Zibo

import pandas as pd 
import numpy as np

def error_classifications(X, y, weight):
    n_error = 0
    for i in range(X.shape[0]):
        if np.sign(y[i]) != np.sign(np.dot(weight, X[i])):
            n_error += 1
    return n_error

def pla_pocket(X, y, limit):
    w = np.random.rand(5)
    min_error = error_classifications(X, y, w)
    final_w = w

    for _ in xrange(limit):
        i = np.random.choice(range(X.shape[0]))
        # 若随机选取的样本能正确分类，则再选一个
        while np.sign(y[i]) == np.sign(np.dot(w, X[i])): 
            i = np.random.choice(range(X.shape[0]))
        temp_w = w + y[i] * X[i]
        n_error = error_classifications(X, y, temp_w)

        w = temp_w

        if n_error <= min_error:
            min_error = n_error
            final_w = temp_w
    return final_w, min_error

if __name__ == '__main__':
    df = pd.read_csv('bezdekIris.data', sep=',', header=None)
    df = df.iloc[50:, 0:4] # 取后100个数据和前4列属性，它们对应的两个类别线性不可分
    df['x0'] = 1.0 # 增加一列 x_0 = 1
    X = df.values 
    y = np.array([1.0] * 50 + [-1.0] * 50) # 前50个数据为一类，后50个数据为另一类
    weight, min_error = pla_pocket(X, y, 100)
    print 'Weight is:', weight
    print 'The number of error classification is: ', min_error