# -*- encoding: utf-8 -*-
# Written by Zibo

import pandas as pd 
import numpy as np

def pla(X, y):
    w = X[0]
    iteration = 0
    while True:
        iteration += 1
        false_data = 0

        for i in range(X.shape[0]):
            if np.sign(y[i]) != np.sign(np.dot(w, X[i])):
                false_data += 1
                w += y[i] * X[i]
        print 'iter %d (%d / %d)' % (iteration, false_data, X.shape[0])
        if not false_data:
            break
    return w

if __name__ == '__main__':
    df = pd.read_csv('bezdekIris.data', sep=',', header=None)
    df = df.iloc[0:100, 0:4] # 取前100个数据和前4列属性
    df['x0'] = 1.0 # 增加一列 x_0 = 0
    X = df.values 
    y = np.array([1.0] * 50 + [-1.0] * 50) # 前50个数据为一类，后50个数据为另一类
    w = pla(X, y)
    print w