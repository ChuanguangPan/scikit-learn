'''
@File    :   sklearn_preprocessing.py
@Time    :   2020/05/10 09:26:03
@Author  :   Pan 
@Version :   1.0
@Contact :   pwd064@mail.ustc.edu.cn
@License :   (C)Copyright 2020-2025, USTC
@Desc    :   None
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import preprocessing
import numpy as np 

n  = 2           # 训练样本数
m  = 5            # 样本维度
X  = np.random.randn(n,m)*10                   # 生成随机自变量
# w  = np.array([1,2,3,4,5,0.01,0.1,0.05,0.0001,6])
# # w = np.arange(1,11,1).T                     # 设置线性系数
# Y  = np.dot(X,w) + np.random.randn(n)/10       # 经过线性系数加权并加噪
# Y[Y>0] = 1
# Y[Y<=0] = 0
# X1 = preprocessing.normalize(X,norm='l2')
X2 = preprocessing.maxabs_scale(X,axis=1)
# print(X,'\n',X2)
X3 = preprocessing.normalize(X,axis=0)
print(X,'\n',X3)

# n = 200           # 训练样本数
# m = 10            # 样本维度
# X = np.random.randn(n,m)*10                   # 生成随机自变量
# w = np.array([1,2,3,4,5,0.01,0.1,0.05,0.0001,6])
# # w = np.arange(1,11,1).T                     # 设置线性系数
# Y = np.dot(X,w) + np.random.randn(n)/10       # 经过线性系数加权并加噪
# Y[Y>0] = 1
# Y[Y<=0] = 0
# Y1 = np.dot(X,w) + np.random.randn(n)/10

# model = linear_model.LinearRegression()
# model.fit(X.T,Y1)
# print(model.coef_)
