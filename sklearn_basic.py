'''
@File    :   general_linear_regression.py
@Time    :   2020/04/29 20:52:40
@Author  :   Pan 
@Version :   1.0
@Contact :   pwd064@mail.ustc.edu.cn
@License :   (C)Copyright 2020-2025, USTC
@Desc    :   None
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np 

## -------------------------------------------------------------------
##              自己创建数据
## -------------------------------------------------------------------
n = 100           # 训练样本数
m = 10            # 样本维度
X = np.random.randn(n,m)*10                   # 生成随机自变量
w = np.array([1,2,3,4,5,0.01,0.1,0.05,0.0001,6])
# w = np.arange(1,11,1).T                     # 设置线性系数
Y = np.dot(X,w) + np.random.randn(n)/10       # 经过线性系数加权并加噪
Y[Y>0] = 1
Y[Y<=0] = 0
Y1 = np.dot(X,w) + np.random.randn(n)/10

## -------------------------------------------------------------------
##               导入datasets中已有的数据集
## -------------------------------------------------------------------= 
# Iris = datasets.load_iris()
# X    = Iris.data
# Y    = Iris.target
# targetnames = list(Iris.target_names)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# model = linear_model.LogisticRegression()
# model.fit(X_train,Y_train)
# print(list(targetnames[i] for i in model.predict(X_test)))
# print(list(targetnames[j] for j in Y_test))
# print(model.coef_,model.intercept_)
# print(model.get_params())
# print(model.score(X_test,Y_test))
## -------------------------------------------------------------------
##              linear regression
## -------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
linear = linear_model.LogisticRegression()      # 调用线性模型
z = linear.fit(X_train,Y_train)
print(z.predict(X_test))
print(Y_test)
print(linear.score(X,Y))
print(linear.coef_,linear.intercept_)
print(linear.get_params())
## -------------------------------------------------------------------
##              ridge regression
## -------------------------------------------------------------------
# ridge = linear_model.Ridge(0.3)
# z = ridge.fit(X,Y)
# print(ridge.coef_)
# X1= np.ones((2,m))
# print(z.predict(X1))
# ridge_cv = linear_model.RidgeCV([0.01,0.5,0.1,0.5,1])
# z = ridge_cv.fit(X,Y)
# print(ridge_cv.coef_,'\n',ridge_cv.alpha_)
## -------------------------------------------------------------------
##              lasso regression
## -------------------------------------------------------------------
# lasso = linear_model.Lasso(alpha = 3)
# lasso = linear_model.LassoCV(eps = 0.0001,n_alphas = 1000,cv=5)
# z = lasso.fit(X,Y)
# print(z.coef_,'\n',lasso.alpha_)
# X1= np.ones((2,m))
# print(z.predict(X1))
## -------------------------------------------------------------------
##              ElasticNet regression
## -------------------------------------------------------------------
# Elastic = linear_model.ElasticNetCV(l1_ratio=0.8)
# z = Elastic.fit(X,Y)
# print(Elastic.coef_,'\n',z.alpha_)
# X1= np.ones((2,m))
# print(z.predict(X1))
## -------------------------------------------------------------------
##              Logistic classification
## -------------------------------------------------------------------
# Logistic = linear_model.LogisticRegression(C=0.5,penalty='l2')
# z = Logistic.fit(X,Y)
# print(Logistic.coef_,z.penalty)
# X1= np.random.randn(n,m)*10
# print(z.predict(X1))
# Logistic = linear_model.LogisticRegressionCV(Cs=1000,l1_ratios=0.1)
# z = Logistic.fit(X,Y)
# print(Logistic.coef_,z.C_)
## -------------------------------------------------------------------
##              SGD classification/regression
## -------------------------------------------------------------------
# sgd_classification = linear_model.SGDClassifier(loss='log')
# z  = sgd_classification.fit(X,Y)
# sgd_regression = linear_model.SGDRegressor()
# z1 = sgd_regression.fit(X,Y1)
# print(z.coef_)
# print(z1.coef_)
# X1= np.random.randn(10,m)
# print(z.predict(X1))
# print(z1.predict(X1))
## -------------------------------------------------------------------
##              Perceptron
## -------------------------------------------------------------------
# perceptron = linear_model.Perceptron(penalty='l2',alpha=0.001)
# f = perceptron.fit(X,Y)
# print(f.alpha,f.coef_)
# X1= np.random.randn(10,m)
# print(f.predict(X1))

