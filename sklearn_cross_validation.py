'''
@File    :   sklearn_cross_validation.py
@Time    :   2020/05/12 19:16:59
@Author  :   Pan 
@Version :   1.0
@Contact :   pwd064@mail.ustc.edu.cn
@License :   (C)Copyright 2020-2025, USTC
@Desc    :   None
'''

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import SCORERS
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

Iris   = datasets.load_iris()        
X      = Iris.data
Y      = Iris.target
# X      = preprocessing.maxabs_scale(X) 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)     
model  = KNeighborsClassifier(n_neighbors=4)               
# model.fit(X_train,Y_train)                 
# scores = model.score(X_test,Y_test)      
# score2 = cross_val_score(model,X,Y,cv=5) 
# print(scores,score2)
train_sizes, train_scores, test_scores = learning_curve(model,X,Y,cv=10,train_sizes=[60,80,100,120])
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean  = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,'o-',label='train')
plt.plot(train_sizes,test_scores_mean,'o-',label='test')
plt.legend()
plt.xlabel('samples')
plt.ylabel('train/cv accuracy')
plt.show()
