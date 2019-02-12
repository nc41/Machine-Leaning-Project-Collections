#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:11:32 2018

@author: chennai-fan
"""


from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
from sklearn import model_selection, metrics, cross_validation
import numpy as np
import plot_utils, utils
from sklearn.preprocessing import PolynomialFeatures


print 'Reading data ...'
bdata = load_boston()
df = pd.DataFrame(data = bdata.data, columns = bdata.feature_names)
X_Or = bdata.data
y = bdata.target
reg = 1.2

print '-------------split data into train and test sets(8:2)----------------'

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_Or, y, test_size =  0.2, random_state=45)
X_training, X_val, y_training, y_val = model_selection.train_test_split(X_train, y_train, test_size =  0.2, random_state=45)
poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X_training)
X_poly,mu,sigma = utils.feature_normalize(X_poly)
XX_poly = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T

X_poly_val = poly.fit_transform(X_val)

X_poly_test = poly.fit_transform(X_test)

X_poly_test = (X_poly_test - mu) / sigma
X_poly_val = (X_poly_val - mu) / sigma

XX_poly_test = np.vstack([np.ones((X_poly_test.shape[0],)),X_poly_test.T]).T
XX_poly_val = np.vstack([np.ones((X_poly_val.shape[0],)),X_poly_val.T]).T

print '---------------trainging--------------'

reg_vec, error_train, error_val = utils.validation_curve(XX_poly,y_training,XX_poly_val,y_val)
print 'Lamda:', reg_vec
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.show()


# calculate r2 scores and MSE from test set
def evaluate_model(theta,X_test,y_test):
    y_pred = np.dot(theta, X_test.T)
    mse = np.mean((y_pred - y_test) **2)
    print "Residual mean squared error: ", mse
    r_squared = metrics.r2_score(y_test,y_pred)
    print "Variance explained by model: ", r_squared
    return mse, r_squared


# testing quality of a linear model for predicting MEDV from LSTAT using k-fold crossvalidation

print '------------- Estimating/Evaluating model by crossvalidation -------------'
k = 5
kfolds = model_selection.KFold(n_splits=k)

mse,r_squared = np.zeros((k,)), np.zeros((k,))
i = 0
for (train,test) in kfolds.split(XX_poly):
    X_trainc, X_testc, y_trainc, y_testc = XX_poly[train, : ], XX_poly[test,: ], y_training[train], y_training[test]
    lr = RegularizedLinearReg_SquaredLoss()
    J_history = lr.train(X_trainc,y_trainc,reg = reg,num_iters=10000)
    mse[i],r_squared[i] = evaluate_model(J_history,X_testc,y_testc)
    i = i + 1

print k, " fold cross_validation MSE = ", np.mean(mse)
print k, " fold cross_validation r_squared = ", np.mean(r_squared)


