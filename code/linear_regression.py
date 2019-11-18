import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import os

def linear_regression(X,Y, intercept=1):
    '''
    linear_regression(X,Y, intercept)
    the function finds the coefficients of the linear regression: Y = beta1*X + beta2
    if intercept = 1 it finds the coefficient beta2
    if intercept = 0, beta is fixed to 0
    '''

    if intercept == 1:
        X = sm.add_constant(X)
        linear_model = sm.OLS(Y,X).fit()
        predictions = linear_model.predict(X) # make the predictions by the model
        X = X[:,1]
    elif intercept == 0:
        linear_model = sm.OLS(Y,X).fit()
        predictions = linear_model.predict(X) # make the predictions by the model
        linear_model.params = np.insert(linear_model.params,0,0)
    else:
        raise ValueError('Intercept can be equal to 0 or 1')
    return linear_model

