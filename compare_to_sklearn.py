"""
Function to run the polished fast gradient logistic regression on the spam
dataset. The function prints to the screen the final beta values, the
predictions, the accuracy score and the confusion matrix.
"""

import copy
import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.linalg
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

filename = 'https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data'
spam = pd.read_csv(filename, delimiter = ' ', header = None)

features = spam.drop(57, axis=1)
response = spam.ix[:,57]

def split_scale(x, y, split=0.75):
    """
    Function to take an input and split into training and test sets and
    subsequently scale them.

    :param x: dataframe or matrix of features
        The features of the data.
    :param y: dataframe or array of response
        The response of the data to be split and scaled.
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split, stratify=y)

    #scale the data
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return(x_train, x_test, y_train, y_test)

##split and scale the data
x_train, x_test, y_train, y_test = split_scale(features, response)

def objective(beta, lamda, x=x_train, y=y_train):
    """
    Function to calculate the objective for l2-regularized logistic regression.

    :param beta: array
        An array of the beta values used to calculated the objective
    :param lambda: float
        The lambda, or regularization parameter, used to calculate the objective.
    :param x: dataframe or Matrix
        The features used to calculate the objective
    :param y: array
        The response from the data.
    """
    n = len(x)
    # the objective function
    obj = 1/n * np.sum(np.log(1+np.exp(-y*np.dot(x,beta)))) + lamda*np.linalg.norm(beta)**2
    return(obj)

def computegrad(beta, lamda, x=x_train, y=y_train):
    """
    Function to compute the gradient of the l2-regularized logistic regression
    objective function.

    :param beta: array
        An array of the beta values used to calculated the objective
    :param lambda: float
        The lambda, or regularization parameter, used to calculate the objective.
    :param x: dataframe or Matrix
        The features used to calculate the objective
    :param y: array
        The response from the data.
    """
    n = len(x)
    #the following tmp objects all calculate a piece of the overall gradient
    tmp1 = x.dot(beta)
    tmp2 = 1/(1+np.exp(y*tmp1))
    gradient =  -1/n*(x.T.dot(y*tmp2))
    #final gradient calculation
    gradient = gradient + lamda*2*beta
    return(gradient)

def backtracking(x, y, beta, lamda, eta, alpha=0.5, beta_bt=0.8, max_iter=100):
    """
    A function to calculate the optimal step size using the backtracking rule.

    :objective: function
        The function to calculate objective function
    :computegrad: function
        The function to calculate gradient for logistic regression
    :param x: dataframe or Matrix
        The features used to calculate the objective
    :param y: array
        The response from the data.
    :param beta: array
        An array of the beta values used to calculated the objective
    :param lambda: float
        The lambda, or regularization parameter, used to calculate the objective.
    :param eta: float
        The starting eta value to be optimized.
    :param alpha: float
        Constant used to define sufficient decrease condition
    :param beta_bt: float
        Fraction by which we decrease eta if the previous eta doesn't work
    :param max_iter: int
        Maximum number of iterations to run the algorithm
    """
    #calculate the gradient and then normalize
    grad_beta = computegrad(beta, lamda)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    itr = 0
    while(found_eta == 0 and itr < max_iter):
        #condition to be met for eta update
        if (objective((beta-eta*grad_beta),lamda,x,y) < objective(beta,lamda,x,y) - alpha*eta*norm_grad_beta**2):
            found_eta = 1
        elif(itr == max_iter):
            print('Maximum number of iterations of backtracking reached')
            break
        else:
            eta = eta*beta_bt
            itr += 1
    return(eta)

def fastgradalgo(x, y, lamda, beta, eta_init, max_iter = 150):
    """
    Function that uses fast gradient descent to calculate the optimized beta
    values used that minimizes the objective function.

    :param x: dataframe or Matrix
        The features used to calculate the objective
    :param y: array
        The response from the data.
    :param lambda: float
        The lambda, or regularization parameter, used to calculate the objective.
    :param beta: array
        An array of the beta values used to calculated the objective
    :param eta_init: float
        The initial eta used that gets updated by the backtracking function
    :param max_iter: int
        The maximum number of iterations to use before stopping the function.
    """
    beta_new = copy.deepcopy(beta)
    #initialize the thetas to 0
    theta = beta
    theta_vals = [theta]
    beta_vals = [beta]
    itr = 0
    eta=eta_init
    while(itr < max_iter):
        #update eta using backtracking rule
        eta = backtracking(x,y,beta_new,lamda,eta)
        #the fast gradient function
        grad_beta = computegrad(theta_vals[itr],lamda,x,y)
        beta_new = theta_vals[itr] - eta*grad_beta
        beta_vals.append(np.array(beta_new))
        theta_new = beta_vals[itr+1] + (itr/(itr+3))*(beta_vals[itr+1] - beta_vals[itr])
        theta_vals.append(np.array(theta_new))
        #update the beta with the results of this run, now rinse and repeat
        beta = beta_new
        itr += 1
    return(beta_vals)

lamda = 1
d = x_train.shape[1]
eta_0 = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train),
eigvals=(d-1, d-1), eigvals_only=True)[0]+lamda)
print('Found eta from Lippshitz:')
print(eta_0)
print('')
##print beta output
betas = fastgradalgo(x_train, y_train, lamda=0.1, beta=np.zeros(x_train.shape[1]),
eta_init=eta_0, max_iter = 150)
final_betas = betas[-1]
print('Final betas:')
print(final_betas)
print('')
##print predictions
predictions = np.dot(x_test, final_betas)
predictions = preprocessing.binarize(predictions.reshape(-1,1), threshold=0)
predictions = preprocessing.label_binarize(np.ravel(predictions), [0,1], neg_label=0)
predictions = np.squeeze(predictions)
print('Predictions on test set:')
print(predictions)
print('')
##print accuracy
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:')
print(accuracy)
print('')
##print confusion Matrix
con_matrix = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(con_matrix)
print('')

##print plot visualizing the training procedure
objecfast = []
for i in range(len(betas)):
    objecfast.append(objective(x=x_train, y=y_train, beta=betas[i], lamda=0.1))
objecfast = pd.DataFrame(objecfast)
plt.plot(objecfast)
plt.title('Visualizing the Training Process')
plt.xlabel('Iteration')
plt.ylabel('Objective')

lamda = 0.1
lr_mod = sklearn.linear_model.LogisticRegression(fit_intercept=False,
C=1/(2*lamda*x_train.shape[0]), max_iter=150)
lr_mod.fit(x_train, y_train)
sk_predict = lr_mod.predict(x_test)

##print coefs
print('SKLearn Betas')
print(lr_mod.coef_)

##print accuracy
accuracy = accuracy_score(y_test, sk_predict)
print('SkLearn Accuracy:')
print(accuracy)
print('')

##print confusion Matrix
con_matrix = confusion_matrix(y_test, sk_predict)
print('SKLearn Confusion Matrix:')
print(con_matrix)
print('')

plt.show(block=True)
