import numpy as np
from random import shuffle
import scipy.sparse


def softmax_loss_naive(theta, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - theta: d x K parameter matrix. Each column is a coefficient vector for class k
    - X: m x d array of data. Data are d-dimensional rows.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to parameter matrix theta, an array of same size as theta
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in J and the gradient in grad. If you are not              #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization term!                                                  #
    ##########################################################################
    s = X.dot(theta)
    ex = np.exp(s)
    for i in range(m):
        avg = ex[i] / np.sum(ex[i, :])
        J += -1 * np.log(avg[y[i]])

        avg[y[i]] -= 1
        for j in range(theta.shape[1]):
            grad[:, j] += X[i, :] * avg[j]

    J /= m
    J += 0.5 * reg * np.sum(theta * theta) / m
    grad /= m
    grad += reg * theta / m

    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return J, grad


def softmax_loss_vectorized(theta, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    ##########################################################################
    s = X.dot(theta)
    ex = np.exp(s)
    row_sum = np.sum(ex, axis=1, keepdims=True)
    avg = ex / row_sum
    data_loss = np.sum(-np.log(avg[range(m), y]))
    reg_loss = np.sum(0.5 * reg * theta * theta)
    J = data_loss / m + reg_loss / m

    Gavg = avg
    Gavg[range(m), y] -= 1
    grad = np.dot(X.T, Gavg)
    grad /= m
    grad += reg * theta / m
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return J, grad
