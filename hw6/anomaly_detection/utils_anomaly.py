import numpy as np


def estimate_gaussian(X):
    """
    Estimate the mean and standard deviation of a numpy matrix X on a column by column basis
    """
    mu = np.zeros((X.shape[1],))
    var = np.zeros((X.shape[1],))
    ####################################################################
    #               YOUR CODE HERE                                     #
    ####################################################################
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=0)
    ####################################################################
    #               END YOUR CODE                                      #
    ####################################################################
    return mu, var


def select_threshold(yval, pval):
    """
    select_threshold(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    bestF1 = 0
    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval) + stepsize, max(pval), stepsize):

        ####################################################################
        #                 YOUR CODE HERE                                   #
        ####################################################################

        largerThanEps = pval < epsilon
        tp = np.sum(np.logical_and(largerThanEps == 1, yval.T == 1))
        fp = np.sum(np.logical_and(largerThanEps == 1, yval.T == 0))
        fn = np.sum(np.logical_and(largerThanEps == 0, yval.T == 1))

        prec = tp / float(tp + fp)
        recall = tp / float(tp + fn)

        F1 = (2 * prec * recall) / (prec + recall)

        if F1 > bestF1:
            bestF1 = F1
            best_epsilon = epsilon
        ####################################################################
        #                 END YOUR CODE                                    #
        ####################################################################
    return best_epsilon, bestF1
