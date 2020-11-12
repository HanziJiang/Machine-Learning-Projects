from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    
    data = np.hstack((data, np.ones((len(data), 1))))
    y = sigmoid(np.dot(data, weights))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y

def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    
    loss = -1 * targets * np.log(y) - (1-targets) * np.log(1-y)
    ce = np.mean(loss)
    frac_correct = np.sum(np.equal(targets, np.around(y))) / float(len(y))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    
    data = np.hstack((data, np.ones((len(data), 1))))
    f = evaluate(targets, y)[0]
    df = np.dot(np.transpose(data), y - targets) / len(data)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y

def logistic_pen(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (M+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points (plus a penalty term), gradient of parameters, and the     #
    # probabilities given by penalized logistic regression.             #
    #####################################################################
    
    data = np.hstack((data, np.ones((len(data), 1))))
    weights[-1] = [0.0]
    f = evaluate(targets, y)[0] + hyperparameters["weight_regularization"] / 2.0 * np.linalg.norm(weights) ** 2
    df = np.matmul(np.transpose(data), y - targets) / len(data) + np.multiply(weights, hyperparameters["weight_regularization"])
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


