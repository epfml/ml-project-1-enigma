import numpy as np


def mean_square_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    Arguments:
        y: Numpy array of shape (N,)
        tx: Numpy array of shape (N,D), D is the number of features
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
    """
    # TODO


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    Arguments:
        y: Numpy array of shape (N,)
        tx: Numpy array of shape (N,D), D is the number of features
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
    """
    # TODO


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Arguments:
        y: Numpy array of shape (N,)
        tx: Numpy array of shape (N,D), D is the number of features
    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
    """
    # TODO


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Arguments:
        y: Numpy array of shape (N,)
        tx: Numpy array of shape (N,D), D is the number of features
        lambda_: The regularization parameter
    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
    """
    # TODO


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD (y ∈ {0, 1})
    Arguments:
        y: Numpy array of shape (N,)
        tx: Numpy array of shape (N,D), D is the number of features
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
    """
    # TODO


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ*∥w∥^2)
    Arguments:
        y: Numpy array of shape (N,)
        tx: Numpy array of shape (N,D), D is the number of features
        lambda_: The regularization parameter
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
    """
    # TODO
