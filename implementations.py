import numpy as np


def mean_square_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    Arguments:
        y:
        tx:
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        (w, loss): The last weight vector of the method, and the corresponding loss value (cost function)
    """
    # TODO


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    Arguments:
        y:
        tx:
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        (w, loss): The last weight vector of the method, and the corresponding loss value (cost function)
    """
    # TODO


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Arguments:
        y:
        tx:
    Returns:
        (w, loss): The last weight vector of the method, and the corresponding loss value (cost function)
    """
    # TODO


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Arguments:
        y:
        tx:
        lambda_: The regularization parameter
    Returns:
        (w, loss): The last weight vector of the method, and the corresponding loss value (cost function) NOT including the penalty term
    """
    # TODO


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD (y ∈ {0, 1})
    Arguments:
        y:
        tx:
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        (w, loss): The last weight vector of the method, and the corresponding loss value (cost function)
    """
    # TODO


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ*∥w∥^2)
    Arguments:
        y:
        tx:
        lambda_: The regularization parameter
        initial_w: The initial weight vector
        max_iters: The number of steps to run
        gamma: The step-size
    Returns:
        (w, loss): The last weight vector of the method, and the corresponding loss value (cost function) NOT including the penalty term
    """
    # TODO
