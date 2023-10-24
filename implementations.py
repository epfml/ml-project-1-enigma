import numpy as np


def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
    # Initialize weights
    w = initial_w

    # Compute the error term
    e = y - tx @ w

    # Loop for max_iters iterations
    for _ in range(max_iters):
        # Compute the error
        e = y - np.dot(tx, w)

        # Compute the gradient
        gradient = -np.dot(tx.T, e) / len(y)

        # Update the weights using the gradient and step size (gamma)
        w = w - gamma * gradient

    # Compute the MSE loss
    loss = 0.5 * np.mean(e ** 2)

    return w, loss


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
    # Initialize weights
    w = initial_w

    # Loop for max_iters iterations
    for _ in range(max_iters):
        # Randomly select a sample index
        i = np.random.randint(0, len(y))

        # Compute the error for the selected sample
        e_i = y[i] - np.dot(tx[i], w)

        # Compute the gradient for the selected sample
        gradient_i = -tx[i] * e_i

        # Update the weights using the gradient of the selected sample and step size (gamma)
        w = w - gamma * gradient_i

    # Compute the overall MSE loss for all samples after the updates
    e = y - np.dot(tx, w)
    loss = 0.5 * np.mean(e ** 2)

    return w, loss


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
    # Perform Singular Value Decomposition to optimize over large datasets
    U, S, VT = np.linalg.svd(tx, full_matrices=False)

    # Calculate the weight vector using SVD
    w = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

    # Compute the MSE loss
    e = y - np.dot(tx, w)
    loss = 0.5 * np.mean(e ** 2)

    return w, loss


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
    # Calculate the weight vector using the normal equations for ridge regression
    D = tx.shape[1]  # Number of features
    w = np.linalg.inv(tx.T @ tx + lambda_ * np.eye(D)) @ tx.T @ y

    # Compute the MSE loss with regularization
    e = y - np.dot(tx, w)
    loss = 0.5 * np.mean(e ** 2) + 0.5 * lambda_ * np.sum(w ** 2)

    return w, loss


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
    # Initialize w
    w = initial_w

    for _ in range(max_iters):
        # Compute the predictions
        predictions = sigmoid(tx @ w)

        # Compute the gradient
        gradient = (tx.T @ (predictions - y))/len(y)

        # Update the weights
        w = w - gamma * gradient

    # Compute the logistic regression loss (cross-entropy loss)
    loss = np.mean(np.log(1 + np.exp(tx.dot(w))) - y * (tx.dot(w)))

    return w, loss


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
    # Initialize the weights with the provided initial values
    w = initial_w

    # Gradient Descent for max_iters iterations
    for _ in range(max_iters):
        # Compute the predicted probabilities
        predictions = sigmoid(tx @ w)

        # Compute the gradient of the regularized logistic loss
        gradient = (tx.T @ (predictions - y)/len(y) + 2 * lambda_ * w)

        # Update the weights
        w = w - gamma * gradient

    # Compute the regularized logistic loss (negative log-likelihood with regularization)
    loss = np.mean(np.log(1 + np.exp(tx.dot(w))) - y * (tx.dot(w)))

    return w, loss
