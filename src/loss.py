import numpy as np


def gaussian_loss(y_pred, data):
    y_true = data.get_label()

    loglikelihood = (
        -0.5 * np.log(2 * np.pi) - 0.5 * y_pred - 0.5 / np.exp(y_pred) * y_true**2
    )
    # remember that boosting minimizes the loss function, but we want to maximize the loglikelihood
    # thus, need to return the negative loglikelihood to the Boosting algorithm
    # also applies to gradient and hessian below

    return "loglike", -loglikelihood, False


def gaussian_loss_gradhess(y_pred, data):
    y_true = data.get_label()

    exp_pred = np.exp(y_pred)

    # pay attention to the chain rule as we exp() the Boosting output before plugging it into the loglikelihood
    grad = -0.5 + 0.5 / exp_pred * y_true**2
    hess = -0.5 / exp_pred * y_true**2

    return -grad, -hess
