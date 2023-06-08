import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray: 
    """sigmoid function with numpy

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        sigmoid result matrix
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """derivative of sigmoid function

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        sigmoid derivative result matrix
    """
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x: np.ndarray) -> np.ndarray:
    """tanh function

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        tanh result matrix
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """derivative of tanh function

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        tanh derivative result matrix
    """
    return 1 - tanh(x) * tanh(x)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLu function

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        ReLu result matrix
    """
    return np.maximum(x, 0)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """derivative of ReLu function

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        ReLu derivative result matrix
    """
    return (x > 0).astype(x.dtype)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function

    Parameters
    ----------
    x : np.ndarray
        operate matrix

    Returns
    -------
    np.ndarray
        softmax result matrix
    """ 
    return np.exp(x) / np.sum(np.exp(x))