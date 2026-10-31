import numpy as np


def mse_gradient(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    input:  shape (batch_size, in_features)
    weight: shape (out_features, in_features)
    bias:   shape (out_features,), or None
    target: shape (batch_size, out_features)

    Returns:
        grad_weight: same shape as weight
        grad_bias: same shape as bias, or None if bias is None
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    pred=(input@weight.T)
    if bias is not None:
        pred=pred+bias
    loss=pred-target
    n=target.size

    grad_wei=(2/n)*(loss.T@input)

    if bias is not None:
        grad_bias=(2/n)*np.sum(loss,axis=0)
    else:
        grad_bias=None
    return grad_wei , grad_bias
