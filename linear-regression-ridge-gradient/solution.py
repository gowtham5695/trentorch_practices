import numpy as np

from _load import load_solution
mse_gradient=load_solution("linear-regresssion-mse-gradient").mse_gradient
def ridge_grad(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute the gradient of the MSE loss with L2 regularization.
    """
    # TODO: Start from mse_gradient()'s result, then add the L2 penalty
    # term to grad_weight only. Leave grad_bias unchanged -- bias is
    # never regularized.
    grad_weight,grad_bias=mse_gradient(input,weight,bias,target)
    return grad_weight+2*lam*weight,grad_bias
