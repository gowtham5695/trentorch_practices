import numpy as np


def gd_step(
    weight: np.ndarray,
    bias: np.ndarray | None,
    grad_weight: np.ndarray,
    grad_bias: np.ndarray | None,
    lr: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Apply one gradient-descent update.

    Returns:
        updated_weight, updated_bias
    """
    # TODO: Implement the update rule from Theory.
    # Return new values -- do not mutate weight or bias in place.
    # bias=None means there's no bias parameter -- updated_bias stays None.
    new_wei=weight-lr*grad_weight

    if bias is not None:
        new_bias=bias-lr*grad_bias
    else:
        new_bias=None
    return new_wei, new_bias
