import numpy as np


def solve(x):
    """Implement Logistic Sigmoid from the mathematical contract in README.md."""
    x=np.array(x,dtype=float)
    return 1/(1+np.exp(-x))
