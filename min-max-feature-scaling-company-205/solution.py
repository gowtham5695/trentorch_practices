import numpy as np

def solve(x):
    """Implement min-max-feature-scaling from the mathematical contract in README.md."""
    x=np.array(x,dtype=float)
    x_min=min(x)
    x_max=max(x)
    if x_min==x_max:
        return np.zeros(len(x),dtype=float)
    return (x-x_min)/(x_max-x_min)
