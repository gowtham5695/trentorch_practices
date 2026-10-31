# Logistic Sigmoid

Beginner | problemset | classical-ml | logistic-regression

Compute the logistic sigmoid $\sigma(x)=1/(1+e^{-x})$ for an array of real logits **without overflow** for large positive or negative inputs.

Implement `solve(x)`.

**Returns.** Return a float NumPy array of the same shape with values in $[0, 1]$.

### Examples

**Example 1**

Input:

```python
solve([0.0])
```

Output:

```text
[0.5]
```

**Example 2**

Input:

```python
solve([2.0, -2.0])
```

Output:

```text
[0.880797, 0.119203]
```

**Example 3**

Input:

```python
solve([1000.0, -1000.0])
```

Output:

```text
[1.0, 0.0]
```
