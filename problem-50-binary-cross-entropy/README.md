# Binary Cross-Entropy

Beginner | problemset | classical-ml | logistic-regression

Compute the mean binary cross-entropy from real-valued logits and 0/1 labels using a numerically stable formula that never evaluates the sigmoid directly.

Implement `solve(logits, labels)`.

**Returns.** Return a non-negative Python float. Very large logits must not overflow.

### Examples

**Example 1**

Input:

```python
solve([0.0, 0.0], [1.0, 0.0])
```

Output:

```text
0.693147
```

**Example 2**

Input:

```python
solve([2.0, -2.0, 1000.0], [1.0, 0.0, 1.0])
```

Output:

```text
0.084619
```
