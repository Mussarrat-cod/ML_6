import numpy as np

# Data
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float) / 100

# Normalize
X = X / np.max(X, axis=0)

# Functions
def sig(x):
    return 1 / (1 + np.exp(-x))

def dsig(x):
    return x * (1 - x)

# Weights
wh = np.random.rand(2, 3)
wout = np.random.rand(3, 1)
bh = np.random.rand(1, 3)
bout = np.random.rand(1, 1)

# Training
for i in range(5000):
    h = sig(np.dot(X, wh) + bh)
    o = sig(np.dot(h, wout) + bout)

    d_o = (y - o) * dsig(o)
    d_h = np.dot(d_o, wout.T) * dsig(h)

    wout += np.dot(h.T, d_o) * 0.1
    wh += np.dot(X.T, d_h) * 0.1
    bout += np.sum(d_o, axis=0)
    bh += np.sum(d_h, axis=0)

# Output
print("\nNormalized Input:\n", X)
print("Actual:\n", y)
print("Predicted:\n", o)