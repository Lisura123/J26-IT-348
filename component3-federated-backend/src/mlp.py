import numpy as np

LAYOUT = [("W1", (8, 16)), ("b1", (16,)), ("W2", (16, 5)), ("b2", (5,))]   # same order as shared files
ACT = {
    "relu":    (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
    "tanh":    (np.tanh,                     lambda z, a: 1 - a ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
}

def init_params(rng, activation="relu"):
    scale1 = np.sqrt(2.0 / 8) if activation == "relu" else np.sqrt(1.0 / 8)
    scale2 = np.sqrt(1.0 / 16)
    return {"W1": rng.normal(0, scale1, (8, 16)), "b1": np.zeros(16),
            "W2": rng.normal(0, scale2, (16, 5)), "b2": np.zeros(5)}

def flatten(p):
    return np.concatenate([np.asarray(p[n], float).ravel() for n, _ in LAYOUT])

def unflatten(vec):
    out, i = {}, 0
    for n, shape in LAYOUT:
        size = int(np.prod(shape))
        out[n] = vec[i:i + size].reshape(shape).copy()
        i += size
    return out

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(p, X, activation="relu"):
    f, _ = ACT[activation]
    z1 = X @ p["W1"] + p["b1"]
    a1 = f(z1)
    return z1, a1, _softmax(a1 @ p["W2"] + p["b2"])

def evaluate(p, X, y, activation="relu"):
    _, _, prob = forward(p, X, activation)
    loss = -np.log(prob[np.arange(len(y)), y] + 1e-12).mean()
    pred = prob.argmax(axis=1)
    per_class = {c: float((pred[y == c] == c).mean()) if (y == c).any() else float("nan")
                 for c in range(5)}
    return loss, float((pred == y).mean()), per_class

def train_local(p, X, y, rng, activation="relu", epochs=3, batch=32, lr=0.05):
    """Mini-batch SGD. Returns (new_params, delta) where delta = new - old (flattened)."""
    start = flatten(p)
    p = {k: v.copy() for k, v in p.items()}
    _, dact = ACT[activation]
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb, yb = X[idx], y[idx]
            z1, a1, prob = forward(p, xb, activation)
            dz2 = prob.copy()
            dz2[np.arange(len(yb)), yb] -= 1
            dz2 /= len(yb)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            dz1 = (dz2 @ p["W2"].T) * dact(z1, a1)
            dW1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)
            p["W1"] -= lr * dW1; p["b1"] -= lr * db1
            p["W2"] -= lr * dW2; p["b2"] -= lr * db2
    return p, flatten(p) - start