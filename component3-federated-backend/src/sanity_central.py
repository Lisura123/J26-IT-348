import numpy as np
from src import pv_data, mlp

site_data, (Xv, yv), scaler = pv_data.get_data()
X = np.vstack([site_data[s][0] for s in pv_data.SITES])
y = np.concatenate([site_data[s][1] for s in pv_data.SITES])
print("pooled train:", X.shape, " validation (OPERATION):", Xv.shape)

for act in ["relu", "tanh", "sigmoid"]:
    rng = np.random.default_rng(0)
    p = mlp.init_params(rng, act)
    for epoch in range(1, 61):
        p, _ = mlp.train_local(p, X, y, rng, act, epochs=1)
        if epoch in (1, 5, 20, 60):
            loss, acc, _ = mlp.evaluate(p, Xv, yv, act)
            print(f"{act:8s} epoch {epoch:3d}  val_loss {loss:.4f}  val_acc {acc:.4f}")