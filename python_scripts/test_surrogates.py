import numpy as np
import pandas as pd

X = np.load('sim_results/surrogate_history_x.npy')
F = np.load('sim_results/surrogate_history_f.npy')

X_train = X[:4000]
F_train = F[:4000]
X_val = X[4000:]
F_val = X[4000:]


from surrogate_models import *

xl = X_train.min(0)
xu = X_train.max(0)
expected_mean = F_train.mean(0)
expected_std = F_train.std(0)
n_var = X_train.shape[1]
n_obj = F_train.shape[1]


model = SurrogateManifoldProblem(n_var, n_obj, xl, xu, expected_mean, expected_std)
model.fit(X_train, F_train)
preds = model.predict(X_val)

error = abs((preds - F_val)/F_val)
print(pd.Series(error.flatten()).describe())


