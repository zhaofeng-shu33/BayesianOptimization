import numpy as np

from bayes_opt import BayesianOptimization


def black_box_function_1(x, y):
    func_val = np.cos(2 * x) * np.cos(y) + np.sin(x)
    constraint_val = np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) - 0.5
    return (-func_val, constraint_val)

# Bounded region of parameter space
pbounds = {'x': (0, 6), 'y': (0, 6)}

optimizer = BayesianOptimization(
    f=black_box_function_1,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=0,
    n_iter=30,
    acq='ei'
)
