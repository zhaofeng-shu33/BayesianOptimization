import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events

class DataVisualization:
    def __init__(self, optimizer):
        self.x = np.empty(shape=(0, 2))
        self.func = optimizer._space.target_func
        self.optimizer = optimizer
    def plot(self):
        self.x = self.optimizer._space._params
        delta = 0.025
        x = np.arange(0.0, 6.0, delta)
        y = np.arange(0.0, 6.0, delta)
        X, Y = np.meshgrid(x, y)
        Z_1, Z_2 = self.func(X, Y)
        plt.contourf(X, Y, Z_2 > 0)
        plt.colorbar()
        feasible_index_set = np.where(optimizer._space._constraint_target <= 0)
        infeasible_index_set = np.where(optimizer._space._constraint_target > 0)
        plt.scatter(self.x[feasible_index_set, 0], self.x[feasible_index_set, 1], color='white', marker='o', s=10, edgecolors='black')
        plt.scatter(self.x[infeasible_index_set, 0], self.x[infeasible_index_set, 1], marker='x', s=12, edgecolors='black', color='black')

        plt.show()

def black_box_function_1(x, y):
    func_val = np.cos(2 * x) * np.cos(y) + np.sin(x)
    constraint_val = np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) + 0.5
    return (-func_val, constraint_val)

def black_box_function_2(x, y):
    func_val = np.sin(x)
    constraint_val = np.sin(x) * np.sin(y) + 0.95
    return (-func_val, constraint_val)

# Bounded region of parameter space
pbounds = {'x': (0, 6), 'y': (0, 6)}

optimizer = BayesianOptimization(
    f=black_box_function_2,
    pbounds=pbounds,
    random_state=1,
)

vis = DataVisualization(optimizer)


optimizer.maximize(
    init_points=0,
    n_iter=30,
    acq='ei'
)
print(np.sum(optimizer._space._constraint_target <= 0) / 30)
vis.plot()