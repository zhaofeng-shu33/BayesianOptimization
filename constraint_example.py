import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events

class DataVisualization:
    def __init__(self, func):
        self.x = np.empty(shape=(0, 2))
        self.func = func
    def update(self, event, instance):
        params = instance.res[-1]['params']
        new_x = np.array([params['x'], params['y']])
        self.x = np.concatenate([self.x, [new_x]])
    def plot(self):
        delta = 0.025
        x = np.arange(0.0, 6.0, delta)
        y = np.arange(0.0, 6.0, delta)
        X, Y = np.meshgrid(x, y)
        Z_1, Z_2 = self.func(X, Y)
        plt.contourf(X, Y, Z_1)
        plt.colorbar()
        plt.scatter(self.x[:, 0], self.x[:, 1], color='white', marker='o', s=6, edgecolors='black')
        plt.show()

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
vis = DataVisualization(black_box_function_1)

optimizer.subscribe(Events.OPTIMIZATION_STEP, vis)

optimizer.maximize(
    init_points=0,
    n_iter=30,
    acq='ei'
)

vis.plot()