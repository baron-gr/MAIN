import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-20,20,100)

class functions:
    def __init__(self, min_x, max_x, x_spacing):
        self.min_x = min_x
        self.max_x = max_x
        self.x_spacing = x_spacing

    def wave_func(self, x_vals, y_vals):
        self.x_vals = x_vals
        self.y_vals = y_vals

    # def cubic_func(self, y):
    #     self.y = y


plt.plot()
plt.show