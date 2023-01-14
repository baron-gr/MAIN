import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
y = x**2
plt.plot(x, y)       # Plot the sine of each x point

plt.show()           # Display the plot