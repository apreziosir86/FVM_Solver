import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Definamos variables
D = 0.03
x = [10]
y = [10]
dt = 0.1
tf = 20

# Vector de tiempos
T = np.arange(0, tf + dt, dt)

# Figura
fig, ax = plt.subplots(figsize=(10, 10))
graf1, = ax.plot([], [], color='salmon')
ax.set_xlim((0, 20))
ax.set_ylim((0, 20))

for i in range(0, len(T)):

    xn = x[i] + np.random.randn() * np.sqrt(4 * D * dt)
    x.append(xn)
    yn = y[i] + np.random.randn() * np.sqrt(4 * D * dt)
    y.append(yn)
    graf1.set_data(x, y)
    plt.draw()
    plt.pause(0.3)

