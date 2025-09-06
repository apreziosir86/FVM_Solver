import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constants values
x0 = 0                          # Coordenada inicio (m)
xL = 1                          # Coordenada final (m)
T0 = 0                          # Tiempo inicial (s)
Tf = 0.5                        # Tiempo final (s)

# Parámetros numéricos de la solución
dx = 0.10                        # Paso espacial (m)
S = 1.5                          # Número de estabilidad
dt = S * dx**2                   # Paso temporal (s)

print(f'El paso de tiempo será de: {dt:.4f} s')
input('Presione enter para continuar... ')

#  Solución analítica
def analytical_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

# Creando el vector de coordenadas espaciales y temporales
x = np.arange(x0, xL + dx, dx)
t = np.arange(T0, Tf + dt, dt)

# Conidción inicial del problema
u_0 = np.sin(np.pi * x)

# Gráfica de la condición inicial
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, u_0, label='Condición inicial')
ax.set_xlabel('x (m)')
ax.set_ylabel('u(x,0)')
ax.set_title('Condición inicial del problema')
ax.legend()
ax.grid()
plt.tight_layout()
plt.draw()
plt.pause(0.01)
input('Presione enter para continuar... ')
plt.close()

# Evolución de la solución analítica con gráfica

# Creando una instancia de figura
fig, ax = plt.subplots()
line, = ax.plot([], [], color='dodgerblue', ls='--', label='Solución analítica')
line2, = ax.plot([], [], color='salmon', label='Solución numérica')
ax.set_xlim((x0, xL))
ax.set_ylim((0, 1))
ax.legend()
ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('u(x,t)')
plt.tight_layout()
plt.ion()
plt.show()

# Iterando sobre t para graficar la solución analítica
for n in range(len(t)):
    
    # Solución analítica
    u_analytical = analytical_solution(x, t[n])
    line.set_data(x, u_analytical)
    
    # Solución numérica por diferencias finitas
    if n == 0:
        u_numerical = u_0.copy()
    else:
        u_numerical_new = u_numerical.copy()
        
        for i in range(1, len(x) - 1):
        
            u_numerical_new[i] = (u_numerical[i + 1] +u_numerical[i - 1]) * S +\
                                 (1 - 2 * S) * u_numerical[i]
        
        u_numerical = u_numerical_new.copy()

    line2.set_data(x, u_numerical)
    ax.set_title(f'Solución numérica en t = {t[n]:.2f} s')
    plt.draw()
    plt.pause(0.5)

input('Presione enter para continuar...')
plt.close()
