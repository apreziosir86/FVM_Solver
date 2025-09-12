import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constants values
x0 = 0                          # Coordenada inicio (m)
xL = 1                          # Coordenada final (m)
T0 = 0                          # Tiempo inicial (s)
Tf = 0.5                        # Tiempo final (s)

# Parámetros numéricos de la solución
dx = 0.05                        # Paso espacial (m)
S = 0.45                         # Número de estabilidad
dt = S * dx**2                   # Paso temporal (s)

print(f'El paso de tiempo será de: {dt:.4f} s')
input('Presione enter para continuar... ')

#  Solución analítica
def analytical_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

# Creando el vector de coordenadas espaciales y temporales
x = np.arange(x0, xL + dx, dx)
t = np.arange(T0, Tf + dt, dt)
error = np.zeros_like(x)
error_ac = []

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
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Gráfico de las soluciones calculadas
line1, = ax[0].plot([], [], color='dodgerblue', ls='--',
                   label='Solución analítica')
line2, = ax[0].plot([], [], color='salmon', label='Solución numérica')
ax[0].set_xlim((x0, xL))
ax[0].set_ylim((0, 1))
ax[0].set_ylabel('u(x,t)')
ax[0].legend()
ax[0].grid()
ax[0].set_xlabel('x (m)')

# Gráfico del error relativo en cada paso de tiempo
line3, = ax[1].semilogy([], [], color='dodgerblue', ls='--',
                        label='Error relativo')
ax[1].set_xlim((x0, xL))
ax[1].set_ylim((1e-6, 1))
ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('Error relativo')
ax[1].legend()
ax[1].grid()

# Gráfico de evolución de la norma del error
line4, = ax[2].semilogy([], [], color='salmon', label='Norma del error')
ax[2].set_xlim((T0, Tf))
ax[2].set_ylim((1e-6, 1))
ax[2].set_xlabel('t (s)')
ax[2].set_ylabel('Norma del error')
ax[2].set_title('Evolución de la norma del error')
ax[2].legend()
ax[2].grid()

plt.tight_layout()
plt.ion()
plt.show()

# Iterando sobre t para graficar la solución analítica
for n in range(len(t)):
    
    # Solución analítica
    u_analytical = analytical_solution(x, t[n])
    line1.set_data(x, u_analytical)
    
    # Solución numérica por diferencias finitas
    if n == 0:

        u_1 = u_0.copy()
    
    else:

        u_1 = u_0.copy()
        
        for i in range(1, len(x) - 1):
        
            u_1[i] = (u_0[i + 1] +u_0[i - 1]) * S + (1 - 2 * S) * u_0[i]
        
        # Imponiendo condiciones de frontera
        u_1[0] = 0
        u_1[-1] = 0

        u_0 = u_1.copy()

    # Calculando el error en el paso de tiempo
    error[1:-1] = np.abs((u_analytical[1:-1] - u_0[1:-1]) / u_analytical[1:-1])
    
    # Calculando la norma del error y guardándola en un vector que acumule estos
    # resultados
    error_ac.append(np.linalg.norm(error, 2))

    # Cuadrando los datos para las gráficas del proceso
    line2.set_data(x, u_0)
    line3.set_data(x, error)
    line4.set_data(t[:n + 1], error_ac)
    ax[0].set_title(f'Solución numérica en t = {t[n]:.2f} s')
    ax[1].set_title(f'Error relativo en t = {t[n]:.2f} s')
    plt.draw()
    plt.pause(0.15)

input('Simulación finalizada.Presione enter para continuar...')
plt.close()