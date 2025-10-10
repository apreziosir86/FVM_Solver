import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Problema de adveciión difusión en 1D con condiciones de contorno 
# periódicas y un pulso de contaminante a la entrada

# Definiciones iniciales
x0 = 0                             # Coordenada inicial x
xL = 50                            # Coordenada final x
Xi = 10                            # Punto de inyección
T0 = 0                             # Tiempo inicial (s)
TF = 15                            # Tiempo final (s)
u = 2.                             # Velocidad del flujo
rho = 1                            # 
dx = 2                             # Tamaño del enmallado
dt = 0.5                           # Tamaño de paso de tiempo
M = 50                             # Masa inicialde contaminante
Tau = 0.8                          # Difusividad

# ==============================================================================
# Algunas funciones rápidas para uso en el problema
# ==============================================================================
# Función Peclet para series de potencias
def PotPe(Pe):

    if Pe < -10.: return -Pe
    elif Pe >= -10 and Pe < 0: return (1 + 0.1 * Pe) ** 5 - Pe
    elif Pe >= 0 and Pe <= 10: return (1 - 0.5 * Pe) ** 5
    else: return 0.0

# Solución analítica 
def Analitica(M, D, X, x0, t, u):

    const = M / (np.sqrt(4 * np.pi * D * t))
    exponencial = np.exp(-(X - x0 - u * t) ** 2 / (4 * D * t))

    return const * exponencial

# ==============================================================================
# Armando los vectores de tiempo y espacio
T = np.arange(T0, TF + dt, dt)
X = np.arange(x0, xL + dx, dx)
C0 = np.zeros_like(X)
C0[np.abs(X - Xi) < dx / 2] = M / dx 


plt.show()

# Pintando la solución analítica
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

# Iniciando la gráfica
P_analit, = axs[0].plot([], [], label='Sol. analítica')

axs[0].set_ylim((0, (M / dx) * 1.05))
axs[0].set_xlim((x0, xL))
axs[0].set_ylabel(r'Concentración $(kg/m)$')
axs[0].grid()
axs[0].legend()

# Iterando en el tiempo y viendo la solución
for t in range(1, len(T)):

    # Calculando la solución analítica
    phi_a = Analitica(M, Tau, X, Xi, T[t], u)
    P_analit.set_data(X, phi_a)

    plt.draw()
    plt.pause(1)




