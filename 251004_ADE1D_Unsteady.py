import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Problema de adveción difusión en 1D con condiciones de contorno 
# periódicas y un pulso de contaminante a la entrada

# Definiciones iniciales
x0 = 0                             # Coordenada inicial x
xL = 100                           # Coordenada final x
Xi = 95                            # Punto de inyección
T0 = 0                             # Tiempo inicial (s)
TF = 30                            # Tiempo final (s)
u = 4.                             # Velocidad del flujo
rho = 1                            # 
dx = 0.1                           # Tamaño del enmallado
dt = 0.5                           # Tamaño de paso de tiempo
M = 500                            # Masa inicialde contaminante
Tau = 0.025                        # Difusividad

# ==============================================================================
# Algunas funciones rápidas para uso en el problema
# ==============================================================================
# Función Peclet para series de potencias
def PotPe(Pe):

    if Pe < -10.: return -Pe
    elif Pe >= -10 and Pe < 0: return (1 + 0.1 * Pe) ** 5 - Pe
    elif Pe >= 0 and Pe <= 10: return (1 - 0.5 * Pe) ** 5
    else: return 0.0

# Solución analítica por el método de las imágenes
def Analitica(M, D, X, x0, t, u, L, n_images=10):
    """
    Solución analítica de advección-difusión en dominio periódico
    usando el método de imágenes para condiciones periódicas.
    """
    if t <= 0:
        # Para tiempo cero, devolvemos una aproximación de la delta
        result = np.zeros_like(X)
        idx = np.argmin(np.abs(X - x0))
        result[idx] = M / dx  # Similar a tu condición inicial numérica
        return result
    
    result = np.zeros_like(X)
    
    # Sumamos contribuciones de la imagen principal + imágenes periódicas
    for k in range(-n_images, n_images + 1):
        x_image = x0 + k * L  # Posición de la imagen k-ésima
        # Solución fundamental de advección-difusión
        factor = M / np.sqrt(4 * np.pi * D * t)
        exponente = -((X - x_image - u * t) ** 2) / (4 * D * t)
        result += factor * np.exp(exponente)
    
    return result

# ==============================================================================
# Armando los vectores de tiempo y espacio
T = np.arange(T0, TF + dt, dt)
X = np.arange(x0, xL, dx)
C0 = np.zeros_like(X)
C0[np.abs(X - Xi) < dx / 2] = M / dx 


plt.show()

# Pintando la solución analítica
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

# Iniciando la gráfica
P_analit, = axs[0].plot([], [], label='Sol. analítica')

axs[0].set_ylim((0, np.max(C0) / 10 ))
axs[0].set_xlim((x0, xL))
axs[0].set_ylabel(r'Concentración $(kg/m)$')
axs[0].grid()
axs[0].legend()

# Iterando en el tiempo y viendo la solución
for t in range(1, len(T)):

    # Calculando la solución analítica
    phi_a = Analitica(M, Tau, X, Xi, T[t], u, xL - x0)
    P_analit.set_data(X, phi_a)

    plt.draw()
    plt.pause(1)




