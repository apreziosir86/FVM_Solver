import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Funciones útiles para procesos dentro del script
#  =============================================================================
def sol_analitica(x, t, Dm, M, Xi):
    """Solución analítica del problema de difusión no estacionaria en 1D con 
    condición inicial de una masa puntual M en la posición Xi.
    """
    return (M / np.sqrt(4 * np.pi * Dm * t)) * np.exp(-(x - Xi) ** 2 /
            (4 * Dm * t))

#  =============================================================================
# Variables físicas del problema
#  =============================================================================
X0 = -10                                  # Frontera izquierda (m)
XL = 10                                   # Frontera derecha (m)
Xi = 0                                    # Posición de inyección (m)
Dm = 2e-3                                 # Difusividad (m2/s)
M = 2.                                    # Masa inyectada (kg)
tmax = 3000                               # Tiempo final de simulación (s)

#  =============================================================================
# Variables numéricas - control de la simulación
#  =============================================================================
dx = 0.5                                  # Tamaño de paso (m)
S = 0.45                                  # Número de estabilidad (-)
dt = S * dx ** 2 / Dm                     # Paso de tiempo (s)
f = 1                                     # Explícito: 0 Implícito: 1 

#  =============================================================================
# Haciendo el dominio espacial y la condición inicial
#  =============================================================================
X = np.arange(X0, XL + dx, dx)            # Vector de posiciones
T = np.arange(0, tmax + dt, dt)           # Vector de tiempos
C0 = np.zeros_like(X)                     # Concentración espacial inicial
C0[np.abs(X - Xi) < dx / 2] = M / dx      # Condición inicial (kg/m)
error_a = np.zeros_like(X)                # Vector de errores absolutos
error_r = np.zeros_like(X)                # Vector de errores relativos
norm_err_a = []                           # Norma del error absoluto (tamaño)
norm_err_r = []                           # Norma del error relativo (tamaño)

#  =============================================================================
# Imprimiendo el valor del paso de tiempo en la terminal
#  =============================================================================
print(f'Tamaño de paso de tiempo: {dt:.4f} s')
print(f'Número de pasos de tiempo: {len(T)}')

#  =============================================================================
# Solicitando al usuario la cada cuántos pasos quiere guardar resultados en el 
# dataframe
#  =============================================================================
n = int(input('¿Cada cuántos pasos de tiempo quiere guardar resultados? '))

#  =============================================================================
# Creando un dataframe para guardar los resultados y otro para los errores
#  =============================================================================
datos = pd.DataFrame({
    'X': X,
    'C ini (kg/m)': C0
})

errores = pd.DataFrame({
    'X (m)': X    
})

#  =============================================================================
# Graficando la condición inicial
plt.figure(figsize=(8, 6))
plt.plot(X, C0, label='Condición inicial', color='dodgerblue')
plt.xlabel('Posición (m)')
plt.ylabel('Concentración (kg/m)')
plt.title('Condición inicial del problema de difusión no estacionaria 1D')
plt.legend()
plt.grid()
plt.tight_layout()
plt.draw()
plt.pause(0.1)

# Cerrando el gráfico y continuando con el recorrido del programa
input('Presione ENTER para continuar...')
plt.close()

#  =============================================================================
# Abriendo la instancia del gráfico de salida y configurándolo
fig, axs =plt.subplots(1, 3, figsize=(15, 8))

p_analit, = axs[0].plot([], [], label='Solución exacta',
            color='orangered', ls='--')

p_numer, = axs[0].plot([], [], label='Solución numérica', 
            color='dodgerblue', ls='-')

err_abs, = axs[1].semilogy([], [], label='Error absoluto', color='turquoise')

err_rel, = axs[1].semilogy([], [], label='Error relativo', color='olivedrab')

p_nerr_a, = axs[2].semilogy([], [], label='Norma error absoluto',
                            color='indigo')

p_nerr_r, = axs[2].semilogy([], [], label='Norma del error relativo',
                            color='blueviolet')

# Cuadrando la gráfica de los resultados
axs[0].set_xlim((X0, XL))
axs[0].set_xlabel('Posición (m)')
axs[0].set_ylim((0, M / dx * 1.1))
axs[0].set_ylabel('Concentración (kg/m)')
axs[0].legend()
axs[0].grid()

# Cuadrando la gráfica del error
axs[1].set_xlim((X0, XL))
axs[1].set_xlabel('Posición (m)')
axs[1].set_ylabel('Error relativo')
axs[1].legend(loc='lower center')
axs[1].grid()
axs[1].set_ylim((1e-6, 1))

# Cuadrando la gráfica de la norma del error
axs[2].set_xlim((0, tmax))
axs[2].set_xlabel('Tiempo (s)')
axs[2].set_ylabel('Norma del error')
axs[2].set_ylim((1e-3, 100))
axs[2].legend(loc='upper right')
axs[2].grid()

#  =============================================================================
# Calculando los coeficientes que no cambian en el tiempo (se construyen b para
# poder tener el Cp libre en un lado del igual)
bE = np.zeros(len(X) - 2)
bW = np.zeros_like(bE)
bP = np.zeros_like(bE)

# Llenando los coeficientes en los vectores que se asocian a cada nodo
for i in range(1, len(X) - 1):

    bE[i - 1] = Dm * dt / (dx * (X[i + 1] - X[i]))
    bW[i - 1] = Dm * dt / (dx * (X[i] - X[i - 1]))
    bP[i - 1] = 1 - bE[i - 1] - bW[i - 1]

# Armando una matriz y un vector de mano derechapara poder solucionar por el
# método implícito
A = np.zeros((len(X), len(X)))

# Llenando la matriz de coeficientes
for i in range(1, len(A) - 1):

    A[i, i - 1] = -bW[i - 1]
    A[i, i + 1] = -bE[i - 1]
    A[i, i] = 1 + bE[i - 1] + bW[i - 1]

# Condiciones de frontera
A[0, 0] = 1
A[-1, -1] = 1

# Iterando en el tiempo
for t in range(1, len(T)):

    # Solución analítica
    Can = sol_analitica(X, T[t], Dm, M, Xi)

    # Calculando solución por método explícito
    if f != 1:

        # Solución numérica explícita
        C1_e = C0.copy()

        # Impongo condiciones de contorno
        C1_e[0] = 0
        C1_e[-1] = 0

        # Iterando para el paso de tiempo
        for i in range(1, len(X) - 1):
            C1_e[i] = bP[i - 1] * C0[i] + bW[i - 1] * C0[i - 1]
            C1_e[i] += bE[i - 1] * C0[i + 1]
    
    else: C1_e = np.zeros_like(X)

    # Calculando solución por método implícito
    if f != 0:

        # Arreglando el vector C0 para que funcione como vector de mano derecha
        # en el sistema lineal de ecuaciones
        C0[0] = 0
        C0[-1] = 0

        # Encontrando la solución al sistema de ecuaciones
        C1_i = np.linalg.solve(A, C0)

    else: C1_i = np.zeros_like(X)

    # Ponderando la solución implícita con la explícita
    C1 = f * C1_i + (1 - f) * C1_e

    # Reemplazando los valores de concentración para siguiente paso de tiempo
    C0 = C1.copy()

    # Calculando el error absoluto y relativo
    error_a = np.abs(Can - C1)
    error_r[1:-1] = np.abs((Can[1:-1] - C1[1:-1]) / Can[1:-1])
    error_r[error_r == np.nan] = 0

    # Calculando la norma del error absoluto y relativo
    norm_err_a.append(np.linalg.norm(error_a))
    norm_err_r.append(np.linalg.norm(error_r, np.inf))    

    # Graficando los resultados del paso de tiempo
    p_analit.set_data(X, Can)
    p_numer.set_data(X, C1)
    err_abs.set_data(X, error_a)
    err_rel.set_data(X, error_r)
    p_nerr_a.set_data(T[1:t + 1], norm_err_a)
    p_nerr_r.set_data(T[1:t + 1], norm_err_r)
    plt.suptitle(f'Resultados para t = {T[t]:.3f} s')
    plt.draw()
    plt.pause(0.25)

    # Agregando columna al dataframe (si es necesario)
    if t % n == 0:
        
        # Haciendo el label de la columna
        datos[f'C(t = {T[t]:.3f} s)'] = C0
        errores[f'C(t = {T[t]:.3f} s)'] = error_a

# Cerrando todo lo que se había abierto

# Guardando el dataframe de resultados en excel
datos.to_excel('Concentraciones_Salida.xlsx', index=False)
errores.to_excel('Errores_Salida.xlsx', index=False)

# Terminando el programa
input('Simulación finalizada, presione Enter')
plt.close()