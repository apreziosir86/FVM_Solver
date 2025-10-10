import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definiendo las variables del problema
phi0 = 10
phiL = 2
L = 5
u = 20.
rho = 2
Tau = 1
dx = 0.25                            # Tamaño de celda de volumen (m)

# Calculando e imprimiendo el número de Peclet:
Pe = rho * u * L / Tau
print(f'Valor del número de Péclet general: {Pe:.4f}')

# ==============================================================================
# FUNCIONES ÚTILES PARA EL CÓDIGO

# Función para calcularla solución analítica del problema
def phi_x(x, phi0, phiL, Pe, L):

    return phi0 + (phiL - phi0) * ((np.exp(Pe * x / L) - 1) / (np.exp(Pe) - 1))

# Función para calcular error en cada método
def error_rel(x1, x2): 

    error = np.zeros_like(x1)

    error[1:-1] = np.abs((x2[1:-1] - x1[1:-1]) / x1[1:-1])

    return error

# Función para calcular el valor de la función de Peclet en exponencial
def Afunc(Pe):

    if abs(Pe) < 1e-12: return 1 - Pe / 2

    else: return Pe / (np.exp(Pe) - 1)

# Funcion para valor de funcion de peclet híbrido
def hibridoPe(Pe):

    if Pe > 2.: return 0.0
    elif Pe < -2.: return -Pe
    else: return 1 - 0.5 * Pe

# Función Peclet para series de potencias
def PotPe(Pe):

    if Pe < -10.: return -Pe
    elif Pe >= -10 and Pe < 0: return (1 + 0.1 * Pe) ** 5 - Pe
    elif Pe >= 0 and Pe <= 10: return (1 - 0.5 * Pe) ** 5
    else: return 0.0

# ==============================================================================
# Calculando a solución analítica con buena definición
Xa = np.linspace(0, L, 500)
phiA = phi_x(Xa, phi0, phiL, Pe, L)
# ==============================================================================
# ==============================================================================
# Definiendo los valor numéricos que quiero tener en la solución
X = np.arange(0, L + dx, dx)

# Calculando la solución analítica en el dominio 
phi_a = phi_x(X, phi0, phiL, Pe, L)
# ==============================================================================
# Vector de mano derecha - el mismo en todos los métodos
b = np.zeros_like(X)
b[0] = phi0
b[-1] = phiL

# ==============================================================================
# Solución numérica con diferencias centradas

# Cálculo de los coeficientes (constantes en este caso) - Funciona en todas las 
# versiones que se estén proponiendo
D = Tau / dx
F = rho * u
Pe_C = F * dx / D
print(f'Peclet en cada celda: {Pe:.2f}')

# Definiendo y llenando la matriz de coeficientes
A = np.zeros((len(X), len(X)))

# Condiciones de contorno del problema (Dirichlet)
A[0, 0] = 1
A[-1, -1] = 1

for i in range(1, len(X) - 1):

    A[i, i - 1] = D * (1 + 0.5 * np.abs(Pe_C)) 
    A[i, i + 1] = D * (1 - 0.5 * np.abs(Pe_C)) 
    A[i, i] = -(A[i, i - 1] + A[i, i + 1])

# Resuelve el sitema de ecuaciones y da los valores de phi en diferencias 
# centradas
phi_CD = np.linalg.solve(A, b)

# ==============================================================================
# Esquema upwind
B = np.zeros_like(A)

B[0, 0] = 1
B[-1, -1] = 1

for i in range(1, len(X) - 1):

    B[i, i - 1] = D + np.max([F, 0])
    B[i, i + 1] = D + np.max([-F, 0])
    B[i, i] = -(B[i, i - 1] + B[i, i + 1])

phi_up = np.linalg.solve(B, b)

# ==============================================================================
# Esquema exponencial
C = np.zeros_like(A)

C[0, 0] = 1
C[-1, -1] = 1

for i in range(1, len(X) - 1):

    # Peclet en caras de nodos
    Pe_w = F / D
    Pe_e = F / D

    # Calculando el Ap
    Aw = D * Afunc(Pe_w) + max(F, 0)
    Ae = D * Afunc(Pe_e) + max(-F, 0)
    Ap = Ae + Aw

    C[i, i - 1] = -Aw
    C[i, i + 1] = -Ae
    C[i, i] = Ap

phi_exp = np.linalg.solve(C, b)

# ==============================================================================
# Esquema híbrido
D1 = np.zeros_like(A)

D1[0, 0] = 1
D1[-1, -1] = 1

for i in range(1, len(X) - 1):

    Ae = D * hibridoPe(Pe_C) + max(-F, 0)
    Aw = D * hibridoPe(Pe_C) + max(F, 0)
    Ap = Ae + Aw

    D1[i, i - 1] = -Aw
    D1[i, i + 1] = -Ae
    D1[i, i] = Ap

phi_hib = np.linalg.solve(D1, b)

# ==============================================================================
# Esquema en serie de potencias
E = np.zeros_like(A)

E[0, 0] = 1
E[-1, -1] = 1

for i in range(1, len(X) - 1):

    Ae = D * PotPe(Pe_C) + max(-F, 0)
    Aw = D * PotPe(Pe_C) + max(F, 0)
    Ap = Ae + Aw

    E[i, i - 1] = -Aw
    E[i, i + 1] = -Ae
    E[i, i] = Ap

phi_sp =np.linalg.solve(E, b)

# Graficando para ver cómo se ven las soluciones
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(X, phi_CD, label ='Diferencias centradas')
ax[0].plot(X, phi_up, label='Esq. upwind')
ax[0].plot(X, phi_exp, label='Esq. exponencial')
ax[0].plot(X, phi_hib, label='Esq. híbrido')
ax[0].plot(X, phi_sp, label='Esq. series potencias')
ax[0].plot(Xa, phiA, label='Solución exacta', ls=':')
ax[0].grid()
ax[0].set_xlim((-0.25, L + 0.25))
ax[0].set_ylabel(r'$\phi(x)$')
ax[0].legend()

ax[1].semilogy(X, error_rel(phi_a, phi_CD), label='Diferencias centradas')
ax[1].semilogy(X, error_rel(phi_a, phi_up), label='Esq. upwind')
ax[1].semilogy(X, error_rel(phi_a, phi_exp), label='Esq. exponencial')
ax[1].semilogy(X, error_rel(phi_a, phi_hib), label='Esq. híbrido')
ax[1].semilogy(X, error_rel(phi_a, phi_sp), label='Esq. series potencias')

ax[1].grid()
ax[1].legend(loc='lower right')
ax[1].set_ylim((1e-16, 1e0))
ax[1].set_xlabel(r'$x(m)$')
ax[1].set_ylabel('Error relativo')

plt.suptitle('Advección difusión estacionaria')
plt.tight_layout()
plt.draw()
plt.pause(0.1)
input('Presione ENTER para continuar...')
plt.close()