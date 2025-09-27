import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definiendo las variables del problema
phi0 = 10
phiL = 2
L = 5
u = -25
rho = 5
Tau = 20

# Calculando e imprimiendo el número de Peclet:
Pe = rho * u * L / Tau
print(f'Valor del número de Péclet general: {Pe:.4f}')

# Función para calcularla solución analítica del problema
def phi_x(x, phi0, phiL, Pe, L):

    return phi0 + (phiL - phi0) * ((np.exp(Pe * x / L) - 1) / (np.exp(Pe) - 1))

# Función para calcular error en cada método
def error_rel(x1, x2): 

    error = np.zeros_like(x1)

    error[1:-1] = np.abs((x2[1:-1] - x1[1:-1]) / x1[1:-1])

    return error


# Calculando a solución analítica con buena definición
Xa = np.linspace(0, L, 500)
phiA = phi_x(Xa, phi0, phiL, Pe, L)

# Definiendo los valor numéricos que quiero tener en la solución
dx = 0.25                            # Tamaño de celda de volumen (m)

X = np.arange(0, L + dx, dx)

# Calculando la solución analítica en el dominio 
phi_a = phi_x(X, phi0, phiL, Pe, L)

# Vector de mano derecha - el mismo en todos los métodos
b = np.zeros_like(X)
b[0] = phi0
b[-1] = phiL

# ==============================================================================
# Solución numérica con diferencias centradas

# Cálculo de los coeficientes (constantes en este caso)
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

    A[i, i - 1] = D * (1 - 0.5 * np.abs(Pe_C)) + np.max([F, 0]) 
    A[i, i + 1] = D * (1 - 0.5 * np.abs(Pe_C)) + np.max([-F, 0])
    A[i, i] = -(A[i, i - 1] + A[i, i + 1])



phi_CD = np.linalg.solve(A, b)

# Esquema upwind

# ==============================================================================
# Esquema exponencial
for i in range(1, len(X) - 1):

    A[i, i - 1] = D * np.abs(Pe_C) / (np.exp(np.abs(Pe_C)) - 1) + np.max([F, 0])
    A[i, i + 1] = D * np.abs(Pe_C) / (np.exp(np.abs(Pe_C)) - 1) +\
            np.max([-F, 0])
    A[i, i] = -(A[i, i - 1] + A[i, i + 1])

phi_exp = np.linalg.solve(A, b)

# Graficando para ver cómo se ven las soluciones
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(X, phi_CD, label ='Diferencias centradas')
ax[0].plot(X, phi_exp, label='Esq. exponencial')
ax[0].plot(Xa, phiA, label='Solución exacta', ls=':')
ax[0].grid()
ax[0].set_xlim((-0.25, L + 0.25))
ax[0].set_ylabel(r'$\phi(x)$')
ax[0].legend()

ax[1].semilogy(X, error_rel(phi_a, phi_CD), label='Diferencias centradas')
ax[1].semilogy(X, error_rel(phi_a, phi_exp), label='Esq. exponencial')

ax[1].grid()
ax[1].legend()
ax[1].set_ylim((1e-8, 1e0))
ax[1].set_xlabel(r'$x(m)$')
ax[1].set_ylabel('Error relativo')

plt.suptitle('Advección difusión estacionaria')
plt.tight_layout()
plt.draw()
plt.pause(0.1)
input('Presione ENTER para continuar...')
plt.close()