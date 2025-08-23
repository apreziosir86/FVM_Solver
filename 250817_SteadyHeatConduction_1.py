# ==============================================================================
# Steady Heat Conduction Solver - Control volume method
# Chapters 3-4
# Steady heat conduction with constant source/sink term in the domain.
# Dirichlet boundary conditions in both sides of the domain
# Antonio Preziosi-Ribero
# ==============================================================================

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt

# Setting up the variables for the problem
nx = 5                              # Number of points in the grid (volume centers)
L = 1.0                             # Length of the domain (m)
k = 1.0                             # Thermal conductivity (W/mK)
S = 2.5                             # Source term (W/m^3)
T0 = 300.0                          # Boundary condition at x=0 (K)
TL = 250.0                          # Boundary condition at x=L (K)

# Creating vector with coordinates
x = np.linspace(0, L, nx)

# Creating matrix to store constant coefficients and RHS vector
A = np.zeros((nx, nx))
b = np.zeros(nx)

# Filling information for inner nodes in the matrix: This is not the best way, 
# but it is comprehensible and gives some idea of the process
for i in range(1, nx - 1): 
    
    # Filling information for matrix coefficients
    A[i, i - 1] = k / (x[i] - x[i - 1])    # Left neighbor
    A[i, i + 1] = k / (x[i + 1] - x[i])    # Right neighbor
    A[i, i] = -A[i, i - 1] - A[i, i + 1]   # Center node

    # Filling information for RHS
    b[i] = -S * (x[i + 1] - x[i - 1]) / 2

# Filling information for boundary conditions in matrix and RHS
A[0, 0] = 1
b[0] = T0
A[nx - 1, nx - 1] = 1
b[nx - 1] = TL

# Solving linear system:
T = np.linalg.solve(A, b)

# Generating exact solution (from solving differential equation)
X = np.linspace(0, L, 200)
T_exact = T0 + (TL - T0) * X / L + S * (L * X - X ** 2) / (2 * k)

# Plotting the result 
fig, ax = plt.subplots()
ax.plot(x, T, marker='o', label='Numerical Solution')
ax.plot(X, T_exact, label='Exact Solution', linestyle='--')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Steady State Temperature Distribution')
plt.grid()
plt.legend()
plt.show()