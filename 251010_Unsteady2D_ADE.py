"""Script CORREGIDO - CON ESCALA DE COLORES FIJA Y SIN ERRORES"""
import numpy as np
import matplotlib.pyplot as plt 

# ==============================================================================
# FUNCIONES ÚTILES
# ==============================================================================

def V_parabolica(X, Lx, umax): 
    """Perfil parabólico de velocidad VERTICAL: V(x) = 4*umax*(x*Lx - x²)/Lx²"""
    return 4 * umax * (X * Lx - X ** 2) / (Lx ** 2)

def esquema_upwind_simple(Pe):
    """Esquema Upwind puro - muy estable"""
    return 1.0 + max(0, -Pe)

# ==============================================================================
# PARÁMETROS - MÁS REALISTAS
# ==============================================================================

Lx = 5.    # Ancho del dominio
Ly = 20.   # Alto del dominio  
Tmax = 15. # Tiempo suficiente para ver movimiento
umax = 0.8 # Velocidad razonable
Tau = 0.5  # Difusividad moderada
Cf = 1.0   # Concentración entrada más visible

dx = 1.
dy = 1.  
dt = 0.05  # Paso temporal más pequeño

nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1
N = nx * ny

print(f'Dominio: {Lx} m x {Ly} m')
print(f'Mallado: {nx} x {ny} = {N} nodos')
print(f'Velocidad máxima: {umax} m/s → Tiempo para cruzar: {Ly/umax:.1f} s')

# Crear coordenadas
X = np.linspace(0, Lx, nx)
Y = np.linspace(0, Ly, ny)
XG, YG = np.meshgrid(X, Y, indexing='xy')

# ==============================================================================
# CONSTRUCCIÓN DE MATRIZ - CONDICIÓN DE SALIDA CORREGIDA
# ==============================================================================

print("Construyendo matriz del sistema...")

A = np.zeros((N, N))
b = np.zeros(N)

for idx in range(N):
    j = idx // nx  # fila (y)
    i = idx % nx   # columna (x)

    # FONDO (y=0): ENTRADA - Dirichlet
    if j == 0: 
        A[idx, idx] = 1.0
        b[idx] = Cf
        
    # PARTE SUPERIOR (y=Ly): SALIDA - CONDICIÓN CONVECTIVA CORREGIDA
    elif j == ny - 1: 
        # Condición de salida: ∂φ/∂t + V·∇φ = 0 (aproximación)
        V_salida = V_parabolica(X[i], Lx, umax)
        
        # Esquema upwind en la salida
        A[idx, idx] = 1.0 + (V_salida * dt / dy) + (dx * dy / dt)
        A[idx, idx - nx] = - (V_salida * dt / dy)  # Solo contribución del nodo anterior
        
    # PARED IZQUIERDA (x=0): Neumann 
    elif i == 0: 
        A[idx, idx] = 1.0 + (dx * dy / dt)
        A[idx, idx + 1] = -1.0
        
    # PARED DERECHA (x=Lx): Neumann  
    elif i == nx - 1: 
        A[idx, idx] = 1.0 + (dx * dy / dt)
        A[idx, idx - 1] = -1.0
        
    # NODOS INTERNOS - ESQUEMA UPWIND SIMPLE
    else:
        # Velocidad VERTICAL en este nodo
        V_actual = V_parabolica(X[i], Lx, umax)
        
        # Coeficientes de difusión
        D_superior = Tau * dx / dy
        D_inferior = Tau * dx / dy  
        D_este = Tau * dy / dx
        D_oeste = Tau * dy / dx
        
        # Números de Peclet
        Pe_superior = V_actual * dy / Tau
        Pe_inferior = V_actual * dy / Tau
        
        # ESQUEMA UPWIND EXPLÍCITO para advección
        a_superior = D_superior * esquema_upwind_simple(Pe_superior) + max(-V_actual * dx, 0)
        a_inferior = D_inferior * esquema_upwind_simple(Pe_inferior) + max(V_actual * dx, 0)
        a_este = D_este
        a_oeste = D_oeste
        
        # Coeficiente central CON TÉRMINO TEMPORAL
        ap = a_superior + a_inferior + a_este + a_oeste + (dx * dy / dt)
        
        # Ensamblar matriz
        A[idx, idx] = ap
        A[idx, idx + nx] = -a_superior   # Norte
        A[idx, idx - nx] = -a_inferior   # Sur  
        A[idx, idx + 1] = -a_este        # Este
        A[idx, idx - 1] = -a_oeste       # Oeste

print("Matriz construida correctamente")

plt.figure(figsize=(8, 6))

plt.spy(A, markersize=1)
plt.draw()
plt.pause(0.1)
input('Presione enter')
plt.close()

# ==============================================================================
# SIMULACIÓN TEMPORAL - CON ESCALA DE COLORES FIJA Y SIN ERRORES
# ==============================================================================

phi0 = np.zeros(N)

plt.ion()
# CREAR FIGURA CON TAMAÑO OPTIMIZADO
fig, ax = plt.subplots(figsize=(8, 12))

# Contorno inicial
contour_data_initial = phi0.reshape(ny, nx)

# Crear el primer contour con escala fija
contour = ax.contourf(XG, YG, contour_data_initial, levels=50, cmap='plasma', 
                      vmin=0, vmax=Cf)

# Crear colorbar
cbar = fig.colorbar(contour, ax=ax, shrink=0.8, aspect=20)
cbar.set_label('Concentración (φ)', fontsize=12, rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=10)

# CORRECCIÓN: Usar el método correcto para fijar límites
cbar.mappable.set_clim(0, Cf)  # Esto es lo correcto

# Añadir perfil de velocidad para referencia
ax.plot(X, V_parabolica(X, Lx, umax) * 2 + Ly * 0.1, 'w--', alpha=0.7, label='Perfil velocidad (x10)')
ax.legend()

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_aspect('equal')
ax.set_title('INICIO - Transporte de Contaminante con Perfil Parabólico', fontsize=14, pad=20)

plt.tight_layout()

print("Iniciando simulación temporal...")

for t in np.arange(dt, Tmax + dt, dt):
    # Vector b - CORREGIDO
    for idx in range(N):
        j = idx // nx
        i = idx % nx

        if j == 0:  # Entrada: Dirichlet
            b[idx] = Cf
        elif j == ny - 1:  # Salida: Término convectivo
            V_salida = V_parabolica(X[i], Lx, umax)
            b[idx] = phi0[idx] * (dx * dy / dt)
        elif i == 0 or i == nx - 1:  # Paredes: Neumann
            b[idx] = 0
        else:  # Internos
            b[idx] = phi0[idx] * (dx * dy / dt)

    try:
        phi = np.linalg.solve(A, b)
        
    except np.linalg.LinAlgError as e:
        print(f"Error en t={t:.2f}: {e}")
        break

    phi0 = phi.copy()

    # Graficar cada 5 pasos
    if int(t/dt) % 10 == 0:
        # Limpiar contornos antiguos
        for coll in ax.collections[:]:
            coll.remove()
        
        # Nuevo contorno CON ESCALA FIJA
        contour_data = phi.reshape(ny, nx)
        
        # Siempre usar vmin=0, vmax=Cf
        contour = ax.contourf(XG, YG, contour_data, levels=50, cmap='plasma', 
                              vmin=0, vmax=Cf)
        
        # CORRECCIÓN: Actualizar la mappable de la colorbar existente
        cbar.mappable.set_array(contour_data)  # Actualizar datos
        cbar.mappable.set_clim(0, Cf)  # Mantener límites fijos
        
        # Calcular posición del frente
        umbral = Cf * 0.1
        frente_y = 0
        for j in range(ny):
            if np.max(contour_data[j, :]) > umbral:
                frente_y = Y[j]
                break
        
        ax.set_title(f'Transporte de Contaminante - t = {t:.2f} s\nFrente en y = {frente_y:.1f} m', fontsize=14, pad=20)
        
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print(f"t = {t:.2f}s - Frente en y = {frente_y:.1f} m, φ_min = {np.min(phi):.3f}, φ_max = {np.max(phi):.3f}")

plt.ioff()

# Gráfica final también con escala fija
plt.figure(figsize=(12, 8))
contour_final = plt.contourf(XG, YG, phi0.reshape(ny, nx), levels=50, cmap='plasma', 
                             vmin=0, vmax=Cf)
cbar_final = plt.colorbar(contour_final, shrink=0.8, aspect=20)
cbar_final.set_label('Concentración (φ)', fontsize=12, rotation=270, labelpad=15)
cbar_final.mappable.set_clim(0, Cf)  # Fijar escala en gráfica final
plt.xlabel('x (m)', fontsize=12)
plt.ylabel('y (m)', fontsize=12)
plt.title('Distribución Final de Concentración', fontsize=14, pad=20)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

print("¡Simulación completada!")

# ==============================================================================
# ANÁLISIS FINAL DEL MOVIMIENTO
# ==============================================================================
print("\n=== ANÁLISIS DEL MOVIMIENTO ===")
distancia_recorrida = np.max([Y[j] for j in range(ny) if np.max(phi0.reshape(ny, nx)[j, :]) > Cf * 0.1])
print(f"Distancia recorrida: {distancia_recorrida:.1f} m")
print(f"Velocidad promedio: {distancia_recorrida/Tmax:.2f} m/s")
print(f"Velocidad máxima teórica: {umax:.2f} m/s")