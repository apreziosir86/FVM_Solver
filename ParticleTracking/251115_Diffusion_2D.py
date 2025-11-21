import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definiendo condiciones inciales del problema
n = 10                            # Número de partículas
D = 0.3                           # Difusividad del medio
t0 = 0                            # Tiempo inicial (s)
tf = 5                            # Tiempo final (s)
dt = 0.1                          # Paso de tiempo (s)
paso_guardar = 5                  # Cada cuántos pasos guarda
x_inicial = 10                    # Coord x inicial (m)
y_inicial = 10                    # Coordenada y inicial (m)

# Armar los vectores que almacenan las coordenadas
n = int(n)
x0 = np.ones(n) * x_inicial
y0 = np.ones(n) * y_inicial

# Vector de tiempos
T = np.arange(t0, tf + dt, dt)

# Cálculo de K
K = 2 * np.sqrt(D * dt)

# Dataframes
Dx = pd.DataFrame({'t=0s': x0})
Dy = pd.DataFrame({'t=0s': y0})
 
# Función de avance en el tiempo
def PasoTiempo(x, K): return x + np.random.randn(len(x)) * K

# Función de graficado
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    limi = x_inicial - 10
    lims = x_inicial + 10
    bins = np.arange(limi, lims + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
scatter_hist(x0, y0, axs['scatter'], axs['histx'], axs['histy'])

plt.pause(0.5)
input('Presione ENTER')


# Bucle temporal
for t in range(1, len(T)):

    # Avanzando los vectores en el tiempo
    x1 = PasoTiempo(x0, K)

    y1 = PasoTiempo(y0, K)

    # Guardando datos y graficando
    if t % paso_guardar == 0:

        # Guardando en dataframe
        Dx['t='+str(T[t])+'s'] = x1
        Dy['t='+str(T[t])+'s'] = y1

        # Poniendo los datos de la gráfica
        scatter_hist(x1, y1, axs['scatter'], axs['histx'], axs['histy'])

    # Avanzando en tiempo
    x0 = x1
    y0 = y1
