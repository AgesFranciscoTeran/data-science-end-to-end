import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

viajes = viajes[(viajes['tarifa'] != 999.99) & (viajes['duracion_min'] >= 0)]
print(f'Filas restantes: {len(viajes)}')

limpios = viajes

plt.scatter(limpios['distancia_km'], limpios['tarifa'])
plt.xlabel('Distancia (km)')
plt.ylabel('Tarifa')
plt.title('Relación entre Distancia y Tarifa')
plt.savefig('figuras/distancia_vs_tarifa.png')
plt.show()

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
print(limpios.groupby('metodo_pago')['propina'].agg(
    promedio='mean', sin_propina=lambda s: (s == 0).sum()))


# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
print(limpios[['distancia_km', 'duracion_min', 'pasajeros', 'propina']].corrwith(limpios['tarifa']))

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print(f'Viajes con pasajeros vacíos: {limpios["pasajeros"].isnull().sum()}')
print(f'Viajes con pasajeros 0: {limpios["pasajeros"].eq(0).sum()}')