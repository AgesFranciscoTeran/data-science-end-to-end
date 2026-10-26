import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

viajes = pd.read_csv('viajes_taxi.csv')

# Quita los errores confirmados en la Semana 8: tarifa 999.99 y duraciones negativas.
viajes_clean = viajes.query('tarifa < 999 and duracion_min >= 0').copy()

print(viajes_clean.head())
viajes_clean.info()

fecha_inicio = pd.to_datetime(viajes_clean['fecha_hora_inicio'])

viajes_clean['hora'] = fecha_inicio.dt.hour
viajes_clean['dia_semana'] = fecha_inicio.dt.dayofweek
viajes_clean['mes'] = fecha_inicio.dt.month
viajes_clean['es_fin_de_semana'] = (viajes_clean['dia_semana'] >= 5).astype(int)

# Hora pico: de lunes a viernes, de 7 a 9 y de 16 a 19.
en_pico = viajes_clean['hora'].between(7, 9) | viajes_clean['hora'].between(16, 19)
viajes_clean['es_hora_pico'] = (en_pico & (viajes_clean['es_fin_de_semana'] == 0)).astype(int)

# La hora como ciclo: 23 y 0 quedan juntas en el circulo.
viajes_clean['hora_sin'] = np.sin(2 * np.pi * viajes_clean['hora'] / 24)
viajes_clean['hora_cos'] = np.cos(2 * np.pi * viajes_clean['hora'] / 24)

print(viajes_clean[['hora', 'hora_sin', 'hora_cos']].head())

def distancia_en_circulo(h1, h2):
    sin1, cos1 = np.sin(2 * np.pi * h1 / 24), np.cos(2 * np.pi * h1 / 24)
    sin2, cos2 = np.sin(2 * np.pi * h2 / 24), np.cos(2 * np.pi * h2 / 24)
    return np.hypot(sin1 - sin2, cos1 - cos2)

print(f"Hora 23 vs 0: como numero {abs(23 - 0)}, en el ciclo {distancia_en_circulo(23, 0):.3f}")
print(f"Hora 12 vs 0: como numero {abs(12 - 0)}, en el ciclo {distancia_en_circulo(12, 0):.3f}")

# Aporta la hora pico a la tarifa? La tarifa depende sobre todo de la
# distancia, asi que se compara la tarifa por km, sin viajes de distancia 0
# ni los de tarifa fija al aeropuerto.
viajes_clean['tarifa_por_km'] = viajes_clean['tarifa'] / viajes_clean['distancia_km']
sin_aeropuerto = viajes_clean[(viajes_clean['distancia_km'] > 0)
                              & (viajes_clean['zona_origen'] != 'JFK Airport')
                              & (viajes_clean['zona_destino'] != 'JFK Airport')]
por_pico = sin_aeropuerto.groupby('es_hora_pico')[['tarifa', 'distancia_km', 'tarifa_por_km', 'duracion_min']].mean()
por_pico['viajes'] = sin_aeropuerto['es_hora_pico'].value_counts()
print(por_pico.round(2).to_string())
print(sin_aeropuerto[['es_hora_pico', 'tarifa_por_km', 'duracion_min']].corr().round(3))
print(sin_aeropuerto.groupby('hora')['tarifa_por_km'].median().round(2))

viajes_codificado = pd.get_dummies(viajes_clean, columns=['zona_origen', 'zona_destino'])

print(viajes_codificado.columns.tolist())

encoder = OrdinalEncoder()

viajes_codificado['metodo_pago_ordinal'] = encoder.fit_transform(viajes_codificado[['metodo_pago']]).ravel()

print(viajes_codificado.head())
