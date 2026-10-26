import numpy as np
import pandas as pd

# El mismo archivo de la exploracion de la Semana 8 (lo arma
# semana-08-sesion-02/plantilla-eda/preparar_datos.py): una fila por SSD C1.
datos = pd.read_csv("../../semana-08-sesion-02/data-contenedor/proyecto/ssd_mc1.csv",
                    parse_dates=["fecha_smart"])

# 1 si el SSD fallo, 0 si nunca fallo.
OBJETIVO = "failure"

# Filas de fallas sin ninguna lectura SMART: no hay nada de que derivar.
smart = [c for c in datos.columns if c.startswith(("r_", "n_"))]
sin_smart = datos[smart].isna().all(axis=1)
print(f"Filas sin lecturas SMART: {sin_smart.sum()} (failure={datos.loc[sin_smart, OBJETIVO].unique()})")
datos = datos[~sin_smart].copy()

# Feature 1 (Hallazgo 2 de la Semana 8): cuantos contadores de error dejaron
# de ser 0. Lo que separa a las clases no es el valor del contador sino que
# empiece a contar.
ERRORES = ["r_5", "r_187", "r_197", "r_199"]
datos["n_contadores_error"] = (datos[ERRORES] > 0).sum(axis=1)

# Feature 2 (Paso 4 de la Semana 8): r_5 en escala logaritmica. r_5 es casi
# siempre 0 y llega a 9071; con log1p, pasar de 0 a 10 sectores pesa mas que
# pasar de 1000 a 1010.
datos["log_r_5"] = np.log1p(datos["r_5"])

# Evidencia.
print(datos.groupby("n_contadores_error")[OBJETIVO].agg(["mean", "count"]).round(3))
print(datos[["n_contadores_error", OBJETIVO]].corr().round(3))
print(datos[["r_5", "log_r_5", OBJETIVO]].corr()[OBJETIVO].round(3))
print(datos.groupby(OBJETIVO)[["r_5", "log_r_5"]].median())

# app es categorica: one-hot para un modelo lineal, sin orden falso entre
# aplicaciones.
app_one_hot = pd.get_dummies(datos["app"], prefix="app", dtype=int)
print(f"Columnas one-hot de app: {app_one_hot.shape[1]}")

print(datos[["disk_id", "app", "n_contadores_error", "log_r_5", OBJETIVO]].head())
