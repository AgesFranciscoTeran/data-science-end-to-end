"""Pipeline del proyecto de fallas de SSD (modelo C1, el MC1 del PSet 2).

Es el pipeline de la Semana 9, Sesion 2 (laboratorios/semana-09-sesion-02/
plantilla-pipeline/), movido al proyecto. La lectura del archivo y la
preparacion de las filas estan separadas para que las pruebas usen datos
propios sin leer data/.

    from pipeline_proyecto import cargar_datos, dividir, preparacion, construir_modelo

Una fila = un SSD C1. failure = 1 si el SSD fallo (SMART del momento de la
falla), 0 si nunca fallo (SMART del 2019-12-31). Los datos salen de
../ingesta/preparar_datos.py y quedan en ../data/processed/ssd_mc1.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AQUI = Path(__file__).resolve().parent
ARCHIVO = AQUI / "../data/processed/ssd_mc1.csv"

OBJETIVO = "failure"
# Contadores SMART crudos que existen el dia de la lectura, mas las 2
# features de la Semana 9, Sesion 1. Fuera: fecha_smart y r_9/n_9 (delatan
# la clase por como se armo el dataset: todos los sanos se midieron el mismo
# dia), disk_id y model (identificador y constante), las n_ que repiten a su
# r_ y las columnas 100 % vacias (r_241, r_242, n_175).
NUMERICAS = ["n_contadores_error", "log_r_5", "r_183", "r_184", "r_187", "r_195",
             "r_197", "r_199", "r_program", "r_erase", "n_wearout", "r_12", "r_174"]
CATEGORICAS = ["app"]
YA_LISTAS = []
COLUMNAS = NUMERICAS + CATEGORICAS + YA_LISTAS

# Contadores de error: cualquier valor mayor que 0 es una senal de desgaste.
ERRORES = ["r_5", "r_187", "r_197", "r_199"]


def agregar_features(datos):
    """Quita los SSD sin ninguna lectura SMART y agrega las dos features:
    cuantos contadores de error estan por encima de 0, y el logaritmo de los
    sectores reasignados (r_5), que va de 0 a decenas de miles."""
    smart = [c for c in datos.columns if c.startswith(("r_", "n_"))]
    datos = datos[~datos[smart].isna().all(axis=1)].copy()
    datos["n_contadores_error"] = (datos[ERRORES] > 0).sum(axis=1)
    datos["log_r_5"] = np.log1p(datos["r_5"])
    return datos


def cargar_datos(archivo=ARCHIVO):
    return agregar_features(pd.read_csv(archivo))


def dividir(datos):
    """Al azar y estratificado por failure. No se puede dividir por fecha:
    todos los sanos se midieron el 2019-12-31, asi que una fecha de corte
    dejaria solo fallas en entrenamiento."""
    X = datos[COLUMNAS]
    y = datos[OBJETIVO]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


def construir_modelo():
    """El modelo final: el bosque de la Semana 10 con hojas de al menos 2
    discos (average precision de prueba 0.50, la mejor de la tabla)."""
    return Pipeline([
        ("preparar", preparacion()),
        ("modelo", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)),
    ])


if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = dividir(cargar_datos())
    print(f"Tasa de falla: entrenamiento {y_ent.mean():.4f}, prueba {y_prueba.mean():.4f}")

    preparar = preparacion()
    X_ent_listo = preparar.fit_transform(X_ent)
    X_prueba_listo = preparar.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparar.get_feature_names_out()))
