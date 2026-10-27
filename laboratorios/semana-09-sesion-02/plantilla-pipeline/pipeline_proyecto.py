from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# El mismo archivo de la Semana 8 (una fila por SSD C1), con las 2 features
# de la Semana 9, Sesion 1 agregadas abajo. La ruta sale de este archivo para
# que modelos_proyecto.py de la Semana 10 pueda importar este modulo.
AQUI = Path(__file__).resolve().parent
ARCHIVO = AQUI / "../../semana-08-sesion-02/data-contenedor/proyecto/ssd_mc1.csv"

OBJETIVO = "failure"
# Contadores SMART crudos que existen el dia de la lectura, mas las 2
# features. Fuera: fecha_smart y r_9/n_9 (delatan la clase por como se armo
# el dataset), disk_id y model (identificador y constante), las n_ que
# repiten a su r_ y las columnas 100 % vacias (r_241, r_242, n_175).
NUMERICAS = ["n_contadores_error", "log_r_5", "r_183", "r_184", "r_187", "r_195",
             "r_197", "r_199", "r_program", "r_erase", "n_wearout", "r_12", "r_174"]
CATEGORICAS = ["app"]
YA_LISTAS = []


def cargar_datos():
    datos = pd.read_csv(ARCHIVO)
    smart = [c for c in datos.columns if c.startswith(("r_", "n_"))]
    datos = datos[~datos[smart].isna().all(axis=1)].copy()
    ERRORES = ["r_5", "r_187", "r_197", "r_199"]
    datos["n_contadores_error"] = (datos[ERRORES] > 0).sum(axis=1)
    datos["log_r_5"] = np.log1p(datos["r_5"])
    return datos


def dividir(datos):
    """Al azar y estratificado por failure. No se puede dividir por fecha:
    todos los sanos se midieron el 2019-12-31, asi que una fecha de corte
    dejaria solo fallas en entrenamiento."""
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
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


if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = dividir(cargar_datos())
    print(f"Tasa de falla: entrenamiento {y_ent.mean():.4f}, prueba {y_prueba.mean():.4f}")

    preparar = preparacion()
    X_ent_listo = preparar.fit_transform(X_ent)
    X_prueba_listo = preparar.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparar.get_feature_names_out()))
