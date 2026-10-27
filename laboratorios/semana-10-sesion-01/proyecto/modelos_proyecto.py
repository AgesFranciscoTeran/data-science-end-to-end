"""Primeros modelos del proyecto de SSD: linea base, regresion logistica y
bosque aleatorio, con el pipeline y la division de la Semana 9, Sesion 2.

    cd laboratorios/semana-10-sesion-01/proyecto
    python3 modelos_proyecto.py

Metrica: average precision (area bajo la curva precision-recall). Solo 5.94 %
de los discos falla, asi que la exactitud de "nunca falla" ya es 94 %; la
average precision mide que tan arriba quedan las fallas al ordenar los discos
por riesgo, y su piso es la proporcion de fallas.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent / "../../semana-09-sesion-02/plantilla-pipeline"))
from pipeline_proyecto import cargar_datos, dividir, preparacion  # noqa: E402

X_ent, X_prueba, y_ent, y_prueba = dividir(cargar_datos())
print(f"Entrenamiento: {len(X_ent)} discos, prueba: {len(X_prueba)}, tasa de falla {y_ent.mean():.4f}")
print()

resultados = []


def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    prob_ent = pipeline.predict_proba(X_ent)[:, 1]
    prob_prueba = pipeline.predict_proba(X_prueba)[:, 1]
    ap_ent = average_precision_score(y_ent, prob_ent)
    ap_prueba = average_precision_score(y_prueba, prob_prueba)
    resultados.append({
        "modelo": nombre,
        "ap_entrenamiento": round(ap_ent, 4),
        "ap_prueba": round(ap_prueba, 4),
        "brecha": round(ap_ent - ap_prueba, 4),
        "auc_prueba": round(roc_auc_score(y_prueba, prob_prueba), 4),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:34s} entrenamiento {ap_ent:.4f}   prueba {ap_prueba:.4f}")
    return pipeline


# Linea base: siempre la proporcion de fallas de entrenamiento. No ordena a
# los discos, asi que su average precision es la tasa de falla.
evaluar("linea base (proporcion)", DummyClassifier(strategy="prior"))

# Lineal: C es el inverso de la regularizacion L2; C=1 es el valor por
# defecto y C=0.01 regulariza 100 veces mas.
evaluar("logistica C=1", LogisticRegression(C=1.0, max_iter=2000))
evaluar("logistica C=0.01", LogisticRegression(C=0.01, max_iter=2000))

# Ensamblaje: sin limite, cada arbol llega a hojas de un disco y memoriza
# el entrenamiento; con min_samples_leaf, cada hoja promedia al menos esa
# cantidad de discos.
evaluar("bosque sin limite", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
evaluar("bosque min_samples_leaf=2",
        RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42))
evaluar("bosque min_samples_leaf=50",
        RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
