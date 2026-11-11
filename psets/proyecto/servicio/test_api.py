"""Pruebas del servicio. Desde la raiz del repositorio:

    pytest psets/proyecto -v

TestClient manda peticiones sin levantar un servidor. Las pruebas reemplazan
modelo.joblib, que no existe en GitHub Actions, por un modelo pequeno que
responde 0.25 para cualquier SSD: revisan el contrato de la API, no la
calidad del modelo.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import COLUMNAS, agregar_features, preparacion
from servicio import api

LECTURA = {
    "disk_id": 48213,
    "app": "WSM",
    "r_5": 12.0,
    "r_183": 0.0,
    "r_184": 0.0,
    "r_187": 3.0,
    "r_195": 0.0,
    "r_197": 0.0,
    "r_199": 1.0,
    "r_program": 0.0,
    "r_erase": 0.0,
    "n_wearout": 98.0,
    "r_12": 41.0,
    "r_174": 30.0,
}


def modelo_de_prueba():
    """La preparacion real con un modelo que responde la proporcion de fallas
    con que se entreno: una de cuatro, 0.25 para cualquier SSD."""
    lecturas = agregar_features(pd.DataFrame([LECTURA] * 4))
    modelo = Pipeline([("preparar", preparacion()), ("modelo", DummyClassifier(strategy="prior"))])
    modelo.fit(lecturas[COLUMNAS], [1, 0, 0, 0])
    return modelo


api.modelo = modelo_de_prueba()
api.info_modelo = {"run_id": "prueba"}
cliente = TestClient(api.app)


def test_salud():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "run_id": "prueba"}


def test_prediccion_valida():
    respuesta = cliente.post("/predecir", json=LECTURA)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"disk_id": 48213, "probabilidad_falla": 0.25, "revisar": True, "run_id": "prueba"}


def test_lectura_sin_disk_id_se_rechaza():
    lectura = LECTURA.copy()
    lectura.pop("disk_id")
    respuesta = cliente.post("/predecir", json=lectura)
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", "disk_id"] for error in respuesta.json()["detail"])


def test_lectura_sin_ningun_contador_se_rechaza():
    respuesta = cliente.post("/predecir", json={"disk_id": 1, "app": "WSM"})
    assert respuesta.status_code == 422


def test_contador_negativo_se_rechaza():
    respuesta = cliente.post("/predecir", json={**LECTURA, "r_5": -1})
    assert respuesta.status_code == 422


def test_acepta_contadores_vacios_y_una_app_nueva():
    lectura = {"disk_id": 7, "app": "una_app_nueva", "r_5": 0.0}
    respuesta = cliente.post("/predecir", json=lectura)
    assert respuesta.status_code == 200
    assert 0 <= respuesta.json()["probabilidad_falla"] <= 1


def test_el_servicio_calcula_las_features_igual_que_el_entrenamiento():
    # Lo que el servicio le pasa al modelo tiene que ser exactamente lo que
    # agregar_features produce para la misma fila en el entrenamiento.
    preparado = api.preparar(api.LecturaSmart(**LECTURA))
    esperado = agregar_features(pd.DataFrame([LECTURA]))[COLUMNAS]

    pd.testing.assert_frame_equal(preparado, esperado, check_dtype=False)
    assert preparado["n_contadores_error"].iloc[0] == 3  # r_5, r_187 y r_199 por encima de 0
    assert np.isclose(preparado["log_r_5"].iloc[0], np.log1p(12))
