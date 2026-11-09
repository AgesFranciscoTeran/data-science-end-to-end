"""Pruebas de pipeline_proyecto.py. Desde la raiz del repositorio:

    pytest psets/proyecto -v

No leen data/: cada prueba arma sus propios SSD, escritos a mano, asi
corren igual en GitHub Actions, donde data/ no existe.
"""

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import (
    COLUMNAS,
    OBJETIVO,
    agregar_features,
    construir_modelo,
    dividir,
    preparacion,
)


def ssd_de_prueba(cantidad=20):
    """SSD con las columnas de ssd_mc1.csv. Uno de cada cinco falla; el
    primero no tiene ninguna lectura SMART."""
    rng = np.random.default_rng(0)
    datos = pd.DataFrame({
        "model": "C1",
        "disk_id": range(cantidad),
        "app": (["WSM", "NAS", "RM", "DB", "none"] * cantidad)[:cantidad],
        "fecha_smart": "2019-12-31",
        "failure": [1 if i % 5 == 0 else 0 for i in range(cantidad)],
    })
    for columna in ["r_5", "r_183", "r_184", "r_187", "r_195", "r_197", "r_199", "r_program",
                    "r_erase", "n_wearout", "r_12", "r_174", "r_9", "n_9"]:
        datos[columna] = rng.integers(0, 50, cantidad).astype(float)
    datos.loc[0, [c for c in datos.columns if c.startswith(("r_", "n_"))]] = np.nan
    return datos


def test_el_objetivo_y_las_columnas_que_delatan_no_llegan_al_modelo():
    X_ent, X_prueba, _, _ = dividir(agregar_features(ssd_de_prueba()))

    assert OBJETIVO not in COLUMNAS
    for prohibida in [OBJETIVO, "fecha_smart", "r_9", "n_9", "disk_id"]:
        assert prohibida not in X_ent.columns
        assert prohibida not in X_prueba.columns


def test_features_calculadas_a_mano():
    datos = pd.DataFrame({
        "r_5": [0.0, 3.0, np.nan],
        "r_187": [0.0, 1.0, 0.0],
        "r_197": [0.0, 0.0, 2.0],
        "r_199": [0.0, 5.0, 0.0],
    })
    features = agregar_features(datos)

    # Contadores de error por encima de 0: ninguno, tres (r_5, r_187, r_199), uno (r_197).
    assert list(features["n_contadores_error"]) == [0, 3, 1]
    assert features["log_r_5"].iloc[0] == 0.0
    assert np.isclose(features["log_r_5"].iloc[1], np.log(4))
    assert np.isnan(features["log_r_5"].iloc[2])


def test_quita_los_ssd_sin_lecturas_smart():
    datos = ssd_de_prueba()
    features = agregar_features(datos)

    assert len(features) == len(datos) - 1
    assert 0 not in features["disk_id"].to_list()


def test_la_division_conserva_la_tasa_de_falla():
    datos = agregar_features(ssd_de_prueba(100))
    _, _, y_ent, y_prueba = dividir(datos)

    assert len(y_ent) + len(y_prueba) == len(datos)
    assert abs(y_ent.mean() - y_prueba.mean()) < 0.02


def test_la_preparacion_acepta_una_app_que_no_vio_y_valores_vacios():
    datos = agregar_features(ssd_de_prueba())
    preparar = preparacion().fit(datos[COLUMNAS])

    nuevo = datos[COLUMNAS].head(2).copy()
    nuevo["app"] = ["una_app_nueva", np.nan]
    nuevo["r_183"] = np.nan
    listo = preparar.transform(nuevo)

    assert listo.shape[0] == 2
    assert not np.isnan(listo).any()


def test_el_modelo_final_entrena_y_da_probabilidades():
    datos = agregar_features(ssd_de_prueba(60))
    modelo = construir_modelo().fit(datos[COLUMNAS], datos[OBJETIVO])
    probabilidades = modelo.predict_proba(datos[COLUMNAS])[:, 1]

    assert len(probabilidades) == len(datos)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
