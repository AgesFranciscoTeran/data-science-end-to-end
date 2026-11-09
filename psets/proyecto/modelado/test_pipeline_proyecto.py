"""Pruebas de pipeline_proyecto.py. Desde la raiz del repositorio:

    pytest psets/proyecto -v

No leen data/: cada prueba arma sus propios SSD de pocos dias, escritos a
mano, asi corren igual en GitHub Actions, donde data/ no existe.
"""

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import (
    OBJETIVO,
    agregar_ventanas,
    columnas_modelo,
    columnas_smart,
    construir_modelo,
    construir_obt,
    etiquetar,
    filas_entrenables,
)


def dias(inicio, cantidad):
    return pd.date_range(inicio, periods=cantidad, freq="D")


def un_ssd(disk_id, fechas, r_5, modelo="MC1"):
    return pd.DataFrame({"DISK_ID": disk_id, "MODEL_CODE": modelo, "OBSERVATION_DATE": fechas, "R_5": r_5})


def sin_fallas():
    return pd.DataFrame({"DISK_ID": pd.Series(dtype="int64"), "MODEL_CODE": pd.Series(dtype=str),
                         "FAILURE_DATE": pd.Series(dtype="datetime64[ns]")})


def flota_de_prueba():
    """Dos SSD con todas las columnas SMART, 40 dias cada uno. El 1 falla el
    dia 35; el 2 no falla."""
    filas = []
    for disk_id in (1, 2):
        ssd = pd.DataFrame({"DISK_ID": disk_id, "MODEL_CODE": "MC1", "OBSERVATION_DATE": dias("2019-01-01", 40)})
        for columna in columnas_smart():
            ssd[columna] = np.arange(40, dtype=float) * disk_id
        filas.append(ssd)
    fallas = pd.DataFrame({"DISK_ID": [1], "MODEL_CODE": ["MC1"], "FAILURE_DATE": pd.to_datetime(["2019-02-04"])})
    return pd.concat(filas, ignore_index=True), fallas


def test_la_etiqueta_no_llega_al_modelo():
    observaciones, fallas = flota_de_prueba()
    obt = construir_obt(observaciones, fallas)
    columnas = columnas_modelo(obt)

    assert OBJETIVO not in columnas
    for prohibida in ["LABEL_STATUS", "DAYS_TO_FAILURE", "FIRST_FAILURE_DATE", "LAST_SEEN_DATE", "DISK_ID"]:
        assert prohibida not in columnas
    assert "R_5_MEAN_7D" in columnas


def test_etiqueta_calculada_a_mano():
    # El SSD 1 falla el 2019-01-11. El SSD 2 no falla y se observa 31 dias.
    observaciones = pd.concat([
        un_ssd(1, pd.to_datetime(["2019-01-01", "2019-01-10", "2019-01-11", "2019-01-12"]), 0.0),
        un_ssd(2, dias("2019-01-01", 31), 0.0),
    ], ignore_index=True)
    fallas = pd.DataFrame({"DISK_ID": [1, 1], "MODEL_CODE": ["MC1", "MC1"],
                           "FAILURE_DATE": pd.to_datetime(["2019-01-11 14:30", "2019-03-01 08:00"])})

    obt = etiquetar(observaciones, fallas)
    ssd1 = obt[obt["DISK_ID"] == 1]
    ssd2 = obt[obt["DISK_ID"] == 2]

    # Cuenta la primera falla, aunque traiga hora.
    assert list(ssd1["DAYS_TO_FAILURE"]) == [10, 1, 0, -1]
    assert list(ssd1["LABEL_STATUS"]) == ["POSITIVE", "POSITIVE", "SAME_DAY_FAILURE", "POST_FAILURE"]
    # 2019-01-01 tiene 30 dias de futuro observado: NEGATIVE. El dia siguiente ya no: CENSORED.
    assert list(ssd2["LABEL_STATUS"].iloc[:2]) == ["NEGATIVE", "CENSORED"]
    assert ssd2[OBJETIVO].iloc[0] == 0
    assert ssd2[OBJETIVO].iloc[1:].isna().all()


def test_ventana_calculada_a_mano():
    # 8 dias seguidos; el contador de sectores reasignados salta a 7 el ultimo dia.
    observaciones = un_ssd(1, dias("2019-01-01", 8), [0, 0, 0, 0, 0, 0, 0, 7.0])
    ultimo = agregar_ventanas(observaciones, ["R_5"]).iloc[-1]

    assert ultimo["R_5_COUNT_7D"] == 7       # del 2 al 8 de enero
    assert ultimo["R_5_MEAN_7D"] == 2.0      # 7 / 7
    assert ultimo["R_5_MEAN_SHIFT_7D"] == 1.0  # la ventana anterior solo tiene el 1 de enero, en 0


def test_las_ventanas_cuentan_dias_de_calendario():
    # Faltan 10 dias de telemetria: la ventana de 7 dias del 15 de enero ya no ve el 1 de enero.
    observaciones = un_ssd(1, pd.to_datetime(["2019-01-01", "2019-01-15"]), [100.0, 0.0])
    ultimo = agregar_ventanas(observaciones, ["R_5"]).iloc[-1]

    assert ultimo["R_5_COUNT_7D"] == 1
    assert ultimo["R_5_MEAN_7D"] == 0.0
    assert ultimo["R_5_COUNT_30D"] == 2


def test_las_ventanas_no_miran_el_futuro():
    fechas = dias("2019-01-01", 10)
    antes = agregar_ventanas(un_ssd(1, fechas, np.arange(10.0)), ["R_5"])
    # Cambia solo el ultimo dia: las features de los dias anteriores no pueden moverse.
    futuro_distinto = np.arange(10.0)
    futuro_distinto[-1] = 1e6
    despues = agregar_ventanas(un_ssd(1, fechas, futuro_distinto), ["R_5"])

    pd.testing.assert_frame_equal(antes.iloc[:-1], despues.iloc[:-1])
    assert despues["R_5_MEAN_7D"].iloc[-1] != antes["R_5_MEAN_7D"].iloc[-1]


def test_las_ventanas_no_mezclan_ssd():
    fechas = dias("2019-01-01", 5)
    solo = agregar_ventanas(un_ssd(1, fechas, 1.0), ["R_5"])
    # El mismo DISK_ID en otro modelo es otro SSD fisico.
    juntos = agregar_ventanas(pd.concat([un_ssd(1, fechas, 1.0), un_ssd(1, fechas, 500.0, modelo="MC2")]), ["R_5"])
    mc1 = juntos[juntos["MODEL_CODE"] == "MC1"]

    assert list(mc1["R_5_MEAN_30D"]) == list(solo["R_5_MEAN_30D"]) == [1.0] * 5


def test_el_modelo_entrena_y_puntua_con_atributos_vacios():
    observaciones, fallas = flota_de_prueba()
    observaciones["R_211"] = np.nan  # un atributo que el lote no reporto
    obt = construir_obt(observaciones, fallas)
    entrenables = filas_entrenables(obt)
    modelo = construir_modelo().fit(entrenables[columnas_modelo(obt)], entrenables[OBJETIVO])

    probabilidades = modelo.predict_proba(obt[columnas_modelo(obt)])[:, 1]
    assert len(probabilidades) == len(obt)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
