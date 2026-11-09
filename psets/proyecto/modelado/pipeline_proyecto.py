"""Pipeline del modelo de prediccion de fallas de SSD (modelo de disco MC1).

Es la version en pandas de lo que hace el job de Spark que construye la OBT
del PSet 2 (build_mc1_obt.py): la misma etiqueta a 30 dias y las mismas
ventanas de 7, 14 y 30 dias, sobre pocas filas, para poder entrenar, puntuar
y probar sin Snowflake.

    from pipeline_proyecto import construir_obt, columnas_modelo, construir_modelo

Una fila de la OBT = un SSD en un dia de observacion. El SSD se identifica por
el par (DISK_ID, MODEL_CODE): el DISK_ID se repite entre modelos distintos.

Desde esta carpeta, con la muestra que deja ../ingesta/descargar_muestra.py:

    python pipeline_proyecto.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Los 22 atributos SMART que reporta MC1 (los mismos del job de Spark).
SMART_IDS = (1, 5, 9, 12, 170, 171, 172, 173, 174, 180, 183, 184, 187, 188,
             194, 195, 196, 197, 198, 199, 206, 211)
# R = valor raw (contador real), N = valor normalizado por el fabricante.
REPRESENTACIONES = ("R", "N")
VENTANAS = (7, 14, 30)
HORIZONTE_DIAS = 30

LLAVE_SSD = ["DISK_ID", "MODEL_CODE"]
FECHA = "OBSERVATION_DATE"
# Columnas que existen en la OBT pero que el modelo nunca debe recibir: la
# identidad del disco y todo lo que se calcula mirando el futuro.
IDENTIDAD = LLAVE_SSD + [FECHA]
ETIQUETA = ["FIRST_FAILURE_DATE", "LAST_SEEN_DATE", "DAYS_TO_FAILURE", "LABEL_STATUS", "TARGET_30D"]
OBJETIVO = "TARGET_30D"


def columnas_smart():
    return [f"{rep}_{smart_id}" for rep in REPRESENTACIONES for smart_id in SMART_IDS]


def normalizar_smart(crudo):
    """Lleva un archivo SMART de Alibaba (disk_id, model, ds, r_5, n_5, ...) a
    los nombres del proyecto. Los atributos que el archivo no trae quedan como
    NaN, para que todas las tablas tengan las mismas columnas."""
    smart = pd.DataFrame({
        "DISK_ID": crudo["disk_id"].astype("int64"),
        "MODEL_CODE": crudo["model"].astype(str),
        FECHA: pd.to_datetime(crudo["ds"].astype(str), format="%Y%m%d"),
    })
    for columna in columnas_smart():
        origen = columna.lower()
        smart[columna] = crudo[origen].astype(float) if origen in crudo.columns else np.nan
    return smart


def etiquetar(observaciones, fallas):
    """Agrega las cinco columnas de etiqueta. fallas tiene DISK_ID, MODEL_CODE
    y FAILURE_DATE; si un SSD fallo varias veces cuenta la primera falla.

    LABEL_STATUS, en orden de prioridad:
      POST_FAILURE      la observacion es posterior a la falla
      SAME_DAY_FAILURE  la observacion es del mismo dia de la falla
      POSITIVE          la falla ocurre entre 1 y 30 dias despues   -> 1
      NEGATIVE          el SSD sigue observado 30 dias o mas despues -> 0
      CENSORED          no hay suficiente futuro para saberlo
    Solo POSITIVE y NEGATIVE llevan TARGET_30D; el resto queda en NaN.
    """
    obs = observaciones.copy()
    obs[FECHA] = pd.to_datetime(obs[FECHA]).dt.normalize()

    primera_falla = (
        fallas.assign(FAILURE_DATE=pd.to_datetime(fallas["FAILURE_DATE"]).dt.normalize())
        .groupby(LLAVE_SSD, as_index=False)["FAILURE_DATE"].min()
        .rename(columns={"FAILURE_DATE": "FIRST_FAILURE_DATE"})
    )
    ultimo_dia = (
        obs.groupby(LLAVE_SSD, as_index=False)[FECHA].max()
        .rename(columns={FECHA: "LAST_SEEN_DATE"})
    )
    obs = obs.merge(primera_falla, on=LLAVE_SSD, how="left").merge(ultimo_dia, on=LLAVE_SSD, how="left")

    dias = (obs["FIRST_FAILURE_DATE"] - obs[FECHA]).dt.days
    obs["DAYS_TO_FAILURE"] = dias
    hay_falla = obs["FIRST_FAILURE_DATE"].notna()
    condiciones = [
        hay_falla & (dias < 0),
        hay_falla & (dias == 0),
        dias.between(1, HORIZONTE_DIAS),
        obs["LAST_SEEN_DATE"] >= obs[FECHA] + pd.Timedelta(days=HORIZONTE_DIAS),
    ]
    obs["LABEL_STATUS"] = np.select(
        condiciones, ["POST_FAILURE", "SAME_DAY_FAILURE", "POSITIVE", "NEGATIVE"], default="CENSORED"
    )
    obs[OBJETIVO] = obs["LABEL_STATUS"].map({"POSITIVE": 1.0, "NEGATIVE": 0.0})
    return obs


def _rodar(obs, columnas, dias):
    """Suma y conteo (de valores no nulos) por SSD en una ventana de `dias`
    dias de calendario que termina hoy, alineados con las filas de obs."""
    ventana = obs.groupby(LLAVE_SSD, sort=False).rolling(f"{dias}D", on=FECHA)[columnas]
    suma = ventana.sum().fillna(0).to_numpy()
    conteo = ventana.count().to_numpy()
    return suma, conteo


def agregar_ventanas(observaciones, columnas=None):
    """Agrega media, conteo y cambio de media (actual - anterior) en ventanas
    de 7, 14 y 30 dias, por SSD.

    Las ventanas son de dias de calendario, no de filas: si faltan dias de
    telemetria, la de 7D sigue cubriendo solo los 7 dias anteriores. Solo
    miran hacia atras y nunca mezclan dos SSD."""
    columnas = list(columnas or columnas_smart())
    obs = observaciones.copy()
    obs[FECHA] = pd.to_datetime(obs[FECHA])
    # Ordenar por SSD y fecha deja las filas en el mismo orden que devuelve
    # groupby(...).rolling(...), asi los resultados se pegan fila a fila.
    obs = obs.sort_values(LLAVE_SSD + [FECHA]).reset_index(drop=True)
    features = {}
    for dias in VENTANAS:
        suma, conteo = _rodar(obs, columnas, dias)
        suma_doble, conteo_doble = _rodar(obs, columnas, 2 * dias)
        conteo_previo = conteo_doble - conteo
        with np.errstate(invalid="ignore", divide="ignore"):
            media = np.where(conteo > 0, suma / conteo, np.nan)
            media_previa = np.where(conteo_previo > 0, (suma_doble - suma) / conteo_previo, np.nan)
        for i, columna in enumerate(columnas):
            features[f"{columna}_MEAN_{dias}D"] = media[:, i]
            features[f"{columna}_COUNT_{dias}D"] = conteo[:, i]
            features[f"{columna}_MEAN_SHIFT_{dias}D"] = media[:, i] - media_previa[:, i]
    return pd.concat([obs, pd.DataFrame(features, index=obs.index)], axis=1)


def construir_obt(observaciones, fallas):
    """Observaciones SMART + fallas -> OBT con features de ventana y etiqueta."""
    return etiquetar(agregar_ventanas(observaciones), fallas)


def columnas_modelo(obt):
    """Las columnas que recibe el modelo: todo menos la identidad y la etiqueta."""
    excluidas = set(IDENTIDAD + ETIQUETA)
    return [columna for columna in obt.columns if columna not in excluidas]


def filas_entrenables(obt):
    """Solo las filas con etiqueta conocida (POSITIVE o NEGATIVE)."""
    return obt[obt[OBJETIVO].notna()]


def construir_modelo():
    """Imputacion + gradient boosting. keep_empty_features deja las columnas
    que vienen vacias (un atributo que un lote no reporto) para que el modelo
    reciba siempre el mismo numero de columnas."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("modelo", HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
    ])


if __name__ == "__main__":
    smart = pd.read_csv("../data/raw/smart_c1_muestra.csv", parse_dates=[FECHA])
    fallas = pd.read_csv("../data/raw/fallas_c1.csv", parse_dates=["FAILURE_DATE"])

    obt = construir_obt(smart, fallas)
    print(f"OBT: {len(obt)} filas, {obt[LLAVE_SSD].drop_duplicates().shape[0]} SSD, "
          f"{len(columnas_modelo(obt))} columnas para el modelo")
    print(obt["LABEL_STATUS"].value_counts().to_string())
    print(f"Filas entrenables: {len(filas_entrenables(obt))}")
