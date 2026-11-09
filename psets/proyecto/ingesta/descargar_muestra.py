"""Descarga una muestra publica de los datos del proyecto.

El dataset completo (SMART diario 2018-2019) esta en Tianchi y exige iniciar
sesion. Alibaba publica en GitHub una parte sin registro:
https://github.com/alibaba-edu/dcbrain/tree/master/ssd_open_data

  - smart_log_20191231.csv.zip: el SMART del ultimo dia (2019-12-31) de todos los SSD.
  - ssd_failure_tag.csv.zip: los tickets de falla 2018-2019.

Este script toma de ahi los SSD del modelo C1, el que reporta los mismos
atributos que MC1, y deja en ../data/raw/ (que no se sube a git):

  - smart_c1_muestra.csv: N SSD al azar (semilla fija), un dia cada uno.
  - fallas_c1.csv: las fallas de esos SSD.

    cd psets/proyecto/ingesta && python descargar_muestra.py [N]
"""

import io
import os
import sys
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelado"))
from pipeline_proyecto import normalizar_smart  # noqa: E402

BASE = "https://raw.githubusercontent.com/alibaba-edu/dcbrain/master/ssd_open_data/"
DESTINO = "../data/raw"
MODELO = "C1"
SEMILLA = 42


def leer_zip(nombre, miembro):
    respuesta = requests.get(BASE + nombre, timeout=120)
    respuesta.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(respuesta.content)) as archivo:
        return pd.read_csv(archivo.open(miembro))


if __name__ == "__main__":
    tamano = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    os.makedirs(DESTINO, exist_ok=True)

    smart = leer_zip("smart_log_20191231.csv.zip", "20191231.csv")
    smart = smart[smart["model"] == MODELO].sample(n=tamano, random_state=SEMILLA)
    smart = normalizar_smart(smart)
    smart.to_csv(f"{DESTINO}/smart_c1_muestra.csv", index=False)

    fallas = leer_zip("ssd_failure_tag.csv.zip", "ssd_failure_tag.csv")
    fallas = fallas[fallas["model"] == MODELO]
    fallas = pd.DataFrame({
        "DISK_ID": fallas["disk_id"].astype("int64"),
        "MODEL_CODE": fallas["model"],
        "FAILURE_DATE": pd.to_datetime(fallas["failure_time"]).dt.normalize(),
    })
    fallas = fallas[fallas["DISK_ID"].isin(smart["DISK_ID"])]
    fallas.to_csv(f"{DESTINO}/fallas_c1.csv", index=False)

    print(f"{len(smart)} SSD {MODELO} en {DESTINO}/smart_c1_muestra.csv, "
          f"{len(fallas)} fallas en {DESTINO}/fallas_c1.csv")
