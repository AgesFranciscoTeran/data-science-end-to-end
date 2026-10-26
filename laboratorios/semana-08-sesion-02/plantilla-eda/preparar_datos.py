"""Arma el dataset del proyecto (PSet 2, fallas de SSD) con los datos abiertos
de Alibaba en https://github.com/alibaba-edu/dcbrain/tree/master/ssd_open_data

La serie diaria completa que uso el PSet 2 solo se baja de Tianchi con cuenta.
Lo abierto son tres archivos:
  - ssd_failure_tag: los SSD que fallaron, con su SMART al momento de la falla.
  - smart_log_20191231: el SMART de todos los SSD en un solo dia, 2019-12-31.
  - location_info_of_ssd: la aplicacion (app) de cada SSD.

Se queda solo con el modelo C1, que es el MC1 del PSet 2. Una fila = un SSD:
failure = 1 si fallo (SMART del momento de la falla), 0 si nunca fallo
(SMART del 2019-12-31). Guarda ssd_mc1.csv en la carpeta actual.
"""
import io
import urllib.request
import zipfile

import pandas as pd

URL = "https://raw.githubusercontent.com/alibaba-edu/dcbrain/master/ssd_open_data/"
MODELO = "C1"


def leer_zip(nombre):
    print(f"Descargando {nombre}")
    with urllib.request.urlopen(URL + nombre) as respuesta:
        archivo = zipfile.ZipFile(io.BytesIO(respuesta.read()))
    # Los zip traen basura de macOS (__MACOSX/._archivo.csv).
    csv = [n for n in archivo.namelist() if not n.startswith("__MACOSX")][0]
    return pd.read_csv(archivo.open(csv))


fallas = leer_zip("ssd_failure_tag.csv.zip")
dia = leer_zip("smart_log_20191231.csv.zip")
ubicacion = leer_zip("location_info_of_ssd.csv.zip")

fallas = fallas[fallas["model"] == MODELO]
dia = dia[dia["model"] == MODELO].copy()

# ssd_failure_tag junta atributos que significan lo mismo con distinto ID
# segun el fabricante. Se hace lo mismo con el dia completo.
dia = dia.assign(
    r_program=dia["r_171"].fillna(dia["r_181"]),
    n_program=dia["n_171"].fillna(dia["n_181"]),
    r_erase=dia["r_172"].fillna(dia["r_182"]),
    n_erase=dia["n_172"].fillna(dia["n_182"]),
    n_blocks=dia["n_170"].fillna(dia["n_180"]),
    n_wearout=dia["n_173"].fillna(dia["n_177"]).fillna(dia["n_233"]),
)

# Sanos: discos C1 del 2019-12-31 que no aparecen en las fallas. La clave es
# (model, disk_id): el mismo disk_id se repite entre modelos distintos.
clave = ["model", "disk_id"]
fallados = fallas[clave].assign(_fallo=True)
dia = dia.merge(fallados, on=clave, how="left")
print(f"Discos fallados que siguen reportando el 2019-12-31: {dia['_fallo'].notna().sum()}")
sanos = dia[dia["_fallo"].isna()].drop(columns="_fallo").copy()
sanos = sanos.merge(ubicacion[clave + ["app"]].drop_duplicates(clave), on=clave, how="left")
sanos = sanos.assign(failure=0, fecha_smart=pd.Timestamp("2019-12-31"))

fallas = fallas.assign(fecha_smart=pd.to_datetime(fallas["failure_time"]).dt.normalize())

columnas = (["model", "disk_id", "app", "fecha_smart", "failure"]
            + [c for c in fallas.columns if c.startswith(("r_", "n_"))])
datos = pd.concat([fallas[columnas], sanos[columnas]], ignore_index=True)
datos.to_csv("ssd_mc1.csv", index=False)
print(f"Guardado ssd_mc1.csv: {len(datos)} filas, {int(datos['failure'].sum())} fallas")
