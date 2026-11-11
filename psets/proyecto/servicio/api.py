"""Servicio de prediccion de fallas de SSD.

Recibe la lectura SMART del dia de un SSD C1, como la reporta el disco, y
devuelve su probabilidad de falla y si va a la lista de revision. El lote
diario (docs/despliegue.md) lo llama una vez por disco, o un tablero lo
consulta para un disco en particular.

Desde psets/proyecto/, despues de python servicio/exportar_modelo.py:

    uvicorn servicio.api:app --port 8000

La documentacion interactiva queda en el puerto 8000, ruta /docs.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

# Las features se calculan con la misma funcion del entrenamiento, importada
# del pipeline del proyecto, no copiada.
from modelado.pipeline_proyecto import COLUMNAS, agregar_features

AQUI = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", AQUI / "modelo.joblib"))
RUTA_INFO = RUTA_MODELO.with_suffix(".json")
# El umbral mas barato de modelado/validacion_proyecto.ipynb: una falla sin
# aviso cuesta 10 veces mas que una falsa alarma.
UMBRAL = 0.10

# El modelo se carga una sola vez, al arrancar el servicio.
modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None
info_modelo = json.loads(RUTA_INFO.read_text(encoding="utf-8")) if RUTA_INFO.exists() else {}

app = FastAPI(title="Fallas de SSD")

CONTADORES = ["r_5", "r_183", "r_184", "r_187", "r_195", "r_197", "r_199",
              "r_program", "r_erase", "n_wearout", "r_12", "r_174"]


class LecturaSmart(BaseModel):
    """La lectura SMART de un SSD C1 en un dia, como la reporta el disco.
    Un disco puede no reportar algun atributo: cada contador es opcional y
    la preparacion imputa los que falten, pero tiene que llegar al menos uno.
    Los contadores nunca son negativos."""
    disk_id: int
    app: str | None = Field(None, description="La aplicacion del servidor (WSM, NAS, RM, ...)")
    r_5: float | None = Field(None, ge=0, description="Sectores reasignados")
    r_183: float | None = Field(None, ge=0)
    r_184: float | None = Field(None, ge=0)
    r_187: float | None = Field(None, ge=0, description="Errores no corregibles")
    r_195: float | None = Field(None, ge=0)
    r_197: float | None = Field(None, ge=0, description="Sectores pendientes")
    r_199: float | None = Field(None, ge=0, description="Errores CRC")
    r_program: float | None = Field(None, ge=0)
    r_erase: float | None = Field(None, ge=0)
    n_wearout: float | None = Field(None, ge=0)
    r_12: float | None = Field(None, ge=0, description="Encendidos")
    r_174: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def al_menos_un_contador(self):
        # agregar_features descarta los SSD sin ninguna lectura: sin este
        # chequeo, el modelo recibiria una tabla vacia.
        if all(getattr(self, contador) is None for contador in CONTADORES):
            raise ValueError("La lectura no trae ningun contador SMART")
        return self


def preparar(lectura: LecturaSmart):
    """La lectura como DataFrame de una fila, con las features del pipeline."""
    return agregar_features(pd.DataFrame([lectura.model_dump()]))[COLUMNAS]


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "run_id": info_modelo.get("run_id")}


@app.post("/predecir")
def predecir(lectura: LecturaSmart):
    probabilidad = float(modelo.predict_proba(preparar(lectura))[0, 1])
    return {
        "disk_id": lectura.disk_id,
        "probabilidad_falla": round(probabilidad, 4),
        "revisar": probabilidad >= UMBRAL,
        "run_id": info_modelo.get("run_id"),
    }
