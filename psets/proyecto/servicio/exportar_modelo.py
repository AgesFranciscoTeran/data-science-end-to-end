"""Exporta el modelo final desde MLflow a servicio/modelo.joblib.

Busca en el experimento fallas-ssd la corrida mas reciente marcada con
candidato = final (la marca experimentos_proyecto.ipynb), carga su pipeline y
lo guarda con joblib. Junto a el deja modelo.json con el run_id y la metrica,
para que el servicio diga que modelo esta sirviendo. Ninguno de los dos
archivos se sube a git: se regeneran con este script.

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py            # la corrida final mas reciente
    python servicio/exportar_modelo.py <run_id>   # una corrida en particular
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

EXPERIMENTO = "fallas-ssd"
AQUI = Path(__file__).resolve().parent
RUTA_MODELO = AQUI / "modelo.joblib"
RUTA_INFO = AQUI / "modelo.json"


def corrida_final():
    experimento = mlflow.get_experiment_by_name(EXPERIMENTO)
    if experimento is None:
        sys.exit(f"No existe el experimento {EXPERIMENTO}: corre primero modelado/experimentos_proyecto.ipynb")
    corridas = mlflow.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="tags.candidato = 'final'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(corridas) == 0:
        sys.exit(f"Ninguna corrida de {EXPERIMENTO} tiene candidato = final")
    return corridas.iloc[0]["run_id"]


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else corrida_final()
    corrida = mlflow.get_run(run_id)
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")

    joblib.dump(modelo, RUTA_MODELO)
    info = {
        "run_id": run_id,
        "nombre": corrida.data.tags.get("mlflow.runName"),
        "ap_prueba": round(corrida.data.metrics["ap_prueba"], 4),
        "huella_datos": corrida.data.params.get("huella_datos"),
        "exportado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RUTA_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Modelo {info['nombre']} (run_id {run_id}, AP de prueba {info['ap_prueba']}) guardado en {RUTA_MODELO}")
