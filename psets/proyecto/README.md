# Predicción de fallas de SSD en un centro de datos

Un centro de datos con cientos de miles de SSD pierde servicio y arriesga datos cada vez que un disco falla sin aviso: el servidor queda degradado, hay que reemplazar el disco de urgencia y reconstruir el arreglo. Este proyecto le da al equipo de operaciones una lista diaria de los SSD con más riesgo de fallar, para migrar sus datos y reemplazarlos antes de que fallen. El modelo lee los contadores SMART de cada SSD del modelo C1 (el MC1 del PSet 2) y estima la probabilidad de que ese disco esté fallando.

**Métrica de éxito:** average precision (área bajo la curva precisión-recall), porque solo falla el 5,94 % de los discos y la exactitud de "nunca falla" ya es 94 %. Para el negocio, el costo de los errores: una falla sin aviso cuesta unos 500 dólares y una falsa alarma unos 50. El umbral que minimiza ese costo es **0.10** (`modelado/validacion_proyecto.ipynb`).

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| Datos abiertos de SSD de Alibaba ([`alibaba-edu/dcbrain`](https://github.com/alibaba-edu/dcbrain/tree/master/ssd_open_data)) | `ssd_failure_tag`: los SSD que fallaron en 2018 y 2019, con su SMART al momento de la falla. `smart_log_20191231`: el SMART de todos los SSD el 2019-12-31. `location_info_of_ssd`: la aplicación de cada SSD | Descarga directa por HTTPS, sin cuenta | `data/processed/ssd_mc1.csv` |

`ssd_mc1.csv` tiene una fila por SSD C1: los que fallaron (con `failure = 1` y su SMART del día de la falla) y los que nunca fallaron (con `failure = 0` y su SMART del 2019-12-31). Son 176.902 filas, de las que 176.900 tienen alguna lectura SMART y 10.508 son fallas. La serie diaria completa del PSet 2 está en Tianchi, que exige cuenta; esta es la parte pública de los mismos datos.

Los datos no están en el repositorio. Para regenerarlos desde cero:

```
cd psets/proyecto/ingesta
python preparar_datos.py
```

## Cómo correr el pipeline de datos

El pipeline completo del PSet 2 (Kestra, dbt sobre Snowflake y Spark para la OBT diaria) está en el repositorio del grupo. En este fork, el pipeline de datos es `ingesta/preparar_datos.py`. Baja los tres archivos abiertos, se queda con el modelo C1, une los atributos SMART que significan lo mismo con distinto ID según el fabricante (`r_program`, `r_erase`, `n_blocks`, `n_wearout`), cruza la aplicación de cada SSD y deja `data/processed/ssd_mc1.csv`.

## Cómo entrenar y evaluar el modelo

Todo vive en `modelado/` y corre en el dev container del curso, con el servidor de MLflow levantado (puerto 5050; si no está, `bash .devcontainer/mlflow-server.sh`):

1. `pipeline_proyecto.py`: la limpieza, las features (`n_contadores_error`, `log_r_5`), la división estratificada (80/20, semilla 42) y la preparación. Los notebooks lo importan, no lo copian. Sus pruebas están en `test_pipeline_proyecto.py`.
2. `experimentos_proyecto.ipynb`: entrena la línea base, la regresión logística, el bosque aleatorio y la red neuronal, registra cada corrida en el experimento `fallas-ssd` de MLflow, marca la final (`candidato` = `final`) y comprueba que el modelo cargado desde el registro reproduce su métrica. Exporta `corridas_proyecto.csv`.
3. `validacion_proyecto.ipynb`: validación cruzada estratificada frente a al azar, de 5 pliegues, del modelo final y el segundo, y el costo de los errores para cada umbral. Exporta `validacion_proyecto.csv` y `costo_proyecto.csv`.

Las pruebas corren desde la raíz del repositorio con `pytest psets/proyecto -v`, y GitHub Actions las corre en cada push.

## Modelo final

**Bosque aleatorio** de 200 árboles con hojas de al menos 2 discos (`min_samples_leaf=2`), `run_id` `5ab7a7cd78bd482ea2333f03602b3922` en el experimento `fallas-ssd`. Average precision de prueba **0.5003** (AUC 0.8619). En validación cruzada estratificada, 0.5134 ± 0.0083, frente a 0.3553 ± 0.0184 de la red neuronal, el segundo mejor. Con el umbral 0.10 anticipa el 68 % de las fallas y reduce el costo de los errores a la mitad respecto de no tener modelo. Se sirve por lotes, una vez al día: el razonamiento está en `docs/despliegue.md`.

## Cómo correr el servicio

El servicio (`servicio/api.py`, FastAPI) recibe la lectura SMART del día de un SSD y devuelve su probabilidad de falla y si va a revisión. Calcula `n_contadores_error` y `log_r_5` con la misma función `agregar_features` de `modelado/pipeline_proyecto.py`.

**1. Exportar el modelo final desde MLflow.** Desde `psets/proyecto/`, con el servidor de MLflow levantado:

```
python servicio/exportar_modelo.py
```

Busca la corrida más reciente con `candidato` = `final` en `fallas-ssd` y deja `servicio/modelo.joblib` y `servicio/modelo.json` (el `run_id` y la métrica). Ninguno de los dos se sube a git. Para exportar otra corrida: `python servicio/exportar_modelo.py <run_id>`.

**2. Levantarlo sin contenedor** (desde `psets/proyecto/`):

```
uvicorn servicio.api:app --port 8000
```

**3. Construir y levantar el contenedor** (también desde `psets/proyecto/`, porque la imagen necesita `modelado/` además de `servicio/`):

```
docker build -f servicio/Dockerfile -t servicio-ssd .
docker run -p 8000:8000 servicio-ssd
```

La imagen usa scikit-learn 1.5.2, la misma versión con que se entrenó el modelo. La documentación interactiva queda en el puerto 8000, ruta `/docs`.

**4. Pedir una predicción.** Un SSD que falló en 2018 (8 sectores reasignados, 112 errores no corregibles, 2 errores CRC):

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"disk_id": 88935, "app": "RM", "r_5": 8, "r_183": 0, "r_184": 0, "r_187": 112, "r_195": 0, "r_197": 0, "r_199": 2, "r_program": 0, "r_erase": 0, "n_wearout": 100, "r_12": 24, "r_174": 19}'
```

```json
{"disk_id": 88935, "probabilidad_falla": 0.7907, "revisar": true, "run_id": "5ab7a7cd78bd482ea2333f03602b3922"}
```

Los contadores son opcionales (la preparación imputa los que falten), pero tiene que llegar al menos uno, y ninguno puede ser negativo: si no, el servicio responde 422. `GET /salud` dice si el modelo está cargado y cuál `run_id` está sirviendo.
