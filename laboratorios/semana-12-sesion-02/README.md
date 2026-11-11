# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

Cuatro pasos. (1) **Guardar el modelo entrenado** en un archivo (con `pickle` o `joblib`), con la preparación incluida, para no reentrenar cada vez. (2) **Envolverlo en una API web**: un servicio que recibe los datos de entrada por HTTP, valida que tengan la forma correcta, calcula las features, llama al modelo y devuelve la predicción como JSON. (3) **Empaquetarlo en un contenedor** con sus dependencias. (4) **Desplegarlo en un servidor o en la nube**, donde otra aplicación lo llama por su dirección, y vigilar que siga respondiendo. Entre medio están las pruebas del contrato de la API: que acepte lo que debe y rechace lo que no.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Porque correr en la máquina de quien lo escribió no garantiza que corra en otra. El servicio depende de una versión de Python y de librerías exactas (un modelo guardado con joblib solo carga con la misma versión de scikit-learn), y de archivos que están a su lado. El contenedor empaqueta todo eso junto con el modelo, así que la misma imagen corre igual en la laptop, en el servidor y en la nube, sin el "en mi máquina sí funciona". Además permite levantar varias copias del servicio cuando llegan más peticiones. Lo vimos hoy: dentro del contenedor, el servicio corre con scikit-learn 1.5.2 aunque la máquina tenga otra versión.

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Tendría que existir un **servicio en línea**: un proceso que corre todo el tiempo con el modelo ya cargado en memoria y expone una ruta HTTP (por ejemplo `POST /predecir`). Cuando el pasajero elige el destino, la app le manda lo que sabe del viaje antes de que empiece: la fecha y hora, la zona de origen y la de destino, la distancia estimada, los pasajeros y el método de pago. El servicio calcula el resto de las features con las mismas funciones del entrenamiento (la hora como seno y coseno, si es hora pico, si es tarifa fija al aeropuerto) y responde la tarifa en milisegundos, mientras el pasajero mira la pantalla.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo se ve sano y nada funciona. Sin resolver, `/predecir` siempre devuelve `{"tarifa_estimada": 0.0}`: el esquema no declara la fecha, la distancia ni los pasajeros, y el código nunca llama al modelo. `/salud` solo dice que el archivo se cargó, no que el modelo se use, y las 3 pruebas pasan porque dos de ellas no tienen ningún `assert`. Un 200 y un `passed` no dicen que la respuesta sea correcta.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Sin `distancia_km`, el servicio igual estima una tarifa: el campo es opcional (`None` por defecto) y la preparación lo imputa con la mediana. El viaje de Midtown al JFK sin distancia ni pasajeros recibió 52.55. Sin `zona_origen`, responde **422**, porque es obligatoria.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo recibiría una feature con un significado distinto del que aprendió, y se equivocaría sin dar ningún error: la petición respondería 200 con una tarifa plausible pero mal calculada, por ejemplo cobrando la congestión de hora pico a las 11:00. Por eso el servicio importa `agregar_features` de `features_taxis.py` en lugar de reescribirla: es la misma función con que se armaron las features de entrenamiento.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce **FastAPI**, no mi código. Pydantic valida el JSON contra el esquema `Viaje` antes de llamar a `predecir`, y al faltar `zona_origen` responde 422 con `{"type": "missing", "loc": ["body", "zona_origen"], "msg": "Field required"}`. La petición **no llega al modelo**: la función `predecir` ni siquiera se ejecuta.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Sin `--host`, uvicorn escucha en `127.0.0.1`, que dentro del contenedor es la interfaz del propio contenedor. `-p 8000:8000` reenvía el tráfico a la interfaz de red del contenedor, donde nadie escucha, así que la conexión falla aunque el log diga `Uvicorn running on http://127.0.0.1:8000`. Con `--host 0.0.0.0`, uvicorn escucha en todas las interfaces y `curl localhost:8000/salud` responde. (Ojo: el comando tiene que terminar en `.`, la carpeta de contexto del build: `docker build -t servicio-tarifa .`.)

**Lo que respondió el servicio:**

Desde el contenedor (`docker run -p 8000:8000 servicio-tarifa`): `/salud` respondió `{"estado":"ok","modelo_cargado":true}`. El viaje de Midtown al JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta) respondió `{"tarifa_estimada":70.39}`. Sin zona de origen respondió **422**: `{"detail":[{"type":"missing","loc":["body","zona_origen"],"msg":"Field required","input":{"fecha_hora_inicio":"2026-11-05 08:30","zona_destino":"JFK Airport","distancia_km":26.4,"pasajeros":2,"metodo_pago":"tarjeta"}}]}`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

El pago de madrugada (02:40, por app, tarjeta nueva, 2 intentos fallidos, 0 viajes previos, 41.30 dólares) recibió `{"probabilidad_fraude": 0.9996, "revisar": true}`. El de la tarde (14:10, tarjeta conocida, 0 intentos fallidos, 22 viajes previos, 19.80 dólares) recibió `{"probabilidad_fraude": 0.0159, "revisar": false}`.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app no tiene que conocer el umbral ni la regla de negocio: recibe la decisión y actúa (detener el cobro o dejarlo pasar). Así la regla vive en un solo lugar y todas las apps la aplican igual. Si mañana el umbral pasa a 0.20, solo cambia `UMBRAL` en `servicio_fraude.py` y se vuelve a desplegar el servicio (construir la imagen y reiniciar el contenedor). Las apps no cambian, y la probabilidad tampoco, porque el modelo es el mismo.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí: con el mismo modelo, el pago de madrugada da 0.9996 en los dos. Para eso tienen que ser iguales tres cosas: el **mismo archivo de modelo** (el mismo `.joblib`, entrenado con los mismos datos y semilla), la **misma preparación** (los dos importan `agregar_features` y `columnas` de `preparacion_pagos.py`, así que `hora` y `monto_por_km` se calculan igual) y las **mismas versiones de librerías**, sobre todo scikit-learn. Si el lote y el servicio cargan modelos entrenados en días distintos, o uno calcula la hora con otra zona horaria, la misma entrada da otra probabilidad.

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde **422**: `{"detail":[{"type":"float_parsing","loc":["body","monto"],"msg":"Input should be a valid number, unable to parse string as a number","input":"cuarenta"}]}`. Lo produce la línea `monto: float` del esquema `Pago`: Pydantic intenta convertir el texto a número, no puede, y FastAPI rechaza la petición antes de llamar a `evaluar_pago`.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe la lectura SMART del día de un SSD como la reporta el disco: `disk_id`, la `app` del servidor y los 12 contadores crudos (`r_5`, `r_183`, `r_184`, `r_187`, `r_195`, `r_197`, `r_199`, `r_program`, `r_erase`, `n_wearout`, `r_12`, `r_174`), todos opcionales, sin negativos y con al menos uno presente. El servicio calcula las dos features derivadas: `n_contadores_error` (cuántos de `r_5`, `r_187`, `r_197` y `r_199` están por encima de 0) y `log_r_5`. Sé que las calcula igual por dos razones. Primero, `servicio/api.py` importa `agregar_features` de `modelado/pipeline_proyecto.py`, la misma función con que se entrenó, en vez de copiarla. Segundo, la prueba `test_el_servicio_calcula_las_features_igual_que_el_entrenamiento` comprueba que lo que el servicio le pasa al modelo es exactamente lo que `agregar_features` produce para la misma fila. Además, el modelo exportado es el pipeline completo (imputación, escalado y codificación de `app` incluidos), así que esa parte tampoco se reescribe.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Contenedor construido desde `psets/proyecto/` con `docker build -f servicio/Dockerfile -t servicio-ssd .` y levantado con `docker run -p 8000:8000 servicio-ssd`, con el modelo exportado de la corrida `5ab7a7cd78bd482ea2333f03602b3922`:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"disk_id": 88935, "app": "RM", "r_5": 8, "r_183": 0, "r_184": 0, "r_187": 112, "r_195": 0, "r_197": 0, "r_199": 2, "r_program": 0, "r_erase": 0, "n_wearout": 100, "r_12": 24, "r_174": 19}'
{"disk_id":88935,"probabilidad_falla":0.7907,"revisar":true,"run_id":"5ab7a7cd78bd482ea2333f03602b3922"}
```

Ese SSD falló en 2018. Un SSD sano (`disk_id` 100048, todos los contadores de error en 0) recibió `{"probabilidad_falla":0.1378,"revisar":true}`: con el umbral 0.10, que minimiza el costo, va a revisión como falsa alarma. Una lectura sin ningún contador recibió 422 (`"La lectura no trae ningun contador SMART"`).

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Para saber qué modelo produjo una predicción, cada respuesta tiene que llevar su identificador. Mi servicio ya lo hace a medias: `exportar_modelo.py` guarda el `run_id` de MLflow en `modelo.json`, y el servicio lo devuelve en `/salud` y en cada respuesta de `/predecir`. Si quien llama guarda la respuesta completa (o el servicio registra cada petición con su `run_id` y su hora), ante un reclamo se busca la predicción y su `run_id` dice exactamente qué corrida la produjo, con sus datos, parámetros y métricas. Para volver al modelo anterior, como MLflow guarda todas las corridas, basta con exportar el `run_id` anterior (`python servicio/exportar_modelo.py <run_id_anterior>`), reconstruir la imagen y reiniciar el contenedor. Mejor aún: etiquetar cada imagen con su `run_id` (`servicio-ssd:5ab7a7cd`), para que volver atrás sea solo levantar la imagen anterior, o usar el registro de modelos de MLflow con versiones y un alias como `produccion`, que se mueve de una versión a otra sin tocar el código.
