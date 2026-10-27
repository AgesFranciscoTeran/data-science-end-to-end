# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

La fuga de información es cuando el modelo, al entrenar, tiene acceso a información que no tendría en el momento de predecir: datos del conjunto de prueba (por ejemplo, estadísticas calculadas con todas las filas) o columnas que solo se conocen después del evento que se quiere predecir. En la evaluación se ve excelente porque la prueba ya no es independiente: el modelo "copió" parte de la respuesta. En producción esa información no existe (o no en esa forma), así que el desempeño cae a su nivel real, que puede ser mucho peor.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La fuga por una feature que solo existe después del evento (*target leakage*): la duración del viaje tiene correlación 0.96 con la tarifa, pero se conoce al terminar el viaje, no al pedir el taxi. Un modelo entrenado con ella se vería casi perfecto y no se podría usar al momento de cotizar.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. La media y la desviación son algo que el pipeline *aprende*, igual que los coeficientes del modelo, así que no pueden ver la prueba. Numéricamente cambia poco con 2975 viajes (8.5034 con todos contra 8.5576 con entrenamiento, ver Ejercicio 1), pero lo correcto es aprender con entrenamiento y aplicar esas mismas estadísticas a prueba, igual que se hará en producción con viajes que todavía no existen.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En `X_preparado = escalador.fit_transform(imputador.fit_transform(X))`: el imputador y el escalador aprenden la mediana, la media y la desviación con las 2975 filas, y recién después se divide. Lo delata `Filas que vio el escalador: 2975.0`: si solo hubiera visto entrenamiento serían 2380 (el 80 %).

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

El error en entrenamiento: el modelo se evalúa sobre las mismas filas con las que ajustó sus coeficientes, así que no mide cómo generaliza a viajes nuevos. El 2.5590 que imprime sin los TODO no es un error de prueba (y además ese pipeline todavía no escala).

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaba que cambiara poco, y casi no cambia: **2.5375 con fuga contra 2.5373 sin fuga**. Eso no significa que no hubiera fuga: el escalador sí vio las 595 filas de prueba. Pasa que aquí la fuga es "barata": con miles de filas la mediana y la media casi no se mueven (6.93 vs 6.94; 8.5034 vs 8.5576), y una regresión lineal da las mismas predicciones con o sin escalar (solo cambian los coeficientes). Con pocos datos, con codificaciones que usan el objetivo o con selección de features hecha sobre todo el dataset, el mismo error puede inflar mucho la métrica. Que el número no cambie no prueba que el procedimiento esté bien.

**Lo que confirmó la salida, TODO por TODO:**

- Compañero: el escalador vio **2975 filas**, media de distancia **8.5034**, error absoluto medio en prueba **2.5375**.
- TODO 1: con `train_test_split(X, y, test_size=0.2, random_state=42)` antes de cualquier estadística, la prueba queda en **595 filas**.
- TODO 2: el pipeline queda `imputar` → `escalar` → `modelo`, y el script ya imprime el escalador.
- TODO 3: con `pipeline.fit(X_ent, y_ent)`, el escalador vio **2380 filas** y usó una media de **8.5576**. Error absoluto medio en prueba: **2.5373**.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

**25 columnas**: 2 numéricas escaladas (`distancia_km`, `pasajeros`) + 5 ya listas que pasan sin cambios (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico`, `es_tarifa_fija`) + 18 de one-hot: 8 zonas de origen + 8 zonas de destino + 2 métodos de pago (`efectivo`, `tarjeta`; los 60 faltantes se imputan con el más frecuente, así que no crean columna propia). Entrenamiento queda en (2380, 25) y prueba en (595, 25).

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Con las de entrenamiento: el imputador aprendió `distancia_km` = **6.94** y `pasajeros` = **1.0**, igual que las medianas de `X_ent`, mientras que con todo el dataset la distancia sería **6.93**. Demuestra que `fit_transform(X_ent)` solo usó entrenamiento, y que `transform(X_prueba)` rellena los faltantes de prueba con lo aprendido, sin recalcular nada con la prueba.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Como `Bronx` no aparece en entrenamiento y el codificador tiene `handle_unknown="ignore"`, **las 8 columnas de `zona_origen` quedan en 0**: el viaje se trata como "origen desconocido", sin error. El resto se codifica normal (`zona_destino_Midtown` y `metodo_pago_tarjeta` en 1), y la fila tiene las mismas 25 columnas, en el mismo orden, que espera el modelo. Con `pd.get_dummies` aplicado solo a ese viaje saldría una columna `zona_origen_Bronx` que el modelo nunca vio y faltarían las de las demás zonas y métodos de pago: las columnas no coincidirían con las de entrenamiento y el modelo fallaría o, si se alinean por posición, predeciría con columnas equivocadas.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

Proyecto: predecir si un SSD del modelo C1 falla (`failure`) con la lectura SMART del día. Código en [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

- **Objetivo:** `failure` (1 si el disco falló, 0 si no); 5.94 % de fallas.
- **Numéricas** (mediana + escalado): las 2 features de la sesión pasada, `n_contadores_error` y `log_r_5`, y los contadores SMART crudos `r_183`, `r_184`, `r_187`, `r_195`, `r_197`, `r_199`, `r_program`, `r_erase`, `n_wearout`, `r_12`, `r_174`. `r_5` entra como `log_r_5`.
- **Categóricas** (más frecuente + one-hot): `app` (9 valores, incluido `none`).
- **Ya listas:** ninguna.
- **Fuera:** `fecha_smart` y `r_9`/`n_9` (los sanos se midieron el 2019-12-31 y los fallados el día de la falla: delatan la clase por cómo se armó el dataset, no por el disco); `disk_id` (identificador) y `model` (siempre C1); las `n_` que repiten normalizada a su `r_`; y `r_241`, `r_242`, `n_175` con sus `n_`, 100 % vacías.

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

Al azar, estratificado por `failure`, 80/20 con `random_state=42`. El modelo sí se usaría para anticipar fallas futuras, pero este dataset no permite dividir por fecha: todos los discos sanos tienen la misma fecha (2019-12-31) y solo los fallados se reparten entre 2018-01-03 y 2019-12-31, así que cualquier fecha de corte dejaría casi solo fallas de un lado y casi solo sanos del otro. Como hay una fila por disco, la división al azar no pone un mismo disco en entrenamiento y prueba. La estratificación mantiene la misma tasa de falla en ambos lados (0.0594 y 0.0594), importante con tan pocos positivos.

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

```
Tasa de falla: entrenamiento 0.0594, prueba 0.0594
Entrenamiento: (141520, 22) Prueba: (35380, 22)
['numericas__n_contadores_error', 'numericas__log_r_5', 'numericas__r_183', 'numericas__r_184', 'numericas__r_187', 'numericas__r_195', 'numericas__r_197', 'numericas__r_199', 'numericas__r_program', 'numericas__r_erase', 'numericas__n_wearout', 'numericas__r_12', 'numericas__r_174', 'categoricas__app_DAE', 'categoricas__app_DB', 'categoricas__app_NAS', 'categoricas__app_RM', 'categoricas__app_SS', 'categoricas__app_WPS', 'categoricas__app_WS', 'categoricas__app_WSM', 'categoricas__app_none']
```

22 columnas: 13 numéricas + 9 de one-hot de `app`, sobre 176900 discos (176902 menos los 2 sin lecturas SMART). La plantilla quedó como módulo (`cargar_datos`, `dividir`, `preparacion`) para que los modelos de la Semana 10 usen exactamente la misma preparación y división.

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: aprendió detalles (ruido) de las filas de entrenamiento que no se repiten en datos nuevos. La brecha entre el error de prueba y el de entrenamiento mide cuánto memorizó en vez de generalizar. Para corregirlo se compara contra una línea base y se reduce la complejidad del modelo (regularización, menos profundidad, más datos).
