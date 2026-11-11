# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada, y tu `pipeline_proyecto.py`.

Como la sesión pasada, los laboratorios son notebooks que importan la preparación de un script (`preparacion_taxis.py` en los Ejercicios 1 y 2, tu `pipeline_proyecto.py` en el 3).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje automático clásico conviene para **datos tabulares y estructurados** (filas con columnas numéricas o categóricas) y problemas bien definidos de predicción, como estimar un número o una clase, cuando hay datos etiquetados de sobra: es barato de entrenar y de correr, rápido, estable y se puede explicar. Un LLM conviene para **texto y lenguaje no estructurado** (resumir, clasificar o extraer información de documentos, conversar) y para tareas con pocos o ningún ejemplo etiquetado, donde lo que importa es el conocimiento general y la comprensión del lenguaje, aceptando que cada predicción cuesta más y es menos predecible.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

El **costo**. Mi proyecto puntúa los 176.900 SSD de la flota todos los días: con un modelo clásico eso tarda menos de un segundo y no cuesta nada por predicción; con un LLM serían miles de dólares al mes (en `psets/proyecto/modelado/comparar_proyecto.ipynb`, unos 3.700 dólares al mes con Claude Opus 5.5). Los datos ya son tabulares y hay 141.520 filas etiquetadas para entrenar, así que el LLM no aporta ninguna ventaja que pague ese costo.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error, mediría: el **costo** de entrenar y, sobre todo, de cada predicción (la app estima miles de viajes al día); la **latencia**, porque la tarifa se muestra mientras el pasajero espera; la **estabilidad**, es decir, si el mismo viaje recibe siempre la misma tarifa (al pasajero se le cobra ese número); la **explicabilidad**, para poder decirle al conductor o al pasajero por qué la tarifa es esa; y qué tan bien se comporta con **casos raros**, como la tarifa fija al aeropuerto.

## Ejercicio 1: una red neuronal frente al ensamblaje

Notebook: [`starter/ejercicio1_red_neuronal.ipynb`](./starter/ejercicio1_red_neuronal.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). Ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división, importadas de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Que el optimizador llegó al máximo de iteraciones (200 por defecto) antes de converger: la red seguía mejorando cuando se detuvo, así que sus pesos quedaron a medio entrenar. Con 1000 iteraciones esperaba que convergiera y bajara su error, quizás hasta igualar al ensamblaje, a cambio de tardar mucho más. Así fue: el MAE de prueba bajó a **1.1695**, mejor que el gradient boosting (1.2754), pero tardó **245 s** contra 0.18 s.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

**1729** números. La capa oculta tiene 25 × 64 = 1600 pesos más 64 sesgos (1664), y la salida tiene 64 pesos más 1 sesgo (65): 1664 + 65 = 1729.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

La **distancia**: la tarifa del taxímetro depende sobre todo de los kilómetros. Así fue: `distancia_km` tiene una importancia de **0.982**, seguida de `es_tarifa_fija` (0.012) y `es_hora_pico` (0.003). Para explicarle al conductor una tarifa de 70 dólares, el **ensamblaje** permite responder: se puede ver qué columnas pesan (y, por viaje, que `es_tarifa_fija` = 1 la lleva a 70). La red mezcla las 25 columnas en 1729 pesos y no da una razón legible.

**Lo que confirmó la salida, TODO por TODO:**

Gradient boosting: MAE de entrenamiento **1.0705**, de prueba **1.2754**, **0.18 s** de entrenamiento. Red neuronal (64 neuronas, 1000 iteraciones): entrenamiento **1.0882**, prueba **1.1695**, **244.6 s**. La red tiene **1729** parámetros. Las 3 columnas más importantes del ensamblaje: `distancia_km` (0.982), `es_tarifa_fija` (0.012) y `es_hora_pico` (0.003).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Notebook: [`starter/ejercicio2_tres_soluciones.ipynb`](./starter/ejercicio2_tres_soluciones.ipynb), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre las celdas hasta la que escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y corre sus celdas y la de la comprobación hasta que todas digan `OK`. Los modelos ya están entrenados en el kernel: no hace falta volver a correr las celdas de arriba. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé **Claude Fable 5.1**, con un precio de entrada de USD 10 por millón de tokens. El error es el MAE en dólares sobre los mismos 20 viajes.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | **0.8957** | 0.34 s en una laptop | prácticamente 0 (corre en la misma máquina, milisegundos por viaje) | Sí: importancia de cada columna (la distancia pesa 0.98) |
| Red neuronal | **0.9637** | 176 a 245 s en una laptop | prácticamente 0 | Poco: 1729 pesos sin significado propio |
| LLM | **1.2745** | 0 (no se entrena; 30 ejemplos en el prompt) | **USD 3.63**: 726 tokens por prompt de 20 viajes, 500 prompts al mes, USD 10 por millón | Da una razón en texto, pero no garantiza que sea la que usó, y cambia entre corridas |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

_(tu respuesta)_

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

Fue un viaje **dentro del aeropuerto JFK** (de JFK Airport a JFK Airport, 1.01 km, en hora pico, en efectivo): la tarifa real fue **11.46** y el LLM respondió **7.60**, 3.86 dólares menos. El LLM lo trató como un viaje corto cualquiera, que fuera del aeropuerto cuesta unos 6.48 dólares. Pero los viajes dentro del aeropuerto cuestan más: en las 2380 filas de entrenamiento hay 7, con tarifas de 12.39 a 18.67 dólares (mediana 15.25). El ensamblaje ve esos 7 viajes y la combinación de zona de origen y destino, así que sí podía aprenderlo; el LLM solo vio 30 ejemplos y probablemente ninguno era de ese tipo.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Porque cada viaje cuesta tokens y hay que pegarlo a mano en el chat: 595 viajes serían 30 prompts, o uno enorme, y el costo y el trabajo crecen con cada viaje. Una comparación sobre 20 es **poco confiable**: con tan pocos viajes, uno solo raro (como el de JFK a JFK, con 3.86 dólares de error) mueve el promedio, y el orden entre modelos puede cambiar con otra muestra. De hecho, en los 20 viajes el gradient boosting (0.8957) le gana a la red (0.9637), al revés que en los 595 de prueba (1.2754 frente a 1.1695). Sirve para tener una idea, no para decidir.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí tu preparación y tu división, y entrene el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

Notebook: `psets/proyecto/modelado/comparar_proyecto.ipynb` (tabla en `comparar_proyecto.csv`, prompt en `prompt_llm_proyecto.txt` y respuestas en `respuestas_llm_proyecto.txt`). El LLM fue Claude Opus 5.5: respondió el prompt leyendo solo los 30 ejemplos y los 10 SSD sin etiqueta. Los 10 SSD son 5 que fallaron y 5 sanos, porque al azar, con 5,94 % de fallas, saldrían 0 o 1 fallas.

| | Average precision en prueba (35.380 SSD) | Average precision en los 10 SSD | Costo | ¿Explica una predicción? |
|---|---|---|---|---|
| Bosque (`min_samples_leaf=2`) | **0.5003** (AUC 0.8619) | **0.9429** | 2.3 s de entrenamiento; puntúa los 176.900 SSD en 0.5 s, sin costo por predicción | Sí: importancia de cada contador y reglas de los árboles |
| Red neuronal 32-16 | 0.3593 (AUC 0.8102) | 0.7862 | 7.3 s de entrenamiento, 1281 parámetros; puntuar es casi gratis | Poco: 1281 pesos sin significado |
| LLM (Claude Opus 5.5) | no medible en 35.380 SSD | 0.6111 | Sin entrenamiento, pero unos 1758 tokens por cada 10 SSD: USD 124 al día, unos USD 3.700 al mes | Da una explicación en texto, pero no garantiza que sea la razón real, y cambia de una corrida a otra |

Usaría el **bosque aleatorio**. Es el que mejor ordena los discos por riesgo, tanto en los 35.380 SSD de prueba (average precision 0.5003, frente a 0.3593 de la red) como en los mismos 10 del LLM (0.9429, frente a 0.7862 de la red y 0.6111 del LLM). Además es el más barato: entrena en 2.3 s y puntúa toda la flota en medio segundo, mientras el LLM costaría unos 3.700 dólares al mes por un resultado peor. El LLM no tiene nada que aprovechar: los datos son contadores numéricos, no texto, y con 30 ejemplos no puede aprender lo que el bosque saca de 141.520 discos. En los 10 SSD, de hecho, le dio probabilidades casi iguales (0.30 a 0.45) a todos los discos con contadores en cero, fallaran o no. Y el bosque también se puede explicar, porque dice qué contadores pesan en cada predicción.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

De cada corrida necesitaría: **qué datos** usó (el archivo exacto, con una huella que lo pruebe), **cómo los dividió** (la semilla o la fecha de corte, y la proporción de prueba), **qué preparación** aplicó, los **hiperparámetros** del modelo (incluido su `random_state`), las **versiones** de las librerías y **cómo midió el error**. Si todo eso coincide, el error tiene que coincidir; la diferencia está en lo que no coincide. Sin un registro de experimentos, nada de eso queda escrito.
