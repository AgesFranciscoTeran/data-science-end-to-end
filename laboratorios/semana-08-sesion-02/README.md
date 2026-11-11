# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los laboratorios de hoy son notebooks de Jupyter, como el diagnóstico de la Semana 4: cada gráfico aparece debajo de la celda que lo dibuja. Una exploración se lee para decidir qué hacer con los datos; por eso va en un notebook y no en un script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Qué forma tienen los datos (cuántas filas y columnas, de qué tipo), qué falta (valores vacíos por columna), qué valores no tienen sentido (errores de captura, imposibles como distancias negativas), cómo se distribuye cada variable y en especial la que se quiere predecir (sesgo, colas, atípicos), y qué variables se relacionan con el objetivo y entre sí. Responderlas antes de entrenar evita que el modelo aprenda de errores y dice qué hay que limpiar o transformar.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para una sola variable: el resumen estadístico (`describe()`: media, mediana, cuartiles, mínimo y máximo), el histograma y el diagrama de caja, que muestra los atípicos. Para la relación entre dos: el diagrama de dispersión cuando las dos son numéricas, la comparación de promedios por grupo (o cajas por categoría) cuando una es categórica, y la matriz de correlaciones para ver muchas parejas a la vez.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo se desvía para acomodar esos 15 viajes: con error cuadrático, cada 999.99 pesa muchísimo más que un viaje normal, así que las estimaciones suben para todos los viajes parecidos (la media de la tarifa salta de 26.52 a 31.39 dólares). Mirando solo el resultado final probablemente no lo notarías: verías un error algo peor y lo atribuirías al modelo, no a los datos. Solo aparece si miras la distribución de la tarifa o el máximo de `describe()`, que es justo lo que hace la exploración.

## Ejercicio 1: resumen, distribución y atípicos

Notebook: [`starter/ejercicio1_explorar.ipynb`](./starter/ejercicio1_explorar.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Ya lee el archivo y arma los gráficos; faltan 4 `TODO`. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El **máximo de la tarifa, 999.99**, demasiado redondo y lejos de todo lo demás (el siguiente valor es 77.81), y el **mínimo de la distancia, 0 km**, un viaje que no se movió pero cobró. También la desviación estándar de la tarifa (70.99), mayor que su media (31.39), delata que hay valores extremos.

**Respuesta 2 (TODO 2):** mira el histograma después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

Porque los 15 viajes de 999.99 estiran el eje horizontal hasta 1000: el histograma reparte sus barras entre 0 y 1000, y los 2985 viajes reales, que están entre 1.60 y 77.81, caen en las primeras barras. Al quitar esos errores, el histograma muestra la forma real: sesgada a la derecha, con un pico aparte en 70.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No. La regla del IQR solo dice que un valor está lejos del centro, no que esté mal. Lo averiguaría mirando los atípicos agrupados: qué valores se repiten y en qué viajes. Ahí se ve que la mayoría tiene exactamente 70 dólares y siempre entre JFK y Manhattan: es la **tarifa fija del aeropuerto**, un precio real. Otros son viajes largos (19 a 24 km) con tarifas que corresponden a esa distancia. Solo los 999.99 son errores.

**Lo que confirmó la salida, TODO por TODO:**

El límite superior del IQR es **60.40** (Q1 = 13.99, Q3 = 32.55). Marcó **334 viajes** atípicos: **302** con tarifa fija de 70 dólares (todos entre JFK y Manhattan), **15** con 999.99 (errores de captura) y **17** viajes largos, de 18.8 a 24.4 km, con tarifas altas pero coherentes con la distancia. Solo los 15 de 999.99 son errores.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Notebook: [`starter/ejercicio2_relaciones.ipynb`](./starter/ejercicio2_relaciones.ipynb), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. La última celda imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la tendencia (la tarifa sube casi en línea recta con la distancia, correlación **0.96** sin errores), aparece una **línea horizontal en 70 dólares**: viajes de distancias muy distintas que cobran exactamente lo mismo. Son los 302 viajes con tarifa fija entre JFK y Manhattan, que no siguen la regla del taxímetro. El modelo necesita saber cuáles son, porque la distancia no los explica.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Con tarjeta, la propina promedio es **4.73** dólares; en efectivo es **0.00**, y los **888** viajes en efectivo tienen propina 0. Eso no describe a los pasajeros (que en efectivo seguro dejan propina), sino al **sistema que registra**: la propina en efectivo se entrega en mano y nunca entra al registro. Si un modelo usara la propina, aprendería un artefacto del sistema.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración (correlación **0.956** con la tarifa) solo se conoce cuando el viaje termina, y el modelo estima la tarifa cuando alguien pide el taxi. Usarla sería **fuga de información**: el modelo se vería excelente en el entrenamiento y no tendría el dato al momento de predecir. La distancia estimada sí está disponible antes del viaje.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar. Copia la plantilla [`plantilla-eda/explorar_proyecto.ipynb`](./plantilla-eda/explorar_proyecto.ipynb) a `psets/proyecto/modelado/`: el notebook es tu entregable, con el código, lo que mostró y lo que concluyes en el mismo lugar.

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3: filas, columnas, faltantes por columna, la distribución del objetivo y cuántos atípicos marca el IQR.

### Paso 2: relaciones (10 minutos)

Completa el paso 4: la relación entre tu objetivo y la variable que esperas que más influya.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

En el paso 5 del notebook, para cada hallazgo: una celda de código con el número o el gráfico que lo confirma, y debajo, en la celda de texto, qué encontraste y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar). Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de exploración en tu fork:**

`laboratorios/semana-08-sesion-02/plantilla-eda/explorar_proyecto.py`, la exploración en script que corre en Docker (la salida queda en `data-contenedor/proyecto/salida.txt`; las figuras, en `data-contenedor/proyecto/figuras/`). El dataset es `ssd_mc1.csv` (176.902 SSD C1, 5,94 % de fallas), que hoy se regenera con `psets/proyecto/ingesta/preparar_datos.py`. Los tres hallazgos: `r_9` (horas de uso) delata la clase porque todos los sanos se midieron el mismo día, así que no entra al modelo; `r_5` (sectores reasignados) es 0 en el 75 % de los discos, pero la tasa de falla sube de 4.6 % con `r_5` = 0 a 42.3 % con más de 100, así que se transforma con logaritmo; y hay 5 columnas 100 % vacías y 8 con un solo valor, que se descartan.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Creándolas como columnas nuevas a partir de las que sí existen. La **tarifa fija**: una columna `es_tarifa_fija` que vale 1 cuando el viaje va de JFK a una zona de Manhattan o al revés, calculada desde `zona_origen` y `zona_destino`. La **hora del día**: se extrae de `fecha_hora_inicio` y, como es cíclica (las 23 están cerca de las 0), se codifica con seno y coseno; además se puede marcar si es hora pico o fin de semana. Así el modelo recibe la regla de negocio que explica la línea de 70 dólares, en vez de tener que adivinarla.
