# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

### Cómo correr los scripts

Localmente, cada uno desde su carpeta:

```
cd starter && python3 ejercicio1_features_tiempo.py
cd starter && python3 ejercicio2_features_zona.py
cd plantilla-features && python3 features_proyecto.py
```

O con Docker, igual que en la Semana 8 (cada servicio deja su `salida.txt` en `data-contenedor/<servicio>`). El servicio `proyecto` lee el `ssd_mc1.csv` que dejó la Semana 8 en `../semana-08-sesion-02/data-contenedor/proyecto`:

```
docker compose up --build
```

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio traduce un mecanismo conocido del dominio (cómo se cobra, cómo se comporta un cliente, cómo se degrada un equipo) a una columna, así que se puede explicar, se sabe por qué debería seguir funcionando con datos nuevos y se puede calcular en el momento de predecir. Una feature que solo mejora la métrica puede ser una coincidencia del conjunto de entrenamiento o, peor, una fuga de información (usar algo que se conoce solo después del evento), y deja de funcionar en producción.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

_(pendiente: revisar con el artículo; no se pudo abrir el enlace al completar el laboratorio)_

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Creando columnas derivadas que codifiquen la regla: `es_tarifa_fija` (1 si el viaje va entre JFK y una zona de Manhattan, 0 si no), y a partir de `fecha_hora_inicio` la hora, el día de la semana, `es_fin_de_semana` y `es_hora_pico`. Así el modelo no tiene que "descubrir" la regla a partir de combinaciones de zonas y horas: se la damos explícita.

## Ejercicio 1: features de fecha y hora

Archivo: [`starter/ejercicio1_features_tiempo.py`](./starter/ejercicio1_features_tiempo.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. El script ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-01/starter
python3 ejercicio1_features_tiempo.py
```

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número quedan a 23 unidades, lo más lejos posible. En el reloj están a 1 hora. Con `hora_sin` y `hora_cos` la hora queda en un círculo: la distancia entre 23 y 0 es 0.261, mientras que entre 12 y 0 es 2.000 (lados opuestos del reloj).

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Más alta. La tarifa del taxímetro cobra por distancia y también por tiempo: en hora pico el tráfico alarga el viaje para la misma distancia, así que cada kilómetro acumula más minutos cobrados. La salida lo confirma: la mediana de tarifa por km es 3.52 en hora pico contra 3.22 fuera de ella (~9 % más).

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

El tráfico. La duración es alta en parte porque hay tráfico, y el tráfico depende de la hora. En hora pico los viajes (sin JFK) duran 23.6 min en promedio contra 17.5 fuera de ella, con casi la misma distancia (6.46 vs 6.25 km). `es_hora_pico` anticipa ese tiempo extra sin usar una columna que solo se conoce al terminar el viaje.

**Lo que confirmó la salida, TODO por TODO:**

- TODO 1 y 2: `hora`, `dia_semana` (0 = lunes) y `es_fin_de_semana` (sábado y domingo) salen de `fecha_hora_inicio`; el viaje del domingo 2026-09-06 a las 13:46 queda con hora 13, día 6 y fin de semana 1.
- TODO 3: tarifa por km, mediana, sin tarifas fijas ni distancia 0: **3.22 fuera de hora pico, 3.52 en hora pico**.
- TODO 4: en el Paso 5, **23 y 0 quedan a 0.26** como ciclo (23 como número), igual que **11 y 12, también a 0.26** (1 como número): con seno y coseno, dos horas consecutivas quedan a la misma distancia, crucen o no la medianoche.

En [`starter/mi_primer_feature_eng.py`](./starter/mi_primer_feature_eng.py) (mi versión antes de que se publicara este script) está la misma comparación con promedios, sin viajes de JFK:

| es_hora_pico | tarifa | distancia_km | tarifa_por_km | duracion_min | viajes |
|---|---|---|---|---|---|
| 0 | 20.06 | 6.25 | 3.93 | 17.49 | 1795 |
| 1 | 22.67 | 6.46 | 4.14 | 23.57 | 786 |

En hora pico la tarifa sube 2.61 en promedio con solo 0.21 km más de distancia. El efecto es pequeño frente a la distancia, pero existe y es información disponible al pedir el taxi.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Archivo: [`starter/ejercicio2_features_zona.py`](./starter/ejercicio2_features_zona.py), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_features_zona.py
```

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

300 viajes de 2975, todos con tarifa mínima y máxima de 70.0 (desviación 0), aunque su distancia va de 0 a 45.43 km. Para esos viajes la tarifa no depende de la distancia. Un modelo lineal con solo `distancia_km` los predice mal; con `es_tarifa_fija` puede aprender un salto de nivel para ese grupo, pero sigue sumándole la pendiente de la distancia. Para capturarlo del todo necesita también la interacción (`distancia_km × (1 − es_tarifa_fija)`), o un modelo de árboles que separe primero por esta columna.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas (una por zona); la ordinal crea 1 (Brooklyn=0, Harlem=1, JFK Airport=2, Lower Manhattan=3, Midtown=4, Queens=5, Upper East Side=6, Upper West Side=7). Para regresión lineal usaría one-hot (quitando una columna de referencia si hay intercepto). La ordinal le dice al modelo que las zonas tienen orden y distancias iguales entre sí: que Upper West Side "vale" 7 veces Harlem − Brooklyn, y que el efecto de la zona crece en línea recta con un número que solo sale del orden alfabético. Eso es falso.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

`duracion_min` (depende del tráfico, se sabe al terminar), `metodo_pago` (se elige al pagar), `propina` (se da al final y depende de la tarifa) y, obviamente, `tarifa`, que es el objetivo. Si se entrenara con ellas habría fuga de datos: la duración tiene correlación 0.96 con la tarifa, así que el modelo se vería excelente en validación pero en producción esas columnas no existirían al momento de predecir y el modelo no se podría usar o fallaría.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

El modelo predice si un SSD del modelo C1 va a fallar (`failure`), usando la lectura SMART de un día. En ese momento existe: los atributos SMART del día (contadores de error `r_5`, `r_187`, `r_197`, `r_199`, desgaste `n_wearout`, horas encendido `r_9`, ciclos `r_12`, etc.) y la aplicación del disco (`app`). No existe: si el disco va a fallar ni la fecha de falla. Ojo con `fecha_smart` y `r_9`: en este dataset los sanos se miden el 2019-12-31 y los fallados el día de la falla (Hallazgo 1 de la Semana 8), así que la fecha y la edad delatan la clase por cómo se armó el dataset, no por el disco; no las uso como feature.

### Paso 2: construye las 2 features (15 minutos)

```python
# Feature 1: cuantos contadores de error dejaron de ser 0.
ERRORES = ["r_5", "r_187", "r_197", "r_199"]
datos["n_contadores_error"] = (datos[ERRORES] > 0).sum(axis=1)

# Feature 2: r_5 en escala logaritmica.
datos["log_r_5"] = np.log1p(datos["r_5"])
```

Antes se quitan 2 filas de fallas sin ninguna lectura SMART.

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`n_contadores_error` sale del Hallazgo 2 de la Semana 8: lo que separa a las clases no es el valor promedio de los contadores de error, sino que dejen de ser 0 (r_187 > 0 en 11.6 % de los fallados contra 0.1 % de los sanos). En el dominio, cada contador mide un mecanismo de degradación distinto (sectores reasignados, errores incorregibles, lecturas pendientes, errores de interfaz), y que varios se activen a la vez es señal de que el disco se está muriendo. Evidencia, tasa de falla por valor:

| n_contadores_error | tasa de falla | discos |
|---|---|---|
| 0 | 4.6 % | 131916 |
| 1 | 7.5 % | 43035 |
| 2 | 63.4 % | 1912 |
| 3 | 100 % | 37 |

Correlación con `failure`: 0.153, el doble que la de cualquier atributo SMART crudo (la mayor era r_5 con 0.076, sin contar r_9). Existe al predecir porque se calcula solo con la lectura SMART del día. Es numérica discreta (0 a 4) con orden real (más contadores activos = peor), así que se puede usar tal cual en un modelo lineal o de árboles.

**Feature 2:**

`log_r_5` sale del Paso 4 de la Semana 8: r_5 está casi siempre en 0 (Q1 = Q3 = 0) y llega a 9071, así que en escala lineal unos pocos discos con miles de sectores dominan la relación. Pero la tasa de falla salta entre 0 sectores (4.6 %), 1-10 (6.2 %) y 11-100 (30.6 %), y luego crece poco (>100: 42.3 %): lo que importa es el orden de magnitud. Con `log1p` (que deja el 0 en 0), la correlación con `failure` sube de 0.076 a 0.207. Existe al predecir porque es la misma lectura SMART del día transformada. Es numérica; la transformación sirve sobre todo para un modelo lineal (regresión logística), a un árbol le da igual.

Si uso `app` (categórica, 9 valores), para regresión logística la codificaría con one-hot: no hay un orden natural entre aplicaciones. Para un modelo de árboles, ordinal también funcionaría.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento, y después se aplica esa misma media y desviación a validación y prueba. Si se calculan con todos, información del conjunto de prueba se filtra al entrenamiento (fuga de datos): el modelo "ya vio" algo de la distribución de los datos con los que se evalúa y la métrica sale optimista. Con 3000 viajes el cambio numérico en la media probablemente sea pequeño, pero el procedimiento es incorrecto, y en producción tampoco tendrías los datos futuros para calcularla.
