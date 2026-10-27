# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina las predicciones de varios modelos (normalmente árboles) en una sola, que suele equivocarse menos que cualquiera de ellos. En **bagging** (como el bosque aleatorio) los modelos se entrenan en paralelo e independientes, cada uno con una muestra con reemplazo de los datos (y, en el bosque, un subconjunto de columnas en cada corte), y se promedian sus predicciones (o se vota): reduce la varianza. En **boosting** (como gradient boosting) los modelos se entrenan en secuencia, cada uno corrigiendo los errores que dejaron los anteriores, y se suman con un peso: reduce sobre todo el sesgo.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque los errores de cada árbol son en parte distintos: cada uno vio otra muestra y otras columnas, así que se equivoca de forma diferente, y al promediar los errores en direcciones opuestas se cancelan. Lo que tienen en común (la señal) se queda; lo que es ruido propio de cada árbol se diluye. Cuanto menos correlacionados están los árboles, más baja la varianza del promedio.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: memorizó detalles y ruido del entrenamiento que no se repiten en datos nuevos. La brecha entre el error en prueba y en entrenamiento mide cuánto memorizó en vez de generalizar.

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Esperaba un error de más de 10 dólares, del orden de la dispersión de las tarifas, y fue **12.53 en prueba**. Sirve como piso: cualquier modelo que use columnas tiene que equivocarse bastante menos que eso para valer la pena. Si un modelo complejo apenas le gana a la mediana, las columnas no están aportando (o algo está mal en la preparación).

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento, casi 0: con hojas de un viaje, el árbol reproduce cada tarifa (salió **0.0029**). En prueba, bastante más alto, porque memorizó también el ruido de cada viaje: salió **1.7017**, el peor de los tres modelos con columnas.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, el error en entrenamiento sube pero la brecha se achica mucho, y el de prueba baja (el árbol ya no puede memorizar viaje por viaje). Con 200 árboles promediados esperaba que la brecha también se achicara frente al árbol sin límite y que el error en prueba fuera el más bajo. Lo primero se cumplió (0.83 contra 1.70); lo segundo no: el bosque (1.3016) quedó apenas peor que el árbol de profundidad 8 (1.2728).

**Lo que confirmó la tabla, TODO por TODO:**

| modelo | entrenamiento | prueba | brecha | segundos |
|---|---|---|---|---|
| línea base (mediana) | 13.1998 | 12.5308 | −0.6690 | 0.02 |
| árbol sin límite | 0.0029 | 1.7017 | 1.6988 | 0.03 |
| árbol `max_depth=8` | 1.0098 | 1.2728 | 0.2630 | 0.02 |
| bosque aleatorio (200) | 0.4739 | 1.3016 | 0.8276 | 3.6 |

- TODO 1: `DummyRegressor(strategy="median")`. Sin el `strategy`, el `DummyRegressor` por defecto predice la media (13.59 / 12.92).
- TODO 2: el árbol sin límite memoriza (entrenamiento ≈ 0) y tiene la brecha más grande.
- TODO 3: `max_depth=8` sube el error de entrenamiento a 1.01 pero baja el de prueba a 1.27: la regularización redujo la brecha de 1.70 a 0.26.
- TODO 4: el bosque reduce la brecha del árbol sin límite a la mitad y mejora la prueba de 1.70 a 1.30, pero sus árboles siguen sin límite (cada uno memoriza), por eso su brecha es mayor que la del árbol de profundidad 8. Además tarda ~150 veces más en entrenar.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe **3275 columnas** y hay **2380 filas** de entrenamiento: más coeficientes que datos. Con eso, la regresión tiene infinitas soluciones que pasan casi exactamente por cada viaje de entrenamiento (error 0.90), y la que elige puede tener coeficientes enormes que se cancelan entre sí en entrenamiento pero no en viajes nuevos. Por eso el error en prueba explota: **3 212 929 068.79** con las versiones del curso (scikit-learn 1.5.2, numpy 1.26.4). Con scikit-learn 1.8 / numpy 2.3 en mi laptop la solución numérica es otra y da 22.99 (por eso, fuera del ambiente del curso, el Paso 2 marca `revisa`: su brecha queda en 22, no >100). En ambos casos es peor que la línea base.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento Ridge empeora un poco (**0.8995 → 1.0272**) y en prueba pasa de **3.2 × 10⁹ a 1.3209**. Ridge suma al error una penalización por el tamaño de los coeficientes (`alpha=10`): ya no puede usar coeficientes gigantes para pasar exactamente por cada viaje, así que ajusta un poco peor el entrenamiento a cambio de una solución estable que generaliza. La brecha cae de miles de millones a 0.29.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

**Gradient boosting**, con **1.2754** en prueba (y la menor brecha, 0.2048), seguido de cerca por el árbol de profundidad 8 del Ejercicio 1 (1.2728, que en realidad es un poco mejor; no está en esta tabla). Frente a la línea base (12.5308) se equivoca 11.26 dólares menos en promedio: un **90 % menos** de error.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

Porque la línea base no aprende nada de las filas: predice una constante (la mediana de entrenamiento, 22.53), y su error solo depende de qué tan lejos de esa constante caen las tarifas de cada conjunto. Por azar de la división, las tarifas del 20 % de prueba quedaron un poco más concentradas (desviación estándar 17.3 contra 18.2 en entrenamiento, y 8.9 % de tarifas fijas de 70 contra 10.4 %), así que el mismo número se equivoca menos en prueba. Una brecha negativa pequeña no es una virtud del modelo, es variación de muestreo.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

`DummyClassifier(strategy="prior")`: le asigna a todos los discos la misma probabilidad de falla, la proporción de entrenamiento (5.94 %). La métrica del proyecto es **average precision** (área bajo la curva precisión-recall), porque con 5.94 % de fallas la exactitud engaña ("nunca falla" acierta 94 %). Con la average precision, un modelo que no ordena a los discos saca exactamente la tasa de falla (0.0594) y el AUC queda en 0.5: es el piso de "no saber nada". No usé una regla de dominio como "mañana igual que hoy" porque hay una sola lectura por disco, sin historia. Código en [`proyecto/modelos_proyecto.py`](./proyecto/modelos_proyecto.py), tabla en [`proyecto/resultados_proyecto.csv`](./proyecto/resultados_proyecto.csv):

| modelo | AP entrenamiento | AP prueba | brecha | AUC prueba | segundos |
|---|---|---|---|---|---|
| línea base (proporción) | 0.0594 | 0.0594 | 0.0000 | 0.5000 | 0.6 |
| logística C=1 | 0.2410 | 0.2214 | 0.0195 | 0.7195 | 2.0 |
| logística C=0.01 | 0.2258 | 0.2083 | 0.0175 | 0.7094 | 1.1 |
| bosque sin límite | 0.8337 | 0.4808 | 0.3529 | 0.8408 | 5.0 |
| bosque `min_samples_leaf=2` | 0.7477 | **0.5003** | 0.2474 | **0.8619** | 4.6 |
| bosque `min_samples_leaf=50` | 0.4141 | 0.3799 | 0.0342 | 0.8273 | 3.7 |

(En la brecha, como la métrica es "más es mejor", resto entrenamiento − prueba.)

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Sí, el bosque sin límite: **0.8337 en entrenamiento contra 0.4808 en prueba**, una brecha de 0.35, diez veces la de cualquier otro modelo. Sus árboles llegan a hojas de un disco y memorizan el entrenamiento (no llega a 1.0 porque hay discos con exactamente las mismas lecturas y distinta etiqueta). La regresión logística no se sobreajusta (brecha 0.02): con 22 columnas y 141520 discos es un modelo simple; su problema es el contrario, le falta capacidad (AP 0.22).

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

En el bosque, `min_samples_leaf` (cada hoja debe promediar al menos esa cantidad de discos). Con **50** la brecha cae de 0.35 a **0.03**, pero la prueba empeora de 0.48 a 0.38: regularicé de más y el bosque perdió la capacidad de separar los pocos patrones de falla. Con **2** la brecha baja a 0.25 y la prueba sube a **0.5003**, el mejor resultado (probé también 5, 10 y 20: 0.476, 0.442 y 0.408 en prueba). Una brecha más chica no siempre es mejor: lo que importa es el error en prueba. En la logística, regularizar 100 veces más (C=0.01) no hizo falta: la brecha casi no cambia (0.0195 → 0.0175) y la prueba empeora (0.2214 → 0.2083). Ojo: elegí `min_samples_leaf=2` mirando la prueba, así que el 0.50 es un poco optimista; la próxima vez esa elección se hace con validación cruzada sobre entrenamiento.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El bosque con `min_samples_leaf=2` saca **0.5003** de average precision en prueba contra **0.0594** de la línea base: **8.4 veces** mejor (y 2.3 veces mejor que la logística). En AUC, 0.86 contra 0.5. En la práctica: si se revisan los discos ordenados por el riesgo que da el modelo, las fallas aparecen muy arriba en la lista, en vez de repartidas al azar.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

- **Latencia:** cuánto tarda cada predicción (la app cotiza al pedir el taxi; un ensamblaje responde en milisegundos, un LLM en segundos).
- **Costo por predicción y de entrenamiento:** el ensamblaje y la red corren en una CPU propia; el LLM cobra por token en cada viaje.
- **Consistencia:** el mismo viaje debe dar la misma tarifa; un LLM puede responder distinto cada vez o no devolver un número.
- **Explicabilidad:** poder decir por qué una tarifa es la que es (distancia, tarifa fija, hora pico), frente a reclamos o reguladores.
- **Dependencia y privacidad:** el LLM depende de un servicio externo y recibe los datos de los viajes.
- **Mantenimiento:** cuánto cuesta reentrenar cuando cambian las tarifas, y cuántos datos necesita cada opción (la red suele necesitar más que el ensamblaje).
