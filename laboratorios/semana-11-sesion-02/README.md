# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Es cuando un modelo tiene una exactitud alta sin servir para nada: si la clase que importa es rara, predecir siempre la clase mayoritaria ya acierta casi todo. En vez de la exactitud recomienda métricas que miren la clase rara: **precisión** (de lo que marqué, cuánto era cierto), **recall o sensibilidad** (de lo que existía, cuánto encontré), su combinación (**F1**), la **matriz de confusión** completa, y curvas como la de precisión-recall, que no dependen de un solo umbral.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

Cuando una métrica se vuelve la meta, la gente (o el modelo) la optimiza por caminos que ya no producen lo que la métrica quería medir, como un sistema que mejora su número sin mejorar el resultado real. Con un modelo pasa lo mismo: si se elige solo por la exactitud, gana el que predice siempre "no es fraude" (95 % en el Ejercicio 2) aunque no detecte ningún fraude. Una sola métrica, sin mirar qué errores comete el modelo ni cuánto cuestan, puede premiar al modelo equivocado.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Cambiaría casi tanto como la diferencia entre modelos, y sí se vio en el registro de la sesión pasada: el mismo gradient boosting dio **1.2754** con la semilla de división 42 y **1.1867** con la semilla 7. Hoy la validación cruzada lo confirma: con 5 pliegues al azar, sus errores van de 1.16 a 1.28 (desviación 0.036). Elegir por un solo número de 595 viajes puede depender más de la suerte de la división que del modelo.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Parecidos, pero no iguales: cada pliegue valida con 595 viajes distintos. Esperaba que variaran unos pocos centavos alrededor de 1.25, como la diferencia entre las semillas 42 y 7 de la sesión pasada (0.09). Salieron entre **1.16 y 1.28**, con desviación **0.036**.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

**Mayor.** Con pliegues al azar, el modelo entrena con viajes de todas las fechas, incluidos algunos posteriores a los que valida; en la validación temporal solo ve el pasado, y el futuro puede traer patrones que no vio (otros días, otro tráfico). Además, los primeros pliegues entrenan con pocos datos. Es más pesimista, pero más parecida a lo que pasa en producción. Salió **1.3099** contra 1.2288 al azar, con el doble de desviación (0.069).

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El que deja afuera **JFK Airport**: sus viajes son largos y muchos tienen la tarifa fija de 70 dólares, así que no se parecen a nada de lo que el modelo ve en las zonas de Manhattan. Así fue: el pliegue 5 (Brooklyn y JFK Airport) tuvo un MAE de **6.43**, mientras los otros cuatro quedaron entre 1.19 y 1.31.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El **porcentaje de viajes subestimados en más de 20 %**, porque cuenta exactamente los casos que generan un reclamo. El MAE y el RMSE mezclan sobreestimar y subestimar, y no dicen cuántos viajes cruzan el límite del 20 %: un MAE de 1.28 dólares puede esconder unos pocos viajes muy subestimados. Con el gradient boosting, el **4.4 %** de los viajes de prueba (26 de 595) sale más de 20 % por encima de lo estimado.

**Lo que confirmó la tabla, esquema por esquema:**

Al azar, gradient boosting: promedio **1.2288**, desviación **0.0361**, peor pliegue **1.28**. Al azar, bosque: **1.2880**, **0.0291**, peor **1.34**. Temporal, gradient boosting: **1.3099**, **0.0693**, peor **1.40**. Por zona, gradient boosting: **2.2996**, **2.0648**, peor **6.43** (el pliegue de Brooklyn y JFK Airport). Con la división de prueba: MAE **1.2754**, RMSE **1.9213** y 4.4 % de viajes subestimados en más de 20 %.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy **0.9508** y recall **0**: acierta el 95 % porque el 95 % de los pagos son legítimos, pero no detecta ni uno de los 295 fraudes.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy mejora apenas **2 puntos** (de 0.9508 a 0.9712); el recall pasa de **0 a 0.461**: detecta casi la mitad de los fraudes, y con precisión de 0.905 (9 de cada 10 alarmas son fraudes reales). El **recall** describe mejor lo que gana la empresa, porque cada fraude detectado es un contracargo de 40 dólares que no se paga; la accuracy casi no se mueve porque los fraudes son pocos.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con el umbral 0.5 deja pasar **159** fraudes de 295 (detecta 136) y da 14 falsas alarmas. El paso 5 eligió el umbral **0.10**: deja pasar **122** fraudes y da **222** falsas alarmas, con un costo de 5.546 dólares frente a 6.402 con el umbral 0.5. Conviene aceptar muchas más falsas alarmas (3 dólares cada una) para detectar más fraudes (40 dólares cada uno).

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

**Sube**, de 0.10 a **0.50**. Con 20 dólares por falsa alarma, las 222 falsas alarmas del umbral 0.10 cuestan 4.440 dólares, y el costo total pasa a 9.320; con el umbral 0.50 (159 fraudes sin detectar y 14 falsas alarmas) cuesta 6.640, el menor de la lista. Cuando las falsas alarmas son caras, conviene que el modelo marque solo los pagos de los que está muy seguro.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

**Estratificado de 5 pliegues** (`StratifiedKFold`, mezclado, semilla 42). Solo el 5,94 % de los SSD falla, así que cada pliegue tiene que tener la misma proporción de fallas: con estratificación, cada uno tiene 2101 o 2102. No corresponde validar por grupo, porque cada SSD aparece una sola vez (una fila por disco). Tampoco por fecha: todos los sanos se midieron el 2019-12-31, así que un corte temporal dejaría pliegues sin sanos. Notebook y tablas: `psets/proyecto/modelado/validacion_proyecto.ipynb`, `validacion_proyecto.csv` y `costo_proyecto.csv`.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

No cambió. Con 10.508 fallas, el esquema al azar ya reparte casi la misma proporción en cada pliegue (entre 2064 y 2151 fallas), así que los promedios casi coinciden. Bosque: estratificado **0.5134 ± 0.0083**, al azar 0.5151 ± 0.0060. Red neuronal: estratificado **0.3553 ± 0.0184**, al azar 0.3570 ± 0.0278. Lo que sí cambia es la variación de la red: al azar sus pliegues van de 0.326 a 0.395, porque cada pliegue tiene una cantidad distinta de fallas. Con menos fallas, la estratificación importaría mucho más.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

Sí, mucho mayor. La diferencia es de **0.158** de average precision (0.5134 frente a 0.3553), y la desviación entre pliegues es de 0.008 para el bosque y 0.018 para la red: unas 9 veces la desviación más grande. Además, el peor pliegue del bosque (0.5001) supera al mejor de la red (0.3825). Sigo eligiendo el bosque.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Elegí el **costo de los errores en dólares**. Una falla sin aviso cuesta unos 500 dólares (reemplazo de urgencia, reconstrucción del arreglo y riesgo para los datos y el SLA), y una falsa alarma unos 50 (revisar o migrar un disco sano), así que un falso negativo es 10 veces más caro. Lo calculé con las probabilidades de validación cruzada estratificada, para varios umbrales. Sin modelo, las 10.508 fallas llegan sin aviso y cuestan **5.254.000** dólares. Con el bosque, el umbral más barato es **0.10**: anticipa 7196 fallas (recall 0.68) a cambio de 19.477 falsas alarmas, y cuesta 2.629.850, **50 % menos**. La red, en su mejor umbral (también 0.10), solo ahorra el 35 %. La average precision decía cuál modelo ordena mejor los discos. El costo dice algo que ella no decía: **dónde cortar**. Con el umbral de 0.5 que se usa por defecto, el bosque solo anticipa el 24 % de las fallas y ahorra el 23 %; bajar el umbral a 0.10 duplica el ahorro, aunque multiplica por 48 las falsas alarmas.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La **tarifa** sí se podría calcular de noche, pero solo para casos conocidos de antemano, como los viajes reservados para el día siguiente; las combinaciones posibles de origen, destino, hora y distancia son demasiadas para precalcularlas todas, así que un viaje que se pide en el momento necesita una respuesta en línea en menos de un segundo, mientras el pasajero mira la pantalla. El **fraude** tiene que responder en el momento del cobro, en milisegundos (unos 100 a 200 ms), si se quiere detener el pago antes de que pase; si basta con revisarlo después, se puede puntuar por lotes cada noche con los pagos del día, como en la Semana 12.
