# Despliegue: el modelo de fallas de SSD se sirve por lotes

**Decisión:** el modelo corre **por lotes, una vez al día**, después de que llega el SMART del día completo. Deja una tabla con la probabilidad de falla a 30 días de cada SSD MC1, y el equipo de operaciones de almacenamiento la usa para armar la lista de discos a reemplazar.

## Qué decisión habilita y cuánto puede esperar

- **Decisión:** qué SSD se reemplazan de forma preventiva, y en cuáles se migran antes los datos a otro disco. El modelo responde "¿este SSD fallará en los próximos 30 días?" (`TARGET_30D`).
- **Quién la usa:** el equipo de operaciones del centro de datos. No hay un cliente esperando una respuesta: la salida es una lista de trabajo que se planifica en turnos.
- **Cuánto puede esperar:** **24 horas.** La predicción vale para una ventana de 1 a 30 días, y migrar datos o programar el cambio de un disco toma horas o días. Una estimación que llega en 50 ms o en 6 horas lleva a la misma decisión. La etiqueta descarta a propósito las fallas del mismo día (`SAME_DAY_FAILURE`), así que el modelo nunca promete avisar con minutos de anticipación.

## Cuántos casos hay y si se conocen de antemano

- Un caso es **un SSD en un día**. En el snapshot público del 2019-12-31 hay unos **168.000 SSD** del modelo C1, el equivalente a MC1, y en el dataset completo son unos 115 millones de observaciones MC1 en dos años.
- **Se conocen todos de antemano:** son el inventario de discos instalados. No llegan casos nuevos desde afuera durante el día. Un disco nuevo entra al lote del día siguiente.
- Puntuar 168.000 filas una vez al día con un gradient boosting toma minutos en una sola máquina. El costo está en calcular las ventanas, y esa parte ya corre por lotes en el pipeline del PSet 2 (Kestra, dbt y Spark).

## Qué datos necesitan las features y cuándo están disponibles

- Las features son el valor del día de los 22 atributos SMART (raw `R_` y normalizado `N_`) y, para cada uno, la **media, el conteo y el cambio de media en ventanas de 7, 14 y 30 días**. La ventana anterior de 30 días mira 59 días hacia atrás.
- El SMART se publica **un archivo por día**. Una ventana de 7 días solo está completa cuando cerró el día de hoy. Antes de eso, la media de hoy se calcularía con medio día de datos.
- Las ventanas necesitan el **historial de 60 días de cada disco**, que vive en el warehouse (Snowflake). Calcularlo en el momento de cada petición obligaría a guardar el estado de cada disco en el servicio, y sin una fuente de SMART en tiempo real no habría nada nuevo que calcular.
- La etiqueta mira 30 días hacia el futuro y nunca llega al modelo. Las pruebas de `modelado/test_pipeline_proyecto.py` lo comprueban.

## Qué se pierde al descartar el servicio en línea

- **Reaccionar a una degradación brusca el mismo día:** un salto en los sectores reasignados (`R_5`) o en los errores CRC (`R_199`) a las 9:00 aparece en la lista recién al día siguiente.
- **Consultar un disco a pedido:** si un técnico quiere saber el riesgo de un disco ahora mismo, lee la última fila de la tabla. Esa fila puede tener hasta un día de antigüedad.
- Las dos cosas solo valdrían la pena si la telemetría llegara cada pocos minutos y la acción fuera automática, por ejemplo sacar el disco de un RAID. Este proyecto no cumple ninguna de las dos condiciones (ver `06_batch_vs_streaming.md` del PSet 2).
