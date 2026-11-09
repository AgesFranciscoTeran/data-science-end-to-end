# Despliegue: el modelo de fallas de SSD se sirve por lotes

**Decisión:** el modelo corre **por lotes, una vez al día**, cuando termina la recolección diaria del SMART de todos los discos. Le asigna a cada SSD C1 una probabilidad de estar fallando y deja una tabla ordenada de mayor a menor riesgo. El equipo de operaciones del centro de datos usa esa tabla para decidir qué discos reemplazar o vaciar primero.

## Qué decisión habilita y cuánto puede esperar

- **Decisión:** a qué SSD se les migran los datos y cuáles se reemplazan de forma preventiva, antes de que la falla deje un servidor sin disco. El modelo responde "¿este SSD se parece a los que fallaron?" a partir de sus contadores SMART (`failure`, con average precision de prueba 0.50 para el bosque aleatorio de la Semana 10).
- **Quién la usa:** técnicos de operaciones, que trabajan una lista por turno. No hay un usuario esperando la respuesta en una pantalla.
- **Cuánto puede esperar:** **24 horas.** Los contadores que delatan una falla (sectores reasignados `r_5`, errores no corregibles `r_187`, sectores pendientes `r_197`, errores CRC `r_199`) crecen durante días o semanas antes de que el disco muera. Migrar los datos y cambiar el disco toma horas. Recibir la lista a las 2:00 en lugar de en 50 ms no cambia qué disco se cambia ni cuándo.

## Cuántos casos hay y si se conocen de antemano

- Un caso es **un SSD en un día**. En los datos abiertos de Alibaba hay **unos 168.000 SSD C1** activos el 2019-12-31. El dataset del proyecto, `ssd_mc1.csv`, tiene 176.900 SSD con alguna lectura SMART: los sanos de ese día más los 10.508 que fallaron en 2018 y 2019.
- **Se conocen todos de antemano:** son el inventario de discos instalados. Un disco nuevo entra al lote del día siguiente, y ninguna petición llega desde afuera.
- Puntuar los 176.900 SSD con el bosque de 200 árboles tomó 0,8 segundos en una laptop, así que una corrida diaria es barata.

## Qué datos necesitan las features y cuándo están disponibles

- Las 13 features numéricas son contadores SMART crudos del día (`r_5`, `r_183`, `r_184`, `r_187`, `r_195`, `r_197`, `r_199`, `r_program`, `r_erase`, `n_wearout`, `r_12`, `r_174`) y dos derivadas de ellos: `n_contadores_error` y `log_r_5`. La categórica es `app`, la aplicación del servidor, que viene del inventario.
- El SMART se recolecta **una vez al día por disco**, y en Alibaba llega como un archivo diario. Antes de que termine la recolección, la lectura de hoy simplemente no existe; con la de ayer, el lote ya está calculado.
- `app` sale del inventario de ubicación, que cambia poco y se puede leer en el mismo lote.
- La fecha de la lectura y `r_9` (horas de uso) no entran al modelo. En este dataset delatan la clase, porque todos los sanos se midieron el mismo día. Las pruebas de `modelado/test_pipeline_proyecto.py` comprueban que no llegan al modelo.

## Qué se pierde al descartar el servicio en línea

- **Reaccionar el mismo día a un salto brusco**, por ejemplo un disco que acumula cientos de errores CRC en una hora. Ese disco aparece en la lista recién al día siguiente.
- **Consultar un disco a pedido:** un técnico que quiere el riesgo de un disco ahora mismo lee la última fila de la tabla, que puede tener hasta un día de antigüedad.
- Ninguna de las dos cosas compensa el costo de un servicio en línea mientras el SMART llegue una vez al día y la respuesta sea un ticket para una persona. Valdría la pena si la telemetría llegara cada pocos minutos y la acción fuera automática, por ejemplo sacar el disco de un RAID.
