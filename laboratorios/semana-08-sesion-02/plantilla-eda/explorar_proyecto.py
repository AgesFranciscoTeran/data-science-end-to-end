import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# Dataset que arma preparar_datos.py con los datos abiertos de Alibaba: una
# fila por SSD del modelo C1 (el MC1 del PSet 2).
datos = pd.read_csv("ssd_mc1.csv", parse_dates=["fecha_smart"])

# 1 si el SSD fallo, 0 si nunca fallo.
OBJETIVO = "failure"

# Atributo que esperamos que mas influya: SMART 5, sectores reasignados.
# Un disco que empieza a reasignar sectores esta perdiendo celdas.
VARIABLE = "r_5"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes.value_counts())
faltantes = datos.isna().mean()
print("Columnas con faltantes (proporcion):")
print(faltantes[faltantes > 0].round(3))
print(datos[["r_5", "r_9", "r_12", "r_174", "r_187", "r_199", "n_wearout"]].describe().round(2))

# Paso 2: distribucion de la variable objetivo.
print(datos[OBJETIVO].value_counts())
print(f"Proporcion de fallas: {datos[OBJETIVO].mean():.2%}")

plt.figure()
datos[OBJETIVO].value_counts().sort_index().plot.bar()
plt.title("SSD C1 sanos (0) y fallados (1)")
plt.xticks(rotation=0)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos con la regla del IQR. Sobre un objetivo 0/1 el IQR no
# tiene sentido, asi que se aplica a la variable principal.
q1 = datos[VARIABLE].quantile(0.25)
q3 = datos[VARIABLE].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[VARIABLE] < q1 - 1.5 * iqr) | (datos[VARIABLE] > q3 + 1.5 * iqr)]
print(f"{VARIABLE}: Q1={q1}, Q3={q3}, IQR={iqr}")
print(f"Atipicos de {VARIABLE} segun IQR: {len(fuera)} ({len(fuera) / len(datos):.2%})")
print(f"Tasa de falla dentro de los atipicos: {fuera[OBJETIVO].mean():.2%} "
      f"vs todos: {datos[OBJETIVO].mean():.2%}")
print(fuera[["disk_id", "app", VARIABLE, OBJETIVO]].head(10))

# Paso 4: relacion entre r_5 y el objetivo. r_5 es casi siempre 0, asi que en
# vez de dispersion se agrupa en rangos y se compara la tasa de falla.
rangos = pd.cut(datos[VARIABLE], bins=[-1, 0, 10, 100, float("inf")],
                labels=["0", "1-10", "11-100", ">100"])
por_rango = datos.groupby(rangos, observed=True)[OBJETIVO].agg(["mean", "count"])
print(por_rango)

plt.figure()
plt.bar(por_rango.index.astype(str), por_rango["mean"])
plt.xlabel(f"{VARIABLE} (sectores reasignados)")
plt.ylabel("Proporcion de SSD que fallaron")
plt.title("Tasa de falla segun sectores reasignados")
plt.savefig("figuras/r5_vs_objetivo.png")
plt.close()

# Paso 5: consultas para los hallazgos.

# Hallazgo 1: la variable mas correlacionada es r_9 (horas encendido), pero
# los sanos se miden el 2019-12-31 y los fallados el dia de la falla: los
# sanos llevan mas tiempo encendidos solo porque se midieron despues.
numericas = datos.select_dtypes("number").drop(columns=["disk_id", OBJETIVO])
correlaciones = numericas.loc[:, numericas.nunique() > 1].corrwith(datos[OBJETIVO])
print(correlaciones.reindex(correlaciones.abs().sort_values(ascending=False).index).head(10).round(3))
print(datos.groupby(OBJETIVO)["r_9"].median())
print(datos.loc[datos[OBJETIVO] == 1, "fecha_smart"].dt.year.value_counts().sort_index())

plt.figure()
for valor, grupo in datos.groupby(OBJETIVO):
    plt.hist(grupo["r_9"] / 24, bins=50, alpha=0.5, density=True, label=f"failure={valor}")
plt.xlabel("Dias encendido (r_9 / 24)")
plt.legend()
plt.title("Edad del disco al momento de la medicion")
plt.savefig("figuras/r9_por_clase.png")
plt.close()

# Hallazgo 2: los contadores de error separan las clases mejor que su valor
# promedio: lo que importa es si dejan de ser 0.
errores = ["r_5", "r_187", "r_197", "r_199"]
print((datos[errores] > 0).groupby(datos[OBJETIVO]).mean().round(3))

# Hallazgo 3: columnas sin informacion y la aplicacion del disco.
vacias = faltantes[faltantes == 1].index.tolist()
constantes = [c for c in numericas.columns if numericas[c].nunique() == 1]
print(f"Columnas 100 % vacias: {vacias}")
print(f"Columnas con un solo valor: {constantes}")
print(datos.groupby("app")[OBJETIVO].agg(["mean", "count"]).sort_values("mean").round(3))
