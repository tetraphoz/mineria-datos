import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    datos = pd.read_csv("csv/accidentes_viales_mty.csv")
except Exception as e:
    print(f"Error al leer archivo: {e}")
    sys.exit(1)
else:
    print("Archivo cargado correctamente :)")

# 2. Preparar los días en orden correcto
dias_orden = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
datos["Dia"] = pd.Categorical(datos["Dia"], categories=dias_orden, ordered=True)

accidentes_por_dia = datos["Dia"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
accidentes_por_dia.plot(kind="bar", color="tomato")
plt.title("Accidentes por día de la semana")
plt.ylabel("Cantidad de accidentes")
plt.xlabel("Día")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("accidentes_por_dia.png")

chi2, p = stats.chisquare(accidentes_por_dia)
print(f"\n¿Los días son diferentes? p = {p:.4f}")
if p < 0.05:
    print("RESULTADO: Hay días con más accidentes que otros")
else:
    print("RESULTADO: No hay diferencia importante entre días")

plt.figure(figsize=(12, 6))
sns.boxplot(x="Dia", y="Hora_num", data=datos, palette="pastel")
plt.title("Horario de accidentes por día")
plt.ylabel("Hora del día (0-24)")
plt.xlabel("Día de la semana")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("horario_de_accidentes.png")

tabla_tipos = pd.crosstab(datos["Dia"], datos["Tipo_de_accidente"])

plt.figure(figsize=(12, 6))
sns.heatmap(tabla_tipos, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5)
plt.title("Tipos de accidente por día")
plt.ylabel("Día de la semana")
plt.xlabel("Tipo de accidente")
plt.tight_layout()
plt.savefig("tipos_accidentes_semana.png")

# Ver si los tipos cambian por día
chi2, p, _, _ = stats.chi2_contingency(tabla_tipos)
print(f"\n¿Cambian los tipos de accidente por día? p = {p:.4f}")
if p < 0.05:
    print("RESULTADO: Sí hay diferencias en los tipos de accidente según el día")
else:
    print("RESULTADO: No hay diferencia en los tipos de accidente entre días")

datos["Tipo_dia"] = datos["Dia"].apply(
    lambda x: "Fin de semana" if x in ["Sábado", "Domingo"] else "Día laboral"
)

# Comparar cantidad
plt.figure(figsize=(6, 4))
datos["Tipo_dia"].value_counts().plot(kind="bar", color=["skyblue", "lightgreen"])
plt.title("Accidentes: Días laborales vs Fin de semana")
plt.ylabel("Cantidad de accidentes")
plt.xticks(rotation=0)
plt.savefig("accidentes_dias_laborales.png")
