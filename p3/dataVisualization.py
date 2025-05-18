import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys


def primera_moda(series):
    """Función auxiliar para obtener la primera moda de una serie"""
    moda = series.mode()
    return moda.iloc[0] if not moda.empty else None


def top_tres_dict(series):
    """Función auxiliar para obtener los 3 valores más comunes en formato dict"""
    return series.value_counts().head(3).to_dict()


try:
    df = pd.read_csv("csv/accidentes_viales_mty.csv")
except Exception as e:
    print(f"Error al leer archivo: {e}")
    sys.exit(1)

print("Archivo cargado correctamente :)")
print("Generado gráficas")

# 1. Distribución de tipos de accidente (Barras horizontales)
accidentes_por_tipo = (
    df.groupby("Tipo_de_accidente")
    .agg(Total=("Folio", "count"), Hora_promedio=("Hora_num", "mean"))
    .sort_values("Total", ascending=False)
)

plt.figure(figsize=(12, 6))
accidentes_por_tipo["Total"].sort_values().plot(kind="barh", color="steelblue")
plt.title("Tipos de accidentes más frecuentes en Monterrey", fontsize=16)
plt.xlabel("Número de accidentes", fontsize=12)
plt.ylabel("Tipo de accidente", fontsize=12)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("distribucion_tipos_accidentes.png", bbox_inches="tight")

# 2. Patrón de accidentes por hora (Gráfica de línea)
accidentes_por_hora = (
    df.groupby("Hora_num")
    .agg(Total=("Folio", "count"), Tipo_mas_comun=("Tipo_de_accidente", primera_moda))
    .sort_index()
)

plt.figure()
sns.lineplot(
    x=accidentes_por_hora.index,
    y="Total",
    data=accidentes_por_hora,
    marker="o",
    color="crimson",
    linewidth=2.5,
)
plt.title("Distribución horaria de accidentes viales", fontsize=16)
plt.xlabel("Hora del día (formato 24h)", fontsize=12)
plt.ylabel("Número de accidentes", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("patron_accidentes_hora.png", bbox_inches="tight")

# 3. Accidentes por día de la semana (Barras con anotaciones)
accidentes_por_dia = (
    df.groupby("Dia")
    .agg(
        Total=("Folio", "count"),
        Hora_moda=("Hora_num", primera_moda),
        Tipo_mas_comun=("Tipo_de_accidente", primera_moda),
    )
    .sort_values("Total", ascending=False)
)

dias_orden = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
accidentes_por_dia = accidentes_por_dia.reindex(dias_orden)

plt.figure()
ax = accidentes_por_dia["Total"].plot(kind="bar", color="teal", alpha=0.8)
plt.title("Accidentes por día de la semana", fontsize=16)
plt.xlabel("Día de la semana", fontsize=12)
plt.ylabel("Número de accidentes", fontsize=12)

# Añadir valores en las barras
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("accidentes_por_dia.png", bbox_inches="tight")

# 4. Top 10 colonias con más accidentes (Mapa de calor)
top_colonias = (
    df.groupby("Nombre_de_asentamiento")
    .agg(
        Total=("Folio", "count"),
        Tipo_mas_comun=("Tipo_de_accidente", primera_moda),
        Hora_moda=("Hora_num", primera_moda),
    )
    .sort_values("Total", ascending=False)
    .head(10)
)

top_colonias_para_heatmap = top_colonias[["Total", "Hora_moda"]].sort_values(
    "Total", ascending=False
)

plt.figure()
sns.heatmap(
    top_colonias_para_heatmap.T,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Valor"},
)
plt.title("Top 10 colonias: Total de accidentes y hora más frecuente", fontsize=16)
plt.xlabel("Colonia", fontsize=12)
plt.ylabel("Métrica", fontsize=12)
plt.tight_layout()
plt.savefig("top_colonias_heatmap.png", bbox_inches="tight")


# Proporción de los 5 tipos más comunes (Pastel)
top5_tipos = accidentes_por_tipo["Total"].nlargest(5)
plt.figure()
top5_tipos.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"],
    explode=(0.05, 0, 0, 0, 0),
)
plt.title("Los 5 tipos de accidente más frecuentes", fontsize=16)
plt.ylabel("")
plt.tight_layout()
plt.savefig("top5_tipos_pie.png", bbox_inches="tight")


print("Gráficas generadas :)")
