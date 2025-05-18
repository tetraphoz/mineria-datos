from sklearn.linear_model import LinearRegression
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.dates as mdates

try:
    df = pd.read_csv("csv/accidentes_viales_mty.csv")
except Exception as e:
    print(f"Error al leer archivo: {e}")
    sys.exit(1)
else:
    print("Archivo cargado correctamente :)")

# Preparar serie de tiempo (accidentes por fecha)
df["Fecha"] = pd.to_datetime(df["Fecha"])
ts_data = df.groupby("Fecha").size().reset_index(name="Accidentes")

# Crear variables de tiempo
ts_data["Dia_num"] = (ts_data["Fecha"] - ts_data["Fecha"].min()).dt.days

# Modelo de regresión para pronóstico
X = ts_data[["Dia_num"]]
y = ts_data["Accidentes"]
modelo_ts = LinearRegression()
modelo_ts.fit(X, y)

# Pronóstico para los próximos 7 días
ultima_fecha = ts_data["Dia_num"].max()
ultimo_dia_real = ts_data["Fecha"].max()
futuro = pd.DataFrame({"Dia_num": range(ultima_fecha + 1, ultima_fecha + 8)})
futuro["Prediccion"] = modelo_ts.predict(futuro[["Dia_num"]])
futuro["Fecha"] = pd.date_range(start=ultimo_dia_real, periods=8)[1:]

# Crear figura con dos gráficos
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
)

# Gráfico 1: Serie temporal completa
ax1.plot(ts_data["Fecha"], y, label="Datos reales")
ax1.plot(
    futuro["Fecha"],
    futuro["Prediccion"],
    "r--",
    label="Pronóstico",
)
ax1.set_title("Serie Temporal Completa con Pronóstico")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Número de Accidentes")
ax1.legend()
ax1.grid(True)

# Gráfico 2: Solo los últimos 30 días y el pronóstico
# Filtrar solo los últimos 30 días de datos reales
ultimos_30 = ts_data[ts_data["Fecha"] >= ts_data["Fecha"].max() - pd.Timedelta(days=30)]

ax2.plot(ultimos_30["Fecha"], ultimos_30["Accidentes"], "b-", label="Últimos 30 días")
ax2.plot(
    futuro["Fecha"],
    futuro["Prediccion"],
    "r--",
    marker="o",
    label="Pronóstico (7 días)",
)
ax2.set_title("Detalle: Últimos 30 días y Pronóstico")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Número de Accidentes")
ax2.legend()
ax2.grid(True)

# Mejorar formato de fechas en el eje x
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("pronostico_accidentes.png")

# Mostrar tabla con los valores pronosticados
print("\nValores pronosticados para los próximos 7 días:")
tabla_pronostico = futuro[["Fecha", "Prediccion"]].copy()
tabla_pronostico["Fecha"] = tabla_pronostico["Fecha"].dt.strftime("%d-%m-%Y")
tabla_pronostico["Prediccion"] = tabla_pronostico["Prediccion"].round(1)
print(tabla_pronostico.to_string(index=False))

# Evaluación del modelo
r2_ts = r2_score(y, modelo_ts.predict(X))
print(f"\nR² del modelo de serie de tiempo: {r2_ts:.4f}")
