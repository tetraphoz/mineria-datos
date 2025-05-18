import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("csv/accidentes_viales_mty.csv")
    print("Archivo cargado correctamente :)")
except Exception as e:
    print(f"Error al leer archivo: {e}")
    exit(1)


def categorizar_hora(hora):
    if 0 <= hora < 6:
        return 0  # madrugada
    elif 6 <= hora < 12:
        return 1  # mañana
    elif 12 <= hora < 18:
        return 2  # tarde
    else:
        return 3  # noche


df["grupo_horario"] = df["Hora_num"].apply(categorizar_hora)

X = df[["Latitud", "Longitud", "es_fin_semana", "grupo_horario"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usar un número fijo de clusters (11 basado en tipos de accidente)
n_clusters = 11
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# Visualizar clusters en mapa
plt.figure(figsize=(12, 8))
plt.scatter(df["Longitud"], df["Latitud"], c=df["Cluster"], cmap="tab20", alpha=0.6)
plt.title("Distribución Geográfica de Clusters de Accidentes")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.colorbar(label="ID del Cluster")
plt.savefig("clusters.png")

# Perfiles de los clusters
perfiles_cluster = (
    df.groupby("Cluster")
    .agg(
        {
            "Tipo_simplificado": lambda x: x.mode()[0],
            "es_fin_semana": "mean",
            "grupo_horario": lambda x: x.mode()[0],
            "Latitud": "mean",
            "Longitud": "mean",
        }
    )
    .sort_values("grupo_horario")
)

print("\nPerfiles de los Clusters:")
print(perfiles_cluster)
