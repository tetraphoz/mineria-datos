import pandas as pd
import sys
import requests
from tabulate import tabulate
import io


def get_csv_from_url(url: str) -> pd.DataFrame:
    s = requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode("utf-8")))


try:
    print("Descargando csv...")
    df = get_csv_from_url(
        "https://nuevoleon.opendatasoft.com/api/explore/v2.1/catalog/datasets/indices-de-estadisticas-de-accidentes-viales-monterrey/exports/csv?lang=en&timezone=America%2FMexico_City&use_labels=true&delimiter=%2C"
    )
    print("Csv descargado.")
except Exception:
    print("Error al descargar el archivo.")
    exit(1)

print("Limpiando nombres de columnas.")
df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("á", "a")
    .str.replace("é", "e")
    .str.replace("í", "i")
    .str.replace("ó", "o")
    .str.replace("ú", "u")
)

print(f"Valores nulos por columna: \n{df.isnull().sum()}")
df_cleaned = df.dropna()
print("Valores nulos eliminados.")

df = df.drop(columns=["Nota", "Ejercicio"])
print("Columna 'Nota', 'Ejercicio' eliminadas.")

df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

valores_invalidos = ["SD", "No Dato", "sd"]
df = df[~df["Hora"].isin(valores_invalidos)]
df["Hora"] = pd.to_datetime(df["Hora"]).dt.time
df["Hora_num"] = pd.to_datetime(df["Hora"].astype(str)).dt.hour
print("Convertimos fechas y horas en datetimes.")

# Separar coordenadas geográficas
df[["Latitud", "Longitud"]] = (
    df["Georreferencia"].str.split(", ", expand=True).astype(float)
)

dias_orden = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
mes_orden = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]

df["Dia_num"] = pd.Categorical(df["Dia"], categories=dias_orden, ordered=True).codes
df["Mes_num"] = pd.Categorical(df["Mes"], categories=mes_orden, ordered=True).codes
df = df.dropna()


def categorizar_hora(hora):
    if 0 <= hora < 6:
        return "madrugada"
    elif 6 <= hora < 12:
        return "mañana"
    elif 12 <= hora < 18:
        return "tarde"
    else:
        return "noche"


df["grupo_horario"] = df["Hora_num"].apply(categorizar_hora)
df["es_fin_semana"] = df["Dia"].isin(["Sabado", "Domingo"]).astype(int)

top_colonias = df["Nombre_de_asentamiento"].value_counts().head(10).index
df["colonia_alto_riesgo"] = df["Nombre_de_asentamiento"].isin(top_colonias).astype(int)

df["Tipo_de_accidente"] = df["Tipo_de_accidente"].str.lower().str.strip()
tipo_counts = df["Tipo_de_accidente"].value_counts()
tipos_validos = tipo_counts[tipo_counts > 300].index
df["Tipo_simplificado"] = df["Tipo_de_accidente"].where(
    df["Tipo_de_accidente"].isin(tipos_validos), "otro"
)

df.to_csv("csv/accidentes_viales_mty.csv")
print("Archivo escrito en csv/accidentes_viales_mty.csv")
print(df.describe)
