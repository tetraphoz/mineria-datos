import pandas as pd
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
else:
    print("Archivo cargado correctamente :)")

print("\n=== Estadísticas generales ===")
print(f"Total de accidentes registrados: {len(df)}")
print(f"Periodo cubierto: desde {df['Fecha'].min()} hasta {df['Fecha'].max()}")

try:
    accidentes_por_tipo = (
        df.groupby("Tipo_de_accidente")
        .agg(Total=("Folio", "count"), Hora_promedio=("Hora_num", "mean"))
        .sort_values("Total", ascending=False)
    )

except Exception as e:
    print(f"\nError en análisis por tipo: {e}")

try:
    accidentes_por_dia = (
        df.groupby("Dia")
        .agg(
            Total=("Folio", "count"),
            Hora_moda=("Hora_num", primera_moda),
            Tipo_mas_comun=("Tipo_de_accidente", primera_moda),
        )
        .sort_values("Total", ascending=False)
    )

    print("\n=== Accidentes por día ===")
    print(accidentes_por_dia)

except Exception as e:
    print(f"\nError en análisis por día: {e}")

try:
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

    print("\n=== Top 10 colonias con más accidentes ===")
    print(top_colonias)

except Exception as e:
    print(f"\nError en análisis por colonia: {e}")

try:
    resolucion_por_tipo = (
        df.groupby(["Tipo_de_accidente", "Resolucion"]).size().unstack(fill_value=0)
    )

    if "Finiquitado" in resolucion_por_tipo.columns:
        resolucion_por_tipo["Porcentaje_Finiquitado"] = (
            resolucion_por_tipo["Finiquitado"] / resolucion_por_tipo.sum(axis=1)
        ) * 100

    print("\n=== Resolución de casos por tipo de accidente ===")
    print(resolucion_por_tipo)

except Exception as e:
    print(f"\nError en análisis de resoluciones: {e}")

try:
    accidentes_por_hora = (
        df.groupby("Hora_num")
        .agg(
            Total=("Folio", "count"), Tipo_mas_comun=("Tipo_de_accidente", primera_moda)
        )
        .sort_index()
    )

    print("\n=== Distribución horaria de accidentes ===")
    print(accidentes_por_hora)

except Exception as e:
    print(f"\nError en análisis horario: {e}")
