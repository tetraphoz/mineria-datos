import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

try:
    df = pd.read_csv("csv/accidentes_viales_mty.csv")
    print("Archivo cargado correctamente :)")
except Exception as e:
    print(f"Error al leer archivo: {e}")
    exit(1)

df = df.drop(
    columns=[
        "Folio",
        "Fecha",
        "Dia",
        "Hora",
        "Tipo_de_accidente",
        "Resolucion",
        "Origen_de_reporte",
        "Tipo_de_asentamiento",
        "Nombre_de_asentamiento",
        "Tipo_de_vialidad",
        "Nombre_de_la_Vialidad",
        "Georreferencia",
    ],
    errors="ignore",
)

X = df[
    [
        "Dia_num",
        "Hora_num",
        "Latitud",
        "Longitud",
        "Mes_num",
        "es_fin_semana",
        "colonia_alto_riesgo",
    ]
]
y = df["Tipo_simplificado"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
)

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="manhattan",
)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
