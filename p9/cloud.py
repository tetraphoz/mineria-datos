from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("csv/accidentes_viales_mty.csv")
except Exception as e:
    print(f"Error al leer archivo: {e}")
    sys.exit(1)
else:
    print("Archivo cargado correctamente :)")

# Preparar el texto (combinamos tipo de accidente y colonia)
text_data = df["Tipo_de_accidente"].str.cat(df["Nombre_de_asentamiento"], sep=" ")

# Contar frecuencias de palabras
all_text = " ".join(text_data).lower()
word_counts = Counter(all_text.split())

# Filtrar palabras irrelevantes (personalizable)
stopwords_es = {"de", "la", "en", "y", "del", "los", "las", "el", "por", "con"}
filtered_words = {
    word: count
    for word, count in word_counts.items()
    if word not in stopwords_es and len(word) > 3
}

wordcloud = WordCloud(
    width=800, height=400, background_color="white", colormap="viridis", max_words=50
).generate_from_frequencies(filtered_words)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Palabras más frecuentes en reportes de accidentes", pad=20, size=16)
plt.savefig("palabras_frecuentes_accidentes.png")

# Versión alternativa: solo tipos de accidente
tipo_counts = df["Tipo_de_accidente"].value_counts().to_dict()
wordcloud_tipos = WordCloud(
    width=600, height=300, background_color="white"
).generate_from_frequencies(tipo_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_tipos, interpolation="bilinear")
plt.axis("off")
plt.title("Tipos de accidente más frecuentes", pad=15, size=14)
plt.savefig("tipos_accidentes_frecuentes.png")
