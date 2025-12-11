import os
import pandas as pd
import streamlit as st

def show_subset_page():
    st.title("1️⃣ Kapitel 1 – Dataset Loader: Genius Song Lyrics (Hugging Face)")

    st.markdown("""
    **Dataset Loader:** Genius Song Lyrics (Hugging Face)  
    **Data Source:** https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics  

    Die ursprüngliche Genius Song Lyrics Dataset enthält ca. **2.76 Millionen Songs** (≈ 9 GB CSV).  
    Um leichtgewichtig experimentieren zu können, erlaubt dieses Skript das Herunterladen und Speichern
    eines **kleineren zufälligen Subsets** (z.B. 1%) als lokale CSV-Datei.
    """)

    st.info(
        "**Hinweis:** Dieser Abschnitt dokumentiert die Schritte aus dem zugehörigen Notebook "
        "`load-data-subset.ipynb`. Alle Berechnungen wurden dort ausgeführt. "
        "Die Streamlit-App lädt lediglich die erzeugten Dateien und visualisiert die Ergebnisse."
    )

    # -----------------------------
    # 1. Dokumentation / Notebook
    # -----------------------------
    st.header("1. Preparations")

    st.subheader("1.1 Load original Dataset from Hugging Face")
    st.markdown("""
    Es wird **nicht** die komplette CSV geladen, sondern das Dataset über Hugging Face Datasets.  
    Standardmäßig lädt `load_dataset` hier nur die **Metadaten + Zugriff auf den `train`-Split**.
    """)
    st.code(
        """
dataset = load_dataset("sebastiandizon/genius-song-lyrics", split="train")""",
        language="python"
    )

    st.subheader("1.3 Configuration")
    st.markdown("""
    - Definiere den Prozentsatz des Subsets (z.B. 1%, 5%, 10%).  
    - Beachte: Hugging Face akzeptiert bei `split`-Notation nur **Ganzzahlen**.  
    - Lege Ausgabeverzeichnis und Dateinamen fest.  
    - Erstelle das Verzeichnis, falls es noch nicht existiert.
    """)
    st.code(
        """
subset_fraction = 1


subset_size = int(len(dataset) * subset_fraction / 100)


output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"lyrics_subset_{subset_fraction}pct.csv")""",
        language="python"
    )

    st.header("2. Load and Save Subset")

    st.subheader("2.1 Load Subset of Dataset")
    st.markdown("""
    - Setze einen **Seed** für Reproduzierbarkeit.  
    - Mische den Datensatz zufällig.  
    - Wähle die ersten `subset_size` Einträge als Subset.
    """)
    st.code(
        """
dataset = dataset.shuffle(seed=42)

dataset_small = dataset.select(range(subset_size))

print(f"Dataset loaded successfully with {len(dataset_small):,} entries.")""",
        language="python"
    )

    st.subheader("2.2 Convert to pandas DataFrame")
    st.markdown("Konvertiere das Subset in ein `pandas.DataFrame` und gib Basis-Statistiken aus.")
    st.code(
        """
df = dataset_small.to_pandas()

print(f"DataFrame shape: {df.shape}")
print(f"Number of Songs: {len(df):,} | Artists: {df['artist'].nunique():,} | Genres: {df['tag'].nunique():,}")""",
        language="python"
    )

    st.subheader("2.3 Save Subset locally")
    st.markdown("Speichere das Subset als CSV-Datei im definierten Ausgabeverzeichnis.")
    st.code(
        """
df.to_csv(output_path, index=False)

print(f"Subset saved to: {output_path}")""",
        language="python"
    )

    st.header("3. Preview of the dataset")

    st.markdown("""
    Zum Abschluss wird ein kurzer Überblick über die ersten Zeilen und die **Genre-Verteilung** gegeben.
    """)

    st.subheader("3.1 Genre-Verteilung")
    st.code(
        """print("\\nGENRE DISTRIBUTION")
print("=" * 60)

category_counts = df['tag'].value_counts().sort_values(ascending=False)

for tag, count in category_counts.items():
    pct = (count / len(df)) * 100
    print(f"{tag}: {count:,} songs ({pct:.2f}%)")""",
        language="python"
    )


    # ----------------------------------------
    # 2. Resultate aus Notebook einlesen
    # ----------------------------------------
    st.header("📁 Datensubset laden Resultat")

    DATA_DIR = "data/raw"
    subset_fraction = 1
    csv_path = os.path.join(DATA_DIR, f"lyrics_subset_{subset_fraction}pct.csv")

    if not os.path.exists(csv_path):
        st.error(
            f"CSV-Datei nicht gefunden: `{csv_path}`. "
            "Bitte zuerst das Notebook ausführen, das das Subset erzeugt."
        )
    else:
        df_real = pd.read_csv(csv_path)

        # Kennzahlen
        song_count = len(df_real)
        artist_count = df_real["artist"].nunique()
        genre_count = df_real["tag"].nunique()

        st.subheader("📌 Dataset-Infos")
        st.markdown(
            f"- **Songs:** {song_count:,}  \n"
            f"- **Artists:** {artist_count:,}  \n"
            f"- **Genres:** {genre_count:,}  \n"
            f"- **Quelle:** `{csv_path}`"
        )

        st.subheader("👀 Vorschau")
        st.dataframe(df_real.head())

        st.subheader("🎵 Genre-Verteilung")
        genre_counts = df_real["tag"].value_counts()
        report_lines = ["GENRE DISTRIBUTION", "=" * 60]
        for genre, count in genre_counts.items():
            pct = (count / len(df_real)) * 100
            report_lines.append(f"{genre}: {count:,} songs ({pct:.2f}%)")

        st.text("\n".join(report_lines))
