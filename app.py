# app.py
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

import json
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
import joblib

import markovify
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


# -------------------------------------------------------
# Basis-Konfiguration
# -------------------------------------------------------

st.set_page_config(
    page_title="Lyrics Text Analytics",
    page_icon="🎶",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_CLEAN_DIR = BASE_DIR / "data" / "clean"
MODELS_DIR = BASE_DIR / "models"


DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# UI: Sidebar Navigation
# -------------------------------------------------------

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Ansicht auswählen:",
    [
        "1️⃣ Datensubset laden",
        "2️⃣ Daten bereinigen",
        "3️⃣ Tokenisierung",
        "4️⃣ Statistische Analyse",
        "5️⃣ Word Embedding",
        "6️⃣ Model Evaluation",
        "7️⃣ Text Classification",
        "8️⃣ Text-Generierung",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Notebooks bleiben für Experimente. "
    "Die App nutzt die gleiche Logik, aber Schritt für Schritt."
)

# -------------------------------------------------------
# Seite 1 – Datensubset laden
# -------------------------------------------------------
if page == "1️⃣ Datensubset laden":
    st.title("1️⃣ Kapitel 1 – Dataset Loader: Genius Song Lyrics (Hugging Face)")

    st.markdown("""
    **Dataset Loader:** Genius Song Lyrics (Hugging Face)  
    **Data Source:** https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics  

    Die ursprüngliche Genius Song Lyrics Dataset enthält ca. **2.76 Millionen Songs** (≈ 9 GB CSV).  
    Um leichtgewichtig experimentieren zu können, erlaubt dieses Skript das Herunterladen und Speichern
    eines **kleineren zufälligen Subsets** (z.B. 1%) als lokale CSV-Datei.
    """)

    # -----------------------------
    # 1. Dokumentation / Notebook
    # -----------------------------
    st.header("1. Preparations")

    st.subheader("1.1 Import Packages")
    st.markdown("Benötigte Python-Packages für Dataset-Download, Verarbeitung und Dateiverwaltung:")
    st.code(
        """from datasets import load_dataset
import pandas as pd
import os""",
        language="python"
    )

    st.subheader("1.2 Load original Dataset from Hugging Face")
    st.markdown("""
    Es wird **nicht** die komplette CSV geladen, sondern das Dataset über Hugging Face Datasets.  
    Standardmäßig lädt `load_dataset` hier nur die **Metadaten + Zugriff auf den `train`-Split**.
    """)
    st.code(
        """# Lädt den 'train'-Split des Genius Song Lyrics Datasets
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
        """# Prozentualer Anteil des Datensatzes, der lokal gespeichert werden soll
subset_fraction = 1

# Anzahl Einträge für das Subset
subset_size = int(len(dataset) * subset_fraction / 100)

# Ausgabeverzeichnis und -pfad
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
        """# Datensatz mischen für zufälliges Subset
dataset = dataset.shuffle(seed=42)

# Subset auswählen
dataset_small = dataset.select(range(subset_size))

print(f"Dataset loaded successfully with {len(dataset_small):,} entries.")""",
        language="python"
    )

    st.subheader("2.2 Convert to pandas DataFrame")
    st.markdown("Konvertiere das Subset in ein `pandas.DataFrame` und gib Basis-Statistiken aus.")
    st.code(
        """# Konvertieren in pandas DataFrame
df = dataset_small.to_pandas()

print(f"DataFrame shape: {df.shape}")
print(f"Number of Songs: {len(df):,} | Artists: {df['artist'].nunique():,} | Genres: {df['tag'].nunique():,}")""",
        language="python"
    )

    st.subheader("2.3 Save Subset locally")
    st.markdown("Speichere das Subset als CSV-Datei im definierten Ausgabeverzeichnis.")
    st.code(
        """# Subset lokal als CSV speichern
df.to_csv(output_path, index=False)

print(f"Subset saved to: {output_path}")""",
        language="python"
    )

    st.header("3. Preview of the dataset")

    st.markdown("""
    Zum Abschluss wird ein kurzer Überblick über die ersten Zeilen und die **Genre-Verteilung** gegeben.
    """)

    st.subheader("3.1 Erste Zeilen anzeigen")
    st.code(
        """# Vorschau der ersten Zeilen
df.head()""",
        language="python"
    )

    st.subheader("3.2 Genre-Verteilung")
    st.code(
        """print("\\nGENRE DISTRIBUTION")
print("=" * 60)

category_counts = df['tag'].value_counts().sort_values(ascending=False)

for tag, count in category_counts.items():
    pct = (count / len(df)) * 100
    print(f"{tag}: {count:,} songs ({pct:.2f}%)")""",
        language="python"
    )

    st.info(
        "Hinweis: Dieser Abschnitt zeigt die **Dokumentation** des Notebooks. "
        "Die eigentlichen Berechnungen (Download, Sampling, Speichern) laufen im Jupyter Notebook. "
        "Im nächsten Abschnitt werden die **tatsächlichen Resultate** aus der vom Notebook gespeicherten CSV geladen."
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

        st.subheader("👀 Vorschau (df.head)")
        st.dataframe(df_real.head())

        st.subheader("🎵 Genre-Verteilung (REAL)")
        genre_counts = df_real["tag"].value_counts()
        report_lines = ["GENRE DISTRIBUTION", "=" * 60]
        for genre, count in genre_counts.items():
            pct = (count / len(df_real)) * 100
            report_lines.append(f"{genre}: {count:,} songs ({pct:.2f}%)")

        st.text("\n".join(report_lines))

        st.info(
            "Die oben angezeigten Werte stammen **direkt aus der vom Notebook erzeugten CSV-Datei**. "
            "In der Streamlit-App wird dazu nur die Datei gelesen – keine neuen Downloads oder "
            "aufwendigen Berechnungen."
        )

# -------------------------------------------------------
# Seite 2 – Daten bereinigen
# -------------------------------------------------------
elif page == "2️⃣ Daten bereinigen":
    st.title("2️⃣ Kapitel 2 – Data Cleaning: Genius Song Lyrics Subset")

    st.markdown("""
    **Purpose:**  
    Bereinigung und Vorverarbeitung der Songtexte für Analysen.  
    Entfernt werden u.a.:
    - Metadaten-Tags (z.B. `[Intro]`, `[Verse]`)
    - Zeilenumbrüche (`\\n`)
    - überflüssige Leerzeichen  

    Ziel ist eine saubere Textspalte, die sich für **NLP** und **statistische Analysen** eignet.
    """)

    # -----------------------------------
    # 1. Dataset Overview (Dokumentation)
    # -----------------------------------
    st.header("1. Dataset Overview")

    st.subheader("1.1 Import Packages and Settings")
    st.code(
        """import pandas as pd
import re
import os""",
        language="python"
    )

    st.subheader("1.2 Load Dataset")
    st.markdown("""
    Es wird das im **Kapitel 1** erzeugte Datensubset geladen, welches die rohen Lyrics enthält.
    """)
    st.code(
        """input_dir = "data/raw"
input_path = os.path.join(input_dir, "lyrics_subset_1pct.csv")

df = pd.read_csv(input_path)""",
        language="python"
    )

    # -----------------------
    # 2. Data Cleaning (Doku)
    # -----------------------
    st.header("2. Data Cleaning")

    st.subheader("2.1 Problem")
    st.markdown("""
    Eine Vorschau der rohen Lyrics zeigt typische Probleme:
    - Metadaten-Tags wie `[Intro]`, `[Verse]`, `[Hook]`
    - Zeilenumbrüche `\\n`
    - Mehrfache bzw. führende/abschließende Leerzeichen  

    Diese müssen vor der Analyse entfernt werden.
    """)

    st.subheader("2.2 Define and Apply Cleaning Function")
    st.markdown("""
    Es wird eine Funktion definiert, die nacheinander:
    - `re.sub(r'\\[.*?\\]', '', text)` → entfernt **alles zwischen** `[` und `]`  
    - `text.replace('\\n', ' ')` → ersetzt Zeilenumbrüche durch ein Leerzeichen  
    - `re.sub(r'\\s+', ' ', text).strip()` → reduziert Mehrfach-Leerzeichen auf eins und trimmt den Text
    """)
    st.code(
        """import re

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1) Metadaten-Tags entfernen, z.B. [Intro], [Verse]
    text = re.sub(r'\\[.*?\\]', '', text)

    # 2) Zeilenumbrüche durch Leerzeichen ersetzen
    text = text.replace("\\n", " ")

    # 3) Mehrfache Leerzeichen entfernen und Text trimmen
    text = re.sub(r"\\s+", " ", text).strip()

    return text

# Neue Spalte mit bereinigten Lyrics
df["lyrics_clean"] = df["lyrics"].apply(clean_lyrics)""",
        language="python"
    )

    st.subheader("2.4 Preview Cleaned Lyrics")
    st.markdown("""
    Nach Anwendung der Cleaning-Funktion sind:
    - Metadaten-Tags (z.B. `[Intro]`, `[Verse]`) entfernt  
    - Zeilenumbrüche `\\n` durch Leerzeichen ersetzt  
    - doppelte oder mehrfache Leerzeichen bereinigt  
    """)
    st.code(
        """# Vorschau: rohe Lyrics vs. bereinigte Lyrics
df[["lyrics", "lyrics_clean"]].head()""",
        language="python"
    )

    # --------------------------
    # 3. Save cleaned Data DOKU
    # --------------------------
    st.header("3. Save cleaned Data")

    st.subheader("3.1 Configuration")
    st.markdown("""
    - Definiere Ausgabeverzeichnis und Dateinamen.  
    - Erstelle das Verzeichnis, falls es noch nicht existiert.
    """)
    st.code(
        """output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "lyrics_subset_1pct_clean.csv")""",
        language="python"
    )

    st.subheader("3.2 Prepare Data for Saving")
    st.markdown("""
    Vor dem Speichern wird:
    - die ursprüngliche Spalte `lyrics` entfernt  
    - `lyrics_clean` in `lyrics` umbenannt, sodass die bereinigten Texte im Feld `lyrics` liegen.
    """)
    st.code(
        """# Originalspalte entfernen
df = df.drop(columns=["lyrics"])

# Bereinigte Spalte umbenennen
df = df.rename(columns={"lyrics_clean": "lyrics"})""",
        language="python"
    )

    st.subheader("3.3 Save Subset locally")
    st.code(
        """df.to_csv(output_path, index=False)
print(f"Cleaned subset saved to: {output_path}")""",
        language="python"
    )

    st.info(
        "Obiger Abschnitt beschreibt nur den **Notebook-Workflow**. "
        "Die eigentliche Bereinigung (Cleaning) wurde im Jupyter Notebook ausgeführt. "
        "Im folgenden Abschnitt werden die **bereits gereinigten Daten** aus der gespeicherten CSV geladen."
    )

    # -------------------------------------------------
    # 4. Bereits gereinigte Daten aus CSV laden
    # -------------------------------------------------
    st.header("📁 Bereinigte Daten")

    CLEAN_DATA_DIR = "data/clean"
    clean_path = os.path.join(CLEAN_DATA_DIR, "lyrics_subset_1pct_clean.csv")

    if not os.path.exists(clean_path):
        st.error(
            f"Bereinigte CSV-Datei nicht gefunden: `{clean_path}`. "
            "Bitte zuerst das Data-Cleaning-Notebook ausführen, das diese Datei erzeugt."
        )
    else:
        df_clean = pd.read_csv(clean_path)

        # Basisinfos – sehr leichte Operationen
        st.subheader("📌 Basis-Infos (bereinigtes Subset)")
        st.markdown(
            f"- **Anzahl Zeilen (Songs):** {len(df_clean):,}  \n"
            f"- **Spalten:** {', '.join(df_clean.columns)}  \n"
            f"- **Quelle:** `{clean_path}`"
        )

        st.subheader("👀 Vorschau der bereinigten Lyrics")
        display_cols = [c for c in ["artist", "title", "tag", "lyrics"] if c in df_clean.columns]
        st.dataframe(df_clean[display_cols].head())

        if "lyrics" in df_clean.columns:
            df_clean["lyrics_len"] = df_clean["lyrics"].astype(str).str.len()
            st.subheader("📝 Textlängen-Überblick (bereinigte Lyrics)")
            st.write(
                f"Durchschnittliche Länge: {int(df_clean['lyrics_len'].mean()):,} Zeichen  \n"
                f"Median: {int(df_clean['lyrics_len'].median()):,} Zeichen"
            )

        st.info(
            "Alle oben angezeigten Inhalte stammen aus der **bereits vom Notebook bereinigten CSV-Datei**. "
            "In der Streamlit-App werden die Daten nur geladen und inspiziert – "
            "es findet **keine erneute Datenbereinigung** statt."
        )

# -------------------------------------------------------
# Seite 3 – Tokenisierung
# -------------------------------------------------------
elif page == "3️⃣ Tokenisierung":
    st.title("3️⃣ Kapitel 3 – Tokenization: Genius Song Lyrics Subset (1%)")

    st.markdown("""
    **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  

    **Purpose:**  
    Vorbereitung der Textdaten für weitere Analysen, indem Songtexte in einzelne Tokens
    zerlegt und Stopwörter entfernt werden.  
    Diese Schritte sorgen für eine saubere und strukturierte Darstellung der Texte und bilden
    die Grundlage für nachgelagerte NLP- und Statistik-Analysen.
    """)

    # -----------------------------
    # 1. Preparation (Dokumentation)
    # -----------------------------
    st.header("1. Preparation")

    st.subheader("1.1 Import Packages and Settings")
    st.code(
        """import pandas as pd
import re
import os
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt""",
        language="python"
    )

    st.subheader("1.2 Load Dataset")
    st.markdown("""
    Es wird das im **Kapitel 2** bereinigte Datenset geladen, das bereits  
    von Metadaten-Tags und Zeilenumbrüchen befreite Lyrics enthält.
    """)
    st.code(
        """input_dir = "data/processed"
input_path = os.path.join(input_dir, "lyrics_subset_1pct_clean.csv")

df = pd.read_csv(input_path)""",
        language="python"
    )

    # -----------------------
    # 2. Tokenization (Doku)
    # -----------------------
    st.header("2. Tokenization")

    st.subheader("2.1 Build Tokens")
    st.markdown("""
    Zuerst werden die Songtexte in einzelne Tokens zerlegt, d.h. jedes Wort wird
    aus dem vollständigen Text extrahiert.  
    Die resultierenden Tokens werden in einer neuen Spalte `words` gespeichert,  
    zusätzlich wird `word_count` als Anzahl Tokens pro Song ergänzt.
    """)
    st.code(
        """# Beispiel: einfache Tokenisierung über Whitespace
def to_words(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return text.split()

df["words"] = df["lyrics"].apply(to_words)
df["word_count"] = df["words"].apply(len)""",
        language="python"
    )

    st.subheader("2.2 Filter Stopwords")
    st.markdown("""
    Anschließend werden Stopwörter entfernt.  
    Dadurch rücken inhaltlich bedeutungsvolle Wörter in den Vordergrund.
    - Die gefilterten Tokens werden in `tokens` gespeichert  
    - Die Anzahl der Tokens pro Song in `token_count`
    """)
    st.code(
        """stop_words = set(stopwords.words("english"))

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in stop_words]

df["tokens"] = df["words"].apply(remove_stopwords)
df["token_count"] = df["tokens"].apply(len)""",
        language="python"
    )

    # 🔹 NEU: 2.3 Visualisierung der häufigsten Wörter
    st.subheader("2.3 Visualisierung der häufigsten Wörter")
    st.markdown("""
    Zur Veranschaulichung werden die häufigsten Wörter **vor** und **nach**
    dem Entfernen der Stopwörter gegenübergestellt.  

    Die Plots zeigen deutlich, dass das Entfernen von Stopwörtern die Verteilung stark verändert:
    Das häufigste Wort **nach** dem Filtern taucht in den ursprünglichen Top-15 gar nicht mehr auf.
    """)
    st.code(
        """top_n = 15
words = [t for row in df["words"] for t in row]
tokens_filtered = [t for row in df["tokens"] for t in row]

word_counts_raw = Counter(words).most_common(top_n)
word_counts_filtered = Counter(tokens_filtered).most_common(top_n)

df_raw = pd.DataFrame(word_counts_raw, columns=["word", "count"])
df_filtered = pd.DataFrame(word_counts_filtered, columns=["word", "count"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].bar(df_raw["word"], df_raw["count"])
axes[0].set_title(f"Top {top_n} Words (Before Stopword Removal)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(df_filtered["word"], df_filtered["count"])
axes[1].set_title(f"Top {top_n} Words (After Stopword Removal)", fontsize=12, fontweight='bold')
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()""",
        language="python"
    )

    st.markdown("""
    *Die oben dargestellten Plots werden im Notebook erzeugt.  
    In der Streamlit-App dienen sie als **Dokumentation des Analyse-Schritts**.*
    """)

    # --------------------------
    # 3. Save final Dataset DOKU
    # --------------------------
    st.header("3. Save final Dataset")

    st.markdown("""
    Das finale Datenset enthält u.a. die neuen Spalten:
    - `words`, `word_count`  
    - `tokens`, `token_count`  

    und dient als bereinigte und vorbereitete Basis für alle weiteren Textanalysen.
    Es wird unter `data/clean/data.csv` gespeichert.
    """)

    st.subheader("3.1 Configuration")
    st.code(
        """output_dir = "data/clean"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "data.csv")""",
        language="python"
    )

    st.subheader("3.2 Save Dataset locally")
    st.code(
        """df.to_csv(output_path, index=False)
print(f"Final tokenized dataset saved to: {output_path}")""",
        language="python"
    )

    st.info(
        "Obiger Abschnitt beschreibt den **Notebook-Workflow** für die Tokenisierung. "
        "Die eigentliche Tokenisierung, Stopwort-Filterung und Plot-Erstellung wurde im Jupyter Notebook ausgeführt. "
        "Im folgenden Abschnitt werden die **fertigen Tokenisierungs-Resultate** aus `data/clean/data.csv` geladen."
    )

    # -------------------------------------------------
    # 4. Analyse-Resultate aus data/clean/data.csv
    # -------------------------------------------------
    st.header("📊 Analyse-Resultate (aus finalem Datensatz)")

    DATA_CLEAN_DIR = "data/clean"
    stats_path = os.path.join(DATA_CLEAN_DIR, "data.csv")

    if not os.path.exists(stats_path):
        st.error(
            f"Finaler Datensatz nicht gefunden: `{stats_path}`. "
            "Bitte zuerst die vorherigen Kapitel-Notebooks bis zur Speicherung von `data/clean/data.csv` ausführen."
        )
    else:
        import ast
        import itertools
        from collections import Counter

        import numpy as np
        import matplotlib.pyplot as plt

        df_stats = pd.read_csv(stats_path)


        # Hilfs-Funktion, um Listenspalten aus CSV zu parsen (words/tokens als Python-Liste)
        def parse_list_column(series):
            return series.fillna("[]").apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )


        if "words" in df_stats.columns:
            df_stats["words"] = parse_list_column(df_stats["words"])
        if "tokens" in df_stats.columns:
            df_stats["tokens"] = parse_list_column(df_stats["tokens"])

        # Basisgrössen
        genre_counts = df_stats["tag"].value_counts() if "tag" in df_stats.columns else None
        word_counts = df_stats["word_count"] if "word_count" in df_stats.columns else None
        token_counts = df_stats["token_count"] if "token_count" in df_stats.columns else None

        all_tokens = list(
            itertools.chain.from_iterable(df_stats["tokens"])
        ) if "tokens" in df_stats.columns else []

        freq_dist = Counter(all_tokens) if all_tokens else Counter()

        # Zipf-Fit
        if freq_dist:
            freq_values = np.array(sorted(freq_dist.values(), reverse=True))
            ranks = np.arange(1, len(freq_values) + 1)
            log_ranks = np.log10(ranks)
            log_freqs = np.log10(freq_values)

            slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
            predicted = slope * log_ranks + intercept
        else:
            slope = intercept = None
            log_ranks = log_freqs = predicted = None

        # Hapax / seltene Wörter
        hapax = [w for w, c in freq_dist.items() if c == 1] if freq_dist else []
        rare = [w for w, c in freq_dist.items() if c <= 5] if freq_dist else []


        # N-Gramme
        def make_ngrams(tokens, n):
            return zip(*[tokens[i:] for i in range(n)])


        if all_tokens:
            unigrams = all_tokens
            bigrams_all = list(
                itertools.chain.from_iterable(
                    make_ngrams(toks, 2) for toks in df_stats["tokens"]
                )
            )
            trigrams_all = list(
                itertools.chain.from_iterable(
                    make_ngrams(toks, 3) for toks in df_stats["tokens"]
                )
            )

            top_15_unigrams = Counter(unigrams).most_common(15)
            top_15_bigrams = Counter(bigrams_all).most_common(15)
            top_15_trigrams = Counter(trigrams_all).most_common(15)
        else:
            top_15_unigrams = top_15_bigrams = top_15_trigrams = []

        # Genre-Stats für Category-Tabelle + Plot
        if "tag" in df_stats.columns and "word_count" in df_stats.columns:
            genre_stats = (
                df_stats.groupby("tag")
                .agg(
                    songs=("title", "count") if "title" in df_stats.columns else ("tag", "count"),
                    avg_word_count=("word_count", "mean"),
                    avg_token_count=("token_count", "mean") if "token_count" in df_stats.columns else ("word_count",
                                                                                                       "mean"),
                )
            )
        else:
            genre_stats = None

        # Tabs mit Grafiken
        (
            tab_genre,
            tab_text,
            tab_token,
            tab_vocab,
            tab_zipf,
            tab_rare,
            tab_cat,
            tab_ngram,
            tab_uni,
            tab_bi,
            tab_tri,
        ) = st.tabs([
            "GENRE DISTRIBUTION",
            "TEXT STATISTICS",
            "TOKEN STATISTICS",
            "VOCABULARY STATISTICS",
            "ZIPF'S LAW ANALYSIS",
            "RARE WORDS ANALYSIS",
            "Category Statistics",
            "N-gram Analysis",
            "TOP 15 UNIGRAMS",
            "TOP 15 BIGRAMS",
            "TOP 15 TRIGRAMS",
        ])

        # --- GENRE DISTRIBUTION ---
        with tab_genre:
            st.subheader("GENRE DISTRIBUTION")
            if genre_counts is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(genre_counts.index, genre_counts.values)
                ax.set_ylabel("Anzahl Songs")
                ax.set_xlabel("Genre")
                ax.set_title("Genre-Verteilung")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.dataframe(genre_counts.to_frame(name="count"))
            else:
                st.warning("Spalte `tag` nicht im Datensatz gefunden.")

        # --- TEXT STATISTICS ---
        with tab_text:
            st.subheader("TEXT STATISTICS (Wortanzahl vor Stopwörtern)")
            if word_counts is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(word_counts, bins=50)
                ax.set_xlabel("Wörter pro Song (word_count)")
                ax.set_ylabel("Häufigkeit")
                ax.set_title("Verteilung der Wortanzahl")
                st.pyplot(fig)
                st.write(word_counts.describe())
            else:
                st.warning("Spalte `word_count` nicht im Datensatz gefunden.")

        # --- TOKEN STATISTICS ---
        with tab_token:
            st.subheader("TOKEN STATISTICS (nach Stopwörtern)")
            if token_counts is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(token_counts, bins=50)
                ax.set_xlabel("Tokens pro Song (token_count)")
                ax.set_ylabel("Häufigkeit")
                ax.set_title("Verteilung der Tokens")
                st.pyplot(fig)
                st.write(token_counts.describe())
            else:
                st.warning("Spalte `token_count` nicht im Datensatz gefunden.")

        # --- VOCABULARY STATISTICS ---
        with tab_vocab:
            st.subheader("VOCABULARY STATISTICS")
            if all_tokens:
                vocab_size = len(freq_dist)
                num_tokens = len(all_tokens)
                ttr = vocab_size / num_tokens
                st.markdown(
                    f"- **Vokabulargröße:** {vocab_size:,}  \n"
                    f"- **Anzahl Tokens:** {num_tokens:,}  \n"
                    f"- **Type–Token Ratio (TTR):** {ttr:.3f}"
                )

                # einfache Log-Log-Vokabular-Kurve (optional)
                fig, ax = plt.subplots(figsize=(6, 4))
                sorted_freqs = np.array(sorted(freq_dist.values(), reverse=True))
                ranks_v = np.arange(1, len(sorted_freqs) + 1)
                ax.loglog(ranks_v, sorted_freqs)
                ax.set_xlabel("Rang (log)")
                ax.set_ylabel("Frequenz (log)")
                ax.set_title("Wortfrequenzen (Log-Log)")
                st.pyplot(fig)
            else:
                st.warning("Tokens konnten nicht berechnet werden.")

        # --- ZIPF'S LAW ANALYSIS ---
        with tab_zipf:
            st.subheader("ZIPF'S LAW ANALYSIS")
            if slope is not None:
                st.markdown(
                    f"- **Steigung (α):** {slope:.2f}  \n"
                    f"- **R²:** {np.corrcoef(log_ranks, log_freqs)[0, 1] ** 2:.3f}"
                )
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(log_ranks, log_freqs, s=5, label="Beobachtet")
                ax.plot(log_ranks, predicted, label="Zipf-Fit", linewidth=2)
                ax.set_xlabel("log10(Rang)")
                ax.set_ylabel("log10(Frequenz)")
                ax.set_title("Zipf's Law – Frequenz vs. Rang")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Zipf-Analyse konnte nicht durchgeführt werden.")

        # --- RARE WORDS ANALYSIS ---
        with tab_rare:
            st.subheader("RARE WORDS ANALYSIS")
            if freq_dist:
                hapax_ratio = len(hapax) / len(freq_dist)
                rare_ratio = len(rare) / len(freq_dist)
                st.markdown(
                    f"- **Anteil Hapax Legomena (Freq=1):** {hapax_ratio * 100:.1f}%  \n"
                    f"- **Anteil Wörter mit ≤ 5 Vorkommen:** {rare_ratio * 100:.1f}%"
                )

                # Verteilung für Freq 1–10
                max_k = 10
                freq_counts = Counter(freq_dist.values())
                xs = list(range(1, max_k + 1))
                ys = [freq_counts.get(k, 0) for k in xs]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(xs, ys)
                ax.set_xlabel("Anzahl Vorkommen")
                ax.set_ylabel("Anzahl unterschiedlicher Wörter")
                ax.set_title("Verteilung seltener Wörter (1–10 Vorkommen)")
                st.pyplot(fig)

                st.markdown("**Beispiel-Hapax (Auszug):**")
                st.write(hapax[:20])
            else:
                st.warning("Keine Token-Frequenzen verfügbar.")

        # --- Category Statistics ---
        with tab_cat:
            st.subheader("Category Statistics (nach Genre)")
            if genre_stats is not None:
                st.dataframe(genre_stats)

                # Plot: durchschnittliche Tokens pro Song je Genre
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(genre_stats.index, genre_stats["avg_token_count"])
                ax.set_xlabel("Genre")
                ax.set_ylabel("Ø Tokens pro Song")
                ax.set_title("Durchschnittliche Songlänge (nach Stopwörtern)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Genre- oder Wortlängen-Spalten fehlen im Datensatz.")

        # --- N-gram Analysis (Übersicht) ---
        with tab_ngram:
            st.subheader("N-gram Analysis – Überblick")
            if all_tokens:
                st.markdown(
                    f"- **Unigram-Vokabular:** {len(freq_dist):,}  \n"
                    f"- **Anzahl Unigram-Tokens:** {len(all_tokens):,}"
                )
                st.markdown("""
                Details zu häufigsten N-Grammen finden sich in den Tabs  
                **TOP 15 UNIGRAMS**, **TOP 15 BIGRAMS** und **TOP 15 TRIGRAMS**.
                """)
            else:
                st.warning("Keine Tokens verfügbar.")

        # --- TOP 15 UNIGRAMS ---
        with tab_uni:
            st.subheader("TOP 15 UNIGRAMS")
            if top_15_unigrams:
                df_uni = pd.DataFrame(top_15_unigrams, columns=["unigram", "count"])
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(df_uni["unigram"], df_uni["count"])
                ax.invert_yaxis()
                ax.set_xlabel("Frequenz")
                ax.set_title("Top 15 Unigrams")
                st.pyplot(fig)
                st.dataframe(df_uni)
            else:
                st.warning("Keine Unigramme verfügbar.")

        # --- TOP 15 BIGRAMS ---
        with tab_bi:
            st.subheader("TOP 15 BIGRAMS")
            if top_15_bigrams:
                df_bi = pd.DataFrame(
                    [(f"{w1} {w2}", c) for (w1, w2), c in top_15_bigrams],
                    columns=["bigram", "count"],
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(df_bi["bigram"], df_bi["count"])
                ax.invert_yaxis()
                ax.set_xlabel("Frequenz")
                ax.set_title("Top 15 Bigrams")
                st.pyplot(fig)
                st.dataframe(df_bi)
            else:
                st.warning("Keine Bigrams verfügbar.")

        # --- TOP 15 TRIGRAMS ---
        with tab_tri:
            st.subheader("TOP 15 TRIGRAMS")
            if top_15_trigrams:
                df_tri = pd.DataFrame(
                    [(f"{w1} {w2} {w3}", c) for (w1, w2, w3), c in top_15_trigrams],
                    columns=["trigram", "count"],
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(df_tri["trigram"], df_tri["count"])
                ax.invert_yaxis()
                ax.set_xlabel("Frequenz")
                ax.set_title("Top 15 Trigrams")
                st.pyplot(fig)
                st.dataframe(df_tri)
            else:
                st.warning("Keine Trigrams verfügbar.")

# -------------------------------------------------------
# Seite 4 – Statistische Analyse (Dokumentation Notebook)
# -------------------------------------------------------
elif page == "4️⃣ Statistische Analyse":
    st.title("4️⃣ Kapitel 4 – Statistical Analysis: Genius Song Lyrics Subset (1%)")

    st.markdown("""
        **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  

        **Genres:**
        - Rap / Hip-Hop
        - Rock
        - Pop
        - R&B (Rhythm and Blues)
        - Country
        - Miscellaneous (verschiedene andere Genres)

        **Purpose:**  
        Statistische Muster in den Songtexten untersuchen:
        - Deskriptive Statistiken (Genre, Text-/Tokenlängen)  
        - Wort-Level-Analyse (Vokabular, Zipf’s Law, Hapax Legomena)  
        - Category Statistics (pro Genre)  
        - N-Gramm-Analyse (Unigrams, Bigrams, Trigrams pro Dataset / Artist / Genre)

        Der folgende Abschnitt dokumentiert das **Jupyter Notebook**.  
        Die eigentlichen Berechnungen und Plots laufen im Notebook und werden als PNG/JSON
        im Ordner `documentation/statistical_analysis` gespeichert.
        """)

    # -----------------------------
    # 1. Dataset Overview
    # -----------------------------
    st.header("1. Dataset Overview")

    st.subheader("1.1 Import Packages and Settings")
    st.markdown("""
        Import der benötigten Bibliotheken für Datenverarbeitung, Statistik,
        Visualisierung und N-Gramm-Analysen.  

        Zusätzlich wird ein Ordner `documentation/statistical_analysis` angelegt,
        in dem alle Grafiken und Kennzahlen gespeichert werden.
        """)
    st.code(
        """import pandas as pd
import re
from collections import defaultdict, Counter
from itertools import tee
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
import os
import ast
import json
#%%
plt.style.use('default')
%matplotlib inline
FIG_DIR = "documentation/statistical_analysis"
os.makedirs(FIG_DIR, exist_ok=True)""",
        language="python",
    )

    st.subheader("1.2 Load Dataset")
    st.markdown("""
        Laden des final bereinigten Datensatzes (`data/clean/data.csv`) und
        Rückkonvertierung der Spalten `words` und `tokens` von String-Repräsentationen
        zu echten Python-Listen (mittels `ast.literal_eval`).
        """)
    st.code(
        """#%% md
## 1.2 Load Dataset
#%%
df = pd.read_csv('data/clean/data.csv')

# convert string representations of lists back into actual Python lists
for col in ["words", "tokens"]:
    if isinstance(df[col].iloc[0], str):
        df[col] = df[col].apply(ast.literal_eval)

print(f"DataFrame shape: {df.shape}")
print(f"Number of Songs: {len(df)} | Artists: {df['artist'].nunique()}")
df.head()""",
        language="python",
    )

    st.subheader("1.3 Descriptive Statistics")
    st.markdown("""
        Zuerst wird die **Genre-Verteilung** analysiert und als Balkendiagramm geplottet.
        Anschließend werden **Text- und Token-Statistiken** berechnet
        (total, min, avg, max) und jeweils als kleine Übersichtsgrafik gespeichert.
        """)
    st.code(
        """#%% md
## 1.3 Descriptive Statistics
Before diving deeper into the analysis, we start with some basic descriptive statistics. First, we examine the genre distribution within the dataset. Then, we analyze characteristics of the lyrics themselves — such as the total number of lyrics and words, as well as the average, minimum, and maximum word counts — both before and after removing stopwords.
#%%
# Genre distribution
print("\\nGENRE DISTRIBUTION")
print("=" * 60)
category_counts = df['tag'].value_counts().sort_values(ascending=False)

for tag,count in category_counts.items():
    pct = (count / len(df)) * 100
    print(f"{tag}: {count:,} songs ({pct:.2f}%)")
#%%
category_counts.plot(kind="bar")
plt.title("Genre Distribution", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "genre_distribution.png"), dpi=150)
plt.show()
plt.close()""",
        language="python",
    )

    st.code(
        """#%%
total_lyrics = len(df)
total_words_raw = df["word_count"].sum()
avg_words_raw = df["word_count"].mean()
min_words_raw = df["word_count"].min()
max_words_raw = df["word_count"].max()

print("TEXT STATISTICS")
print("=" * 60)
print(f"Total lyrics (songs):     {total_lyrics:,}")
print(f"Total words:              {total_words_raw:,}")
print(f"Average words/lyric:      {avg_words_raw:.2f}")
print(f"Shortest lyric:           {min_words_raw} words")
print(f"Longest lyric:            {max_words_raw} words")

# TEXT STATISTICS plot
plt.figure()
plt.bar(["min", "avg", "max"], [min_words_raw, avg_words_raw, max_words_raw])
plt.title("Text Statistics (word_count)", fontweight='bold')
plt.ylabel("Words per lyric")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "text_statistics.png"), dpi=150)
plt.close()""",
        language="python",
    )

    st.code(
        """#%%
tokens_per_row = df["tokens"]
tokens = [t for row in tokens_per_row for t in row]

total_lyrics = len(df)
total_tokens = len(tokens)
unique_tokens = len(set(tokens))
avg_tokens = df["token_count"].mean()
min_tokens = df["token_count"].min()
max_tokens = df["token_count"].max()

print("TOKEN STATISTICS")
print("=" * 60)
print(f"Total lyrics (songs):     {total_lyrics:,}")
print(f"Total tokens:             {total_tokens:,}")
print(f"Unique tokens:            {unique_tokens:,}")
print(f"Average tokens/lyric:     {avg_tokens:.2f}")
print(f"Shortest lyric:           {min_tokens} tokens")
print(f"Longest lyric:            {max_tokens} tokens")

# TOKEN STATISTICS plot
plt.figure()
plt.bar(["min", "avg", "max"], [min_tokens, avg_tokens, max_tokens])
plt.title("Token Statistics (token_count)", fontweight='bold')
plt.ylabel("Tokens per lyric")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "token_statistics.png"), dpi=150)
plt.close()""",
        language="python",
    )

    st.subheader("1.4 Top-15 Wörter vor/nach Stopwörtern")
    st.markdown("""
        Vergleich der **15 häufigsten Wörter** vor und nach Stopwort-Entfernung.
        Die Ergebnisse werden als Doppelbalkendiagramm geplottet und gespeichert.
        """)
    st.code(
        """#%%
top_n = 15
words = [t for row in df["words"] for t in row]
tokens_filtered = [t for row in df["tokens"] for t in row]

word_counts_raw = Counter(words).most_common(top_n)
word_counts_filtered = Counter(tokens_filtered).most_common(top_n)

df_raw = pd.DataFrame(word_counts_raw, columns=["word", "count"])
df_filtered = pd.DataFrame(word_counts_filtered, columns=["word", "count"])


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].bar(df_raw["word"], df_raw["count"])
axes[0].set_title(f"Top {top_n} Words (Before Stopword Removal)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(df_filtered["word"], df_filtered["count"])
axes[1].set_title(f"Top {top_n} Words (After Stopword Removal)", fontsize=12, fontweight='bold')
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
fig.savefig(os.path.join(FIG_DIR, "top15_words_before_after_stopwords.png"), dpi=150)
plt.close(fig)""",
        language="python",
    )

    st.markdown("""
        Die Plots zeigen die 15 häufigsten Wörter vor und nach Stopwort-Entfernung.
        Das häufigste Wort nach dem Filtern erscheint nicht mehr in den ursprünglichen Top-15.
        """)

    # -----------------------------
    # 2. Word-Level Analysis
    # -----------------------------
    st.header("2. Word-Level Analysis")

    st.subheader("2.1 Vocabulary Statistics")
    st.markdown("""
        Bestimmung der Vokabulargröße, Gesamtzahl der Worttokens und Type–Token Ratio (TTR).
        """)
    st.code(
        """#%% md
---
# 2. Word-Level Analysis

Now we take a deeper look at the lyrics and words by analyzing the vocabulary, examining Zipf’s law, identifying rare words (hapax legomena), and exploring various category statistics.

## 2.1 Vocabulary Statistics
#%%
# get all words
all_tokens = [token for tokens in df["words"] for token in tokens]

# count unique words
word_counts = Counter(all_tokens)
vocab_size = len(word_counts)
type_token_ratio = vocab_size / len(all_tokens)

print("VOCABULARY STATISTICS")
print("=" * 60)
print(f"Total word tokens:          {len(all_tokens):,}")
print(f"Unique words (vocabulary):  {vocab_size:,}")
print(f"Type-token ratio:           {type_token_ratio:.4f}")""",
        language="python",
    )

    st.subheader("2.2 Zipf's Law Analysis")
    st.markdown("""
        Analyse der Beziehung zwischen Wortfrequenz und Rang gemäss Zipf’s Law:
        - Fit eines Potenzgesetzes \\( f(r) = C / r^\\alpha \\)  
        - Visualisierung (Log-Log-Plot + Top-30-Vergleich)  
        - Speichern der Parameter in `zipf_stats.json`
        """)
    st.code(
        """#%% md
## 2.2 Zipf's Law Analysis

**Zipf's Law:** In natural language, word frequency is inversely proportional to rank.

Mathematical form: **f(r) = C / r^α**
#%%
all_word_freq = Counter(words).most_common(100)
ranks = list(range(1, len(all_word_freq) + 1))
frequencies = [freq for word, freq in all_word_freq]

# Fit power law model (top 100 words)
log_ranks_100 = np.log(ranks).reshape(-1, 1)
log_freq_100 = np.log(frequencies)

model = LinearRegression()
model.fit(log_ranks_100, log_freq_100)

r_squared = model.score(log_ranks_100, log_freq_100)
slope = model.coef_[0]
intercept = model.intercept_
coefficient_C = np.exp(intercept)

print("ZIPF'S LAW ANALYSIS")
print("=" * 60)
print(f"Fitted equation: f(r) = {coefficient_C:.2f} / r^{abs(slope):.3f}")
print(f"\\nModel parameters:")
print(f"  Slope (α):         {slope:.4f}")
print(f"  R^2 (fit quality): {r_squared:.4f}")
print(f"  Ideal Zipf slope:  -1.0000")
print(f"  Deviation:         {abs(slope + 1.0):.4f}")""",
        language="python",
    )

    st.code(
        """#%%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Log-log plot
axes[0].loglog(ranks, frequencies, 'o', alpha=0.7, color='steelblue', label='Actual')
fitted_zipf = [coefficient_C / (r ** abs(slope)) for r in ranks]
axes[0].loglog(ranks, fitted_zipf, 'g-', linewidth=2, alpha=0.8, label=f'Fitted (α={abs(slope):.3f})')
ideal_zipf = [frequencies[0] / r for r in ranks]
axes[0].loglog(ranks, ideal_zipf, 'r--', linewidth=2, alpha=0.7, label='Ideal Zipf (α=1.0)')
axes[0].set_xlabel('Rank (log scale)')
axes[0].set_ylabel('Frequency (log scale)')
axes[0].set_title("Zipf's Law: Actual vs Fitted vs Ideal", fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Top 30 comparison
n = 30
axes[1].plot(ranks[:n], frequencies[:n], 'o', color='steelblue', label='Actual')
axes[1].plot(ranks[:n], [coefficient_C / (r ** abs(slope)) for r in ranks[:n]],
             'g-', label=f'Fitted (α={abs(slope):.3f})', linewidth=2, alpha=0.7)
axes[1].plot(ranks[:n], [frequencies[0] / r for r in ranks[:n]],
             'r--', label='Ideal Zipf (a=1.0)', linewidth=2, alpha=0.6)
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Top 30 Words: Detailed Comparison', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "zipf_loglog_and_top30.png"), dpi=150)
plt.close(fig)
plt.show()

zipf_stats = {
    "C": float(coefficient_C),
    "alpha": float(slope),
    "r2": float(r_squared),
    "ideal_slope": -1.0,
    "deviation": float(abs(slope + 1.0)),
}

with open(os.path.join(FIG_DIR, "zipf_stats.json"), "w") as f:
    json.dump(zipf_stats, f, indent=2)""",
        language="python",
    )

    st.subheader("2.3 Hapax Legomena (Rare Words)")
    st.markdown("""
        Analyse der seltensten Wörter (Hapax Legomena, Count=1) und aller Wörter mit ≤5 Vorkommen.  
        Zusätzlich wird die Verteilung „Wie viele Wörter kommen X-mal vor?“ als Balkendiagramm gespeichert
        und Kennzahlen in `rare_words_stats.json` geschrieben.
        """)
    st.code(
        """#%% md
## 2.3 Hapax Legomena (Rare Words)

**Hapax legomena** = words appearing only once in the corpus
#%%
# Find hapax legomena
word_counts = Counter(words)
hapax = [word for word, count in word_counts.items() if count == 1]
hapax_pct = (len(hapax) / vocab_size) * 100

# Find words appearing 2-5 times
rare_2 = [word for word, count in word_counts.items() if count == 2]
rare_3_5 = [word for word, count in word_counts.items() if 3 <= count <= 5]
rare_le_5 = len(hapax) + len(rare_2) + len(rare_3_5)

print("RARE WORDS ANALYSIS")
print("=" * 60)
print(f"Hapax legomena (count=1):     {len(hapax):,} words ({hapax_pct:.1f}% of vocab)")
print(f"Words appearing twice:        {len(rare_2):,} words")
print(f"Words appearing 3-5 times:    {len(rare_3_5):,} words")
print(f"\\nTotal rare words (≤5 times):  {rare_le_5:,} words ({(rare_le_5 / vocab_size) * 100:.1f}% of vocab)")
print(f"\\nExamples of hapax legomena:")
print(f"  {hapax[:10]}")""",
        language="python",
    )

    st.code(
        """#%%
freq_distribution = Counter(word_counts.values())
freq_bins = sorted(freq_distribution.keys())[:20]  # First 20 bins
freq_counts = [freq_distribution[f] for f in freq_bins]

plt.figure()
plt.bar(freq_bins, freq_counts, color='steelblue', edgecolor='black')
plt.xlabel('Word Frequency')
plt.ylabel('Number of Words')
plt.title('Distribution: How Many Words Appear X Times?', fontweight='bold')
plt.xticks(freq_bins)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rare_words_distribution.png"), dpi=150)
plt.close()
plt.show()

rare_stats = {
    "hapax_count": len(hapax),
    "hapax_pct": hapax_pct,
    "rare_le_5": rare_le_5,
    "rare_le_5_pct": (rare_le_5 / vocab_size) * 100,
    "example_hapax": hapax[:10],
}

with open(os.path.join(FIG_DIR, "rare_words_stats.json"), "w") as f:
    json.dump(rare_stats, f, indent=2)""",
        language="python",
    )

    st.subheader("2.4 Category Statistics")
    st.markdown("""
        Berechnung von Kennzahlen **pro Genre** (Tag):  
        - Anzahl Songs  
        - Gesamt- und Durchschnittswörter  
        - Vokabulargröße  
        - Anteil Songs mit Zahlen im Text  

        Die Ergebnisse werden als Dreifach-Balkenplot gespeichert (`category_statistics.png`).
        """)
    st.code(
        """#%% md
## 2.4 Category Statistics
#%%
categories = df['tag'].unique()

category_stats = {}
for cat in categories:
    cat_df = df[df['tag'] == cat]
    cat_text = ' '.join(cat_df['lyrics'].str.lower())
    cat_words = cat_text.split()
    cat_vocab = len(set(cat_words))

    has_number_pct = sum(any(char.isdigit() for char in lyric) for lyric in cat_df['lyrics']) / len(cat_df) * 100

    category_stats[cat] = {
        'songs': len(cat_df),
        'total_words': len(cat_words),
        'avg_words': len(cat_words) / len(cat_df),
        'vocab': cat_vocab,
        'has_number_pct': has_number_pct
    }
#%%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

avg_words_cat = [category_stats[cat]['avg_words'] for cat in categories]
axes[0].bar(categories, avg_words_cat, color='steelblue')
axes[0].set_title('Average Words per Song', fontweight='bold')
axes[0].set_ylabel('Words')
axes[0].set_xticklabels(categories, rotation=45)

vocab_sizes = [category_stats[cat]['vocab'] for cat in categories]
axes[1].bar(categories, vocab_sizes, color='coral')
axes[1].set_title('Vocabulary Size per Genre', fontweight='bold')
axes[1].set_ylabel('Unique Words')
axes[1].set_xticklabels(categories, rotation=45)

has_number_pct = [category_stats[cat]['has_number_pct'] for cat in categories]
axes[2].bar(categories, has_number_pct, color='lightgreen')
axes[2].set_title('Songs Containing Numbers (%)', fontweight='bold')
axes[2].set_ylabel('Percentage')
axes[2].set_xticklabels(categories, rotation=45)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "category_statistics.png"), dpi=150)
plt.show()""",
        language="python",
    )

    # -----------------------------
    # 3. N-gram Analysis
    # -----------------------------
    st.header("3. N-gram Analysis")

    st.subheader("3.1 Unigram, Bigram, Trigram")
    st.markdown("""
        Erstellung von Unigrams, Bigrams und Trigrams über alle Songs hinweg,
        Ausgabe der Top-15 pro N-Gramm-Typ und Speicherung als Plot `top15_ngrams.png`.
        """)
    st.code(
        """#%% md
---
# 3. N-gram Analysis

In this chapter, we analyze n-grams — unigrams, bigrams, and trigrams — to explore common word patterns and recurring phrases.
## 3.1 Unigram, Bigram, Trigram
#%%
def ngrams(tokens, n):
    \"\"\"generate n-grams\"\"\"
    if n <= 0:
        return []
    iters = tee(tokens, n)
    for i, it in enumerate(iters):
        for _ in range(i):
            next(it, None)
    return zip(*iters)
#%%
unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for tokens in df["tokens"]:
    unigram_counts.update(tokens)
    bigram_counts.update(ngrams(tokens, 2))
    trigram_counts.update(ngrams(tokens, 3))

top_unigrams = pd.DataFrame(unigram_counts.most_common(15), columns=["word", "count"])

top_bigrams = pd.DataFrame(
    [(" ".join(k), v) for k, v in bigram_counts.most_common(15)],
    columns=["bigram", "count"]
)
top_trigrams = pd.DataFrame(
    [(" ".join(k), v) for k, v in trigram_counts.most_common(15)],
    columns=["trigram", "count"]
)
#%%
print("TOP 15 UNIGRAMS:")
print("=" * 60)
print(top_unigrams)
#%%
print("TOP 15 BIGRAMS:")
print("=" * 60)
print(top_bigrams)
#%%
print("TOP 15 TRIGRAMS:")
print("=" * 60)
print(top_trigrams)
#%%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].barh(top_unigrams["word"][::-1], top_unigrams["count"][::-1])
axes[0].set_title("Top 15 Unigrams", fontweight='bold')
axes[0].set_xlabel("Frequency")

axes[1].barh(top_bigrams["bigram"][::-1], top_bigrams["count"][::-1])
axes[1].set_title("Top 15 Bigrams", fontweight='bold')
axes[1].set_xlabel("Frequency")

axes[2].barh(top_trigrams["trigram"][::-1], top_trigrams["count"][::-1])
axes[2].set_title("Top 15 Trigrams", fontweight='bold')
axes[2].set_xlabel("Frequency")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "top15_ngrams.png"), dpi=150)
plt.show()""",
        language="python",
    )

    st.subheader("3.2 N-Grams per Artist/Genre")
    st.markdown("""
        Für jede Gruppe (Artist / Genre) wird das jeweils häufigste N-Gramm (Uni/ Bi/ Trigram)
        bestimmt und die Top-20 bzw. Top-Listen visualisiert.  
        Die Resultate werden als `top20_ngrams_per_artist.png` und `top_ngrams_per_genre.png` gespeichert.
        """)
    st.code(
        """#%% md
## 3.2 N-Grams per Artist/Genre
#%%
def most_common_ngram_for_group(group_df: pd.DataFrame, label_col: str, n: int) -> pd.DataFrame:
    \"""
    returns, for each group (artist/tag), the most frequent n-gram along with its count.
    Columns: [label_col, 'ngram', 'count', 'songs']
    \"""
    rows = []
    for label, sub in group_df.groupby(label_col):
        c = Counter()
        for toks in sub["tokens"]:
            c.update(ngrams(toks, n))
        if c:
            top_ngram, cnt = c.most_common(1)[0]
            rows.append({label_col: label, "ngram": " ".join(top_ngram), "count": cnt, "songs": len(sub)})
        else:
            rows.append({label_col: label, "ngram": None, "count": 0, "songs": len(sub)})
    return pd.DataFrame(rows).sort_values([label_col]).reset_index(drop=True)
#%%
top_unigrams_artist = most_common_ngram_for_group(df, "artist", n=1).sort_values("count", ascending=False).head(20)
top_bigrams_artist  = most_common_ngram_for_group(df, "artist", n=2).sort_values("count", ascending=False).head(20)
top_trigrams_artist = most_common_ngram_for_group(df, "artist", n=3).sort_values("count", ascending=False).head(20)
#%%
# create labels
top_unigrams_artist["label"] = top_unigrams_artist["artist"] + " - " + top_unigrams_artist["ngram"]
top_bigrams_artist["label"]  = top_bigrams_artist["artist"]  + " - " + top_bigrams_artist["ngram"]
top_trigrams_artist["label"] = top_trigrams_artist["artist"] + " - " + top_trigrams_artist["ngram"]


fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].barh(top_unigrams_artist["label"][::-1], top_unigrams_artist["count"][::-1])
axes[0].set_title("Top 20 Unigrams per Artist", fontweight='bold')
axes[0].set_xlabel("Frequency")

axes[1].barh(top_bigrams_artist["label"][::-1], top_bigrams_artist["count"][::-1])
axes[1].set_title("Top 20 Bigrams per Artist", fontweight='bold')
axes[1].set_xlabel("Frequency")

axes[2].barh(top_trigrams_artist["label"][::-1], top_trigrams_artist["count"][::-1])
axes[2].set_title("Top 20 Trigrams per Artist", fontweight='bold')
axes[2].set_xlabel("Frequency")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "top20_ngrams_per_artist.png"), dpi=150)
plt.show()
#%%
top_unigrams_genre = most_common_ngram_for_group(df, "tag", n=1).sort_values("count", ascending=False)
top_bigrams_genre  = most_common_ngram_for_group(df, "tag", n=2).sort_values("count", ascending=False)
top_trigrams_genre = most_common_ngram_for_group(df, "tag", n=3).sort_values("count", ascending=False)
#%%
# create labels
top_unigrams_genre["label"] = top_unigrams_genre["tag"] + " - " + top_unigrams_genre["ngram"]
top_bigrams_genre["label"]  = top_bigrams_genre["tag"]  + " - " + top_bigrams_genre["ngram"]
top_trigrams_genre["label"] = top_trigrams_genre["tag"] + " - " + top_trigrams_genre["ngram"]


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].barh(top_unigrams_genre["label"][::-1], top_unigrams_genre["count"][::-1])
axes[0].set_title("Top Unigrams per Genre", fontweight='bold')
axes[0].set_xlabel("Frequency")

axes[1].barh(top_bigrams_genre["label"][::-1], top_bigrams_genre["count"][::-1])
axes[1].set_title("Top Bigrams per Genre", fontweight='bold')
axes[1].set_xlabel("Frequency")

axes[2].barh(top_trigrams_genre["label"][::-1], top_trigrams_genre["count"][::-1])
axes[2].set_title("Top Trigrams per Genre", fontweight='bold')
axes[2].set_xlabel("Frequency")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "top_ngrams_per_genre.png"), dpi=150)
plt.show()""",
        language="python",
    )

    st.info(
        "Dieser Abschnitt dokumentiert vollständig den Statistical-Analysis-Workflow im Notebook. "
        "Die erzeugten PNG- und JSON-Dateien werden anschließend in der App (Tabs) nur noch geladen "
        "und angezeigt – ohne erneute Berechnung."
    )

    st.markdown("Alle Resultate wurden im Notebook berechnet und als Grafiken gespeichert.")

    # =================================================== #
    # 4. NOTEBOOK-RESULTATE → TABS
    # =================================================== #

    FIG_DIR = "documentation/statistical_analysis"

    (
        tab_genre,
        tab_text,
        tab_token,
        tab_top15,
        tab_zipf,
        tab_rare,
        tab_category,
        tab_ngrams,
        tab_artist,
        tab_genres,
    ) = st.tabs([
        "GENRE DISTRIBUTION",
        "TEXT STATISTICS",
        "TOKEN STATISTICS",
        "TOP 15 WORDS",
        "ZIPF'S LAW ANALYSIS",
        "RARE WORDS",
        "CATEGORY STATISTICS",
        "TOP 15 N-GRAMS",
        "N-GRAMS PER ARTIST",
        "N-GRAMS PER GENRE",
    ])

    # --------------------------------------------------------------------------
    # GENRE DISTRIBUTION
    with tab_genre:
        st.subheader("GENRE DISTRIBUTION")
        img = os.path.join(FIG_DIR, "genre_distribution.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # TEXT STATISTICS
    with tab_text:
        st.subheader("TEXT STATISTICS")
        img = os.path.join(FIG_DIR, "text_statistics.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # TOKEN STATISTICS
    with tab_token:
        st.subheader("TOKEN STATISTICS")
        img = os.path.join(FIG_DIR, "token_statistics.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # TOP 15 WORDS BEFORE/AFTER STOPWORDS
    with tab_top15:
        st.subheader("TOP 15 WORDS – BEFORE & AFTER STOPWORDS")
        img = os.path.join(FIG_DIR, "top15_words_before_after_stopwords.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # ZIPF’S LAW ANALYSIS
    with tab_zipf:
        st.subheader("ZIPF’S LAW ANALYSIS")

        # stats JSON
        stats = os.path.join(FIG_DIR, "zipf_stats.json")
        if os.path.exists(stats):
            import json
            with open(stats, "r") as f:
                z = json.load(f)
            st.code(
f"""ZIPF'S LAW ANALYSIS
============================================================
Fitted equation: f(r) = {z['C']:.2f} / r^{z['alpha']:.3f}
Slope (α):         {z['alpha']:.4f}
R^2 (fit quality): {z['r2']:.4f}
Ideal slope:       -1.0
Deviation:         {z['deviation']:.4f}""",
                language="text",
            )

        img = os.path.join(FIG_DIR, "zipf_loglog_and_top30.png")
        if os.path.exists(img):
            st.image(img, caption="Zipf Log-Log + Top-30 Comparison", use_column_width=200)

    # --------------------------------------------------------------------------
    # RARE WORDS ANALYSIS
    with tab_rare:
        st.subheader("RARE WORDS ANALYSIS")

        stats = os.path.join(FIG_DIR, "rare_words_stats.json")
        if os.path.exists(stats):
            with open(stats, "r") as f:
                r = json.load(f)
            st.code(
f"""RARE WORDS ANALYSIS
============================================================
Hapax legomena:         {r['hapax_count']:,}  ({r['hapax_pct']:.1f}% vocab)
Rare words ≤5 times:    {r['rare_le_5']:,}  ({r['rare_le_5_pct']:.1f}% vocab)
Examples:
  {r['example_hapax']}""",
                language="text",
            )

        img = os.path.join(FIG_DIR, "rare_words_distribution.png")
        if os.path.exists(img):
            st.image(img, caption="Word Frequency Distribution", use_column_width=200)

    # --------------------------------------------------------------------------
    # CATEGORY STATISTICS
    with tab_category:
        st.subheader("CATEGORY STATISTICS")
        img = os.path.join(FIG_DIR, "category_statistics.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # TOP 15 N-GRAMS
    with tab_ngrams:
        st.subheader("TOP 15 N-GRAMS")
        img = os.path.join(FIG_DIR, "top15_ngrams.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # TOP N-GRAMS PER ARTIST
    with tab_artist:
        st.subheader("TOP N-GRAMS PER ARTIST")
        img = os.path.join(FIG_DIR, "top20_ngrams_per_artist.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)

    # --------------------------------------------------------------------------
    # TOP N-GRAMS PER GENRE
    with tab_genres:
        st.subheader("TOP N-GRAMS PER GENRE")
        img = os.path.join(FIG_DIR, "top_ngrams_per_genre.png")
        if os.path.exists(img):
            st.image(img, use_column_width=200)



# -------------------------------------------------------
# Seite 5 – Word Embedding (Dokumentation + Resultate)
# -------------------------------------------------------
elif page == "5️⃣ Word Embedding":
    st.title("5️⃣ Kapitel 5 – Word Embedding: Genius Song Lyrics Subset (1%)")

    st.markdown("""
    **Purpose:**  
    Erstellung und Exploration von **Word Embeddings** für Songtexte.  

    Auf Basis der tokenisierten Lyrics werden:
    - ein **Word2Vec-Modell** trainiert,
    - semantische Beziehungen zwischen Wörtern untersucht,
    - Songtexte über TF-IDF + Word2Vec auf **Dokument-Embeddings** abgebildet
    - und diese im Embedding-Space analysiert.

    Wie in den vorherigen Kapiteln:
    - Das **Notebook** übernimmt Training & schwere Berechnungen  
    - Die **Streamlit-App** dokumentiert nur den Code und lädt am Ende die fertigen Ergebnisse.
    """)

    # =========================
    # 1. Train Model – DOKU
    # =========================
    st.header("1. Train Model")

    st.subheader("1.1 Import & Setup")
    st.markdown("""
    Import der benötigten Bibliotheken für:
    - Datenverarbeitung (`pandas`, `numpy`),
    - Word2Vec-Training (`gensim`),
    - Visualisierung (`matplotlib`, `plotly`),
    - Dimensionalitätsreduktion (`PCA`),
    - TF-IDF und Ähnlichkeitsberechnungen (`sklearn`),
    - sowie Hilfsfunktionen (`ast`, `os`).
    """)
    st.code(
        """import pandas as pd
import os
import re
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity""",
        language="python",
    )

    st.subheader("1.2 Load and prepare Data")
    st.markdown("""
    Relevante Spalten im Datensatz:
    - `language_cld3` – erkannte Sprache (z.B. `en` für Englisch)
    - `tokens` – Liste der Wörter eines Songs  

    Es werden **nur englische Songs** verwendet und falls `tokens` als String gespeichert sind,
    wieder in echte Python-Listen konvertiert.
    """)
    st.code(
        """#%% md
**Relevante Spalten:**
- `language_cld3` — erkannte Sprache (typisch ISO-Code, z. B. `en` für Englisch)
- `tokens` — Liste der Token (Wörter) der Lyrics pro Zeile

**Hinweis:** Für Word2Vec müssen es **Listen von Strings** sein.
#%%
df = pd.read_csv("data/clean/data.csv")

# Nur englische Songs
df = df[df["language_cld3"] == "en"]

# Tokens von String-Repräsentation zurück in Listen konvertieren
if isinstance(df["tokens"].iloc[0], str):
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

# Eingabedaten für Word2Vec: Liste von Token-Listen
sentences = df["tokens"].dropna().tolist()""",
        language="python",
    )

    st.subheader("1.3 Train Word2Vec Model")
    st.markdown("""
    **Ziel:** Lernen von Wortvektoren aus den Lyrics-Token mit `gensim.Word2Vec`.

    Wichtige Parameter:
    - `vector_size=50` → 50-dimensionale Embeddings (kompakt, schnell)
    - `window=5` → Kontextfenster von 5 Wörtern links/rechts
    - `min_count=2` → Wörter mit weniger als 2 Vorkommen werden ignoriert
    - `epochs=100` → 100 Trainingsdurchläufe für stabilere Vektoren
    """)
    st.code(
        """#%% md
## 1.2 Train Word2Vec Model

**Ziel:** Lernen von Wortvektoren aus den Lyrics-Token.
**Bibliothek:** `gensim.models.Word2Vec`

### Wichtige Parameter

| Parameter       | Bedeutung |
|----------------|-----------|
| `sentences`    | Liste von Wortlisten (Token pro Song) |
| `vector_size`  | Dimension der Vektoren |
| `window`       | Kontextfenstergröße |
| `min_count`    | Minimalhäufigkeit für Wörter |
| `workers`      | Anzahl Threads |
| `epochs`       | Trainingsdurchläufe |
#%%
model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=2,
    workers=4,
    epochs=100
)

print("Model trained!")
print("Vocabulary size:", len(model.wv))
print("Vector size:", model.wv.vector_size)""",
        language="python",
    )

    st.markdown("""
    **Ergebnis:**  
    Ein trainiertes Word2Vec-Modell, das jedes Wort als Punkt in einem **50-dimensionalen Raum**
    repräsentiert – Wörter mit ähnlichem Kontext liegen nah beieinander.
    """)

    # =========================
    # 2. Explore Embedding Space – DOKU
    # =========================
    st.header("2. Explore Embedding Space")

    st.subheader("2.1 Similar Words & Kontext – Beispiel: 'love'")
    st.markdown("""
    Zuerst werden zu einem Wort (z.B. `love`) die ähnlichsten Wörter
    per **Kosinus-Ähnlichkeit** gesucht (`model.wv.most_similar`),  
    anschließend für Testwörter (`baby`, `love`, `happy`) grafisch dargestellt.
    """)
    st.code(
        """#%% md
## 2.1 Embedding Examples: 'love'
#%%
word = "love"
if word in model.wv:
    print(f"\\nVector for '{word}':", model.wv[word][:10])
    print("\\nMost similar words to 'love':")
    print(model.wv.most_similar(word, topn=5))
else:
    print(f"'{word}' not in vocabulary.")
#%%
def find_similar_words(word, model, top_n=5):
    if word not in model.wv:
        return None, None
    similar = model.wv.most_similar(word, topn=top_n)
    words = [w for w, _ in similar]
    scores = [s for _, s in similar]
    return words, scores


test_words = ["baby", "love", "happy"]

# Store results for plotting in next cell
similar_results = {}

for word in test_words:
    print(f"\\n🔍 Finding words similar to '{word}'...")
    words, scores = find_similar_words(word, model, top_n=5)

    if words:
        similar_results[word] = (words, scores)

        for w, s in zip(words, scores):
            bar = '█' * int(s * 10)
            print(f"  {w:10} {bar} {s:.2f}")
    else:
        similar_results[word] = None
        print(f"  '{word}' not in vocabulary.")
#%%
fig, axes = plt.subplots(1, len(test_words), figsize=(5 * len(test_words), 4))

# Handle single-axis case
if len(test_words) == 1:
    axes = [axes]

for ax, word in zip(axes, test_words):
    result = similar_results[word]

    if result:
        words, scores = result
        ax.barh(words, scores)
        ax.set_title(f"'{word}'")
        ax.set_xlabel("Similarity")
        ax.invert_yaxis()
    else:
        ax.set_visible(False)

fig.suptitle("Most Similar Words", fontsize=14)
fig.tight_layout()
plt.show()""",
        language="python",
    )

    st.markdown("""
    Interpretation:  
    - `happy` – `sad` können hohe Ähnlichkeit haben, obwohl sie Gegenteile sind,  
      weil Word2Vec **Kontextähnlichkeit**, nicht logische Gegensätze lernt.  
    - `happy` – `happier` → gleiche Wortfamilie, grammatische Variante.  
    - `love` – `loving` / `loves` → gleicher semantischer Kern, leicht anderer Kontext.
    """)

    st.subheader("2.1b 3D-Visualisierung der Nachbarn von 'love'")
    st.code(
        """#%%
def explore_similar_words(word, model, top_n=10):

    if word not in model.wv:
        raise ValueError(f"'{word}' not in model vocabulary.")

    similar = model.wv.most_similar(word, topn=top_n)
    words = [word] + [w for w, _ in similar]
    scores = [1.0] + [s for _, s in similar]

    vectors = np.array([model.wv[w] for w in words])

    # PCA auf 3 Dimensionen reduzieren (falls nötig)
    if vectors.shape[1] > 3:
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)
        axis_titles = ("PC1", "PC2", "PC3")
    else:
        vectors_3d = vectors
        axis_titles = tuple(f"Dim{i+1}" for i in range(vectors.shape[1]))

    sizes = np.array(scores) * 25
    colors = np.array(scores)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vectors_3d[:, 0],
                y=vectors_3d[:, 1],
                z=vectors_3d[:, 2] if vectors_3d.shape[1] > 2 else np.zeros(len(words)),
                mode="markers+text",
                text=words,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale="viridis",
                    showscale=True,
                    colorbar=dict(title="Similarity"),
                    line=dict(width=1, color="black"),
                    symbol=["diamond"] + ["circle"] * (len(words) - 1)
                ),
                hovertemplate="<b>%{text}</b><br>Similarity: %{marker.color:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"Similar Words to '{word}' (Top {top_n}) — Color/Size = Similarity",
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig

fig = explore_similar_words("love", model, top_n=30)
fig.show()""",
        language="python",
    )

    st.markdown("""
    Die 3D-Darstellung zeigt **Cluster** um das Wort *love*:
    - Zentrum: *love, loving, loves*  
    - Rand: emotionale Substantive (*baby, babe, heart*)  
    - Abstrakte Begriffe (*forever, always, never*) usw.
    """)

    st.subheader("2.2 Globaler Word-Embedding Space")
    st.code(
        """#%% md
## 2.2 Embedding Space
#%%
def explore_embedding_space(model, n_words=30):
    vocab = list(model.wv.index_to_key)
    if len(vocab) == 0:
        raise ValueError("The model has an empty vocabulary.")
    n = min(n_words, len(vocab))
    words = vocab[:n]

    vectors = np.array([model.wv[w] for w in words])

    if vectors.shape[1] > 3:
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)
        axis_titles = ("PC1", "PC2", "PC3")
    else:
        vectors_3d = vectors
        axis_titles = tuple(f"Dim{i+1}" for i in range(vectors.shape[1]))

    distances = np.linalg.norm(vectors_3d, axis=1)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vectors_3d[:, 0],
                y=vectors_3d[:, 1] if vectors_3d.shape[1] > 1 else np.zeros_like(distances),
                z=vectors_3d[:, 2] if vectors_3d.shape[1] > 2 else np.zeros_like(distances),
                mode="markers+text",
                text=words,
                textposition="top center",
                marker=dict(
                    size=np.clip(distances * 3, 4, 24),
                    color=distances,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Distance"),
                ),
                hovertemplate="<b>%{text}</b><br>Distance: %{marker.color:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"Explore the Word Space (Top {n} words) — Size/Color = Distance from Origin",
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1] if len(axis_titles) > 1 else "",
            zaxis_title=axis_titles[2] if len(axis_titles) > 2 else "",
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig

fig = explore_embedding_space(model, n_words=50)
fig.show()""",
        language="python",
    )

    st.markdown("""
    Hier werden viele Wörter gleichzeitig im 3D-Raum gezeigt.  
    Größe & Farbe codieren den Abstand vom Ursprung → „markantere“ Wörter stechen hervor.
    """)

    # =========================
    # 3. TF-IDF + Dokument-Embeddings – DOKU
    # =========================
    st.header("3. TF-IDF & Dokument-Embeddings")

    st.markdown("""
    Word2Vec liefert **Wortvektoren** – um ganze Songs zu repräsentieren,
    werden die Wortvektoren mit **TF-IDF gewichtet** gemittelt.
    """)
    st.code(
        """#%% md
# 3. TF-IDF
#%%
tfidf_vect = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    lowercase=False
)

X_tfidf = tfidf_vect.fit_transform(df["tokens"])
terms = tfidf_vect.get_feature_names_out()

dim = model.wv.vector_size
doc_emb_tfidf = np.zeros((X_tfidf.shape[0], dim), dtype=np.float32)

for i in range(X_tfidf.shape[0]):
    row = X_tfidf[i]
    if row.nnz == 0:
        continue
    idxs = row.indices
    wts = row.data
    vecs = []
    w = []
    for j, wt in zip(idxs, wts):
        term = terms[j]
        if term in model.wv:
            vecs.append(model.wv[term])
            w.append(wt)
    if w:
        vecs = np.vstack(vecs)
        w = np.asarray(w, dtype=np.float32)
        doc_emb_tfidf[i] = (vecs * w[:, None]).sum(axis=0) / (w.sum() + 1e-9)

print("TF-IDF-Embeddings:", doc_emb_tfidf.shape)

# Optional: Nullvektoren rausfiltern
keep = np.linalg.norm(doc_emb_tfidf, axis=1) > 0
df_use = df.reset_index(drop=True).loc[keep].reset_index(drop=True)
emb_use = doc_emb_tfidf[keep]
print("Nach Filter:", emb_use.shape)""",
        language="python",
    )

    st.markdown("""
    Ergebnis: pro Song ein Embedding-Vektor, der beide Welten kombiniert:
    - **Word2Vec** (semantische Struktur)  
    - **TF-IDF** (Gewichtung wichtiger Wörter)
    """)

    # =========================
    # 4. Embedding of whole songs – DOKU
    # =========================
    st.header("4. Embedding of whole songs")

    st.markdown("""
    Alternative: Song-Embeddings als einfacher Durchschnitt der Wortvektoren (`get_song_vector`),
    anschließend Visualisierung im 3D-Raum und per PCA.
    """)

    st.code(
        '''#%% md
# 4. Embedding of whole songs
#%%
GENRE_COL = "tag"

def get_song_vector(tokens, w2v_model):
    """
    Compute a single vector representation for one song by
    averaging all word vectors for its tokens.
    If no token is in the vocabulary, return a zero vector.
    """
    if not isinstance(tokens, (list, tuple)):
        return np.zeros(w2v_model.vector_size, dtype=np.float32)

    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]

    if not vectors:
        return np.zeros(w2v_model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)

# Nur Songs mit Tokens & Genre
df_songs = df.dropna(subset=["tokens", GENRE_COL]).copy()

# Embedding pro Song
df_songs["embedding"] = df_songs["tokens"].apply(
    lambda toks: get_song_vector(toks, model)
)

X = np.vstack(df_songs["embedding"].values)        # (n_songs, embedding_dim)
y = df_songs[GENRE_COL].astype(str).values         # (n_songs,)

print("Song embeddings shape:", X.shape)
print("Number of songs:", len(y))
print("Example genres:", y[:10])
''',
        language="python",
    )

    st.code(
        '''#%%
def explore_doc_space(embeddings, labels=None, n_max=None):
    """
    Interaktiver 3D-Plot von Dokument-/Song-Embeddings.
    Nutzt ggf. PCA auf 3 Dimensionen und färbt/skalieren nach Distanz vom Ursprung.
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Keine Embeddings übergeben.")

    X = np.asarray(embeddings)
    if n_max is not None:
        X = X[:n_max]
        if labels is not None:
            labels = labels[:n_max]

    # Auf 3 Dimensionen reduzieren (falls nötig)
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X3 = pca.fit_transform(X)
        axis_titles = ("PC1", "PC2", "PC3")
    else:
        X3 = X
        axis_titles = tuple(f"Dim{i+1}" for i in range(X.shape[1]))
        # ggf. auf 3 Achsen auffüllen
        if X3.shape[1] == 1:
            X3 = np.hstack([X3, np.zeros((X3.shape[0], 2))])
        elif X3.shape[1] == 2:
            X3 = np.hstack([X3, np.zeros((X3.shape[0], 1))])

    distances = np.linalg.norm(X3, axis=1)

    if labels is None:
        labels = [f"Doc {i}" for i in range(X3.shape[0])]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X3[:, 0],
                y=X3[:, 1],
                z=X3[:, 2],
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=np.clip(distances * 3, 4, 24),
                    color=distances,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Distance"),
                    opacity=0.9,
                ),
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Explore the Song Space — Size/Color = Distance from Origin",
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1] if len(axis_titles) > 1 else "",
            zaxis_title=axis_titles[2] if len(axis_titles) > 2 else "",
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

embeddings = X

# Labels: Titel oder Artist oder Tag
df_use = df_songs.copy()
labels = None
for col in ["title", "artist", "tag"]:
    if col in df_use.columns:
        labels = df_use[col].astype(str).tolist()
        break

if labels is None:
    labels = df_use.index.astype(str).tolist()

fig_docs = explore_doc_space(embeddings, labels=labels, n_max=40)
fig_docs.show()

# PCA 3D nach Genre
pca = PCA(n_components=3, random_state=42)
X_3d = pca.fit_transform(X)

df_plot = pd.DataFrame({
    "pc1": X_3d[:, 0],
    "pc2": X_3d[:, 1],
    "pc3": X_3d[:, 2],
    "genre": y.astype(str)
})

viridis_256 = px.colors.sample_colorscale("Viridis", np.linspace(0, 1, 6))

fig = px.scatter_3d(
    df_plot,
    x="pc1",
    y="pc2",
    z="pc3",
    color="genre",
    opacity=0.3,
    title="Songs in Word2Vec embedding space (PCA 3D)",
    color_discrete_sequence=viridis_256
)

fig.update_traces(marker=dict(size=4))
fig.show()
''',
        language="python",
    )

    # =========================
    # 5. Save Model – DOKU
    # =========================
    st.header("5. Save Model & Features")

    st.markdown("""
    Zum Schluss werden alle wichtigen Artefakte für die App und spätere Modelle gespeichert:
    - `data/features/song_embeddings.npy` – Song-Embedding-Matrix `X`  
    - `data/features/song_labels.npy` – Genre-Labels `y`  
    - `data/features/song_metadata.csv` – Metadaten (Genre, Titel, Artist, …)  
    - `models/word2vec_lyrics.model` – trainiertes Word2Vec-Modell
    """)

    st.code(
        """#%% md
# 5. Save Model
#%%
# 1) Save feature matrix X and label vector y
os.makedirs("data/features", exist_ok=True)

np.save("data/features/song_embeddings.npy", X)
np.save("data/features/song_labels.npy", y)

print("Saved song embeddings and labels to 'data/features/'")

# 2) Save metadata (optional but very useful)
meta_cols = [GENRE_COL]
for col in ["title", "artist", "id", "song_id"]:
    if col in df_songs.columns:
        meta_cols.append(col)

df_songs[meta_cols].to_csv("data/features/song_metadata.csv", index=False)
print("Saved song metadata to 'data/features/song_metadata.csv'")

# 3) Save the trained Word2Vec model
os.makedirs("models", exist_ok=True)
model.save("models/word2vec_lyrics.model")
print("Saved Word2Vec model to 'models/word2vec_lyrics.model'")""",
        language="python",
    )

    st.info(
        "Oben ist der komplette Notebook-Workflow dokumentiert. "
        "Im nächsten Abschnitt werden nur die fertig berechneten Artefakte geladen – "
        "kein erneutes Training in der Streamlit-App."
    )

    # =================================================== #
    # 📁 NOTEBOOK-RESULTATE – Word Embeddings
    # =================================================== #
    st.header("📁 Notebook-Resultate – Word Embeddings")

    import os
    import numpy as np
    import pandas as pd

    MODEL_PATH = "models/word2vec_lyrics.model"
    EMB_PATH = "data/features/song_embeddings.npy"
    LABEL_PATH = "data/features/song_labels.npy"
    META_PATH = "data/features/song_metadata.csv"

    # 5.1 Word2Vec-Modell laden & interaktiv ähnliche Wörter anzeigen
    st.subheader("🔤 Word2Vec-Modell & ähnliche Wörter")

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Word2Vec-Modell nicht gefunden: `{MODEL_PATH}`.\n"
            "Bitte zuerst das Word-Embedding-Notebook ausführen."
        )
    else:
        from gensim.models import Word2Vec

        model = Word2Vec.load(MODEL_PATH)

        st.markdown(
            f"- **Vocabulary size:** {len(model.wv):,}  \n"
            f"- **Vector size:** {model.wv.vector_size}"
        )

        default_word = "love"
        query_word = st.text_input("Wort für Ähnlichkeits-Suche:", value=default_word)

        if query_word.strip():
            if query_word in model.wv:
                similar = model.wv.most_similar(query_word, topn=10)
                st.markdown(f"**Top 10 ähnliche Wörter zu** `{query_word}`:")
                sim_df = pd.DataFrame(similar, columns=["word", "similarity"])
                st.dataframe(sim_df)
            else:
                st.warning(f"`{query_word}` ist nicht im Vokabular des Modells enthalten.")

    st.markdown("---")

    # 5.2 Song-Embeddings laden & kurz visualisieren
    st.subheader("🧱 Song-Embeddings (Dokument-Vektoren)")

    if not (os.path.exists(EMB_PATH) and os.path.exists(LABEL_PATH)):
        st.error(
            f"Song-Embeddings oder Label-Datei nicht gefunden: "
            f"`{EMB_PATH}` / `{LABEL_PATH}`.\n"
            "Bitte das Notebook bis zum Speicherschritt ausführen."
        )
    else:
        X = np.load(EMB_PATH)
        y = np.load(LABEL_PATH, allow_pickle=True)

        st.markdown(
            f"- **Embedding-Matrix X:** {X.shape[0]:,} Songs × {X.shape[1]} Dimensionen  \n"
            f"- **Anzahl Labels:** {len(y):,}"
        )

        if os.path.exists(META_PATH):
            df_meta = pd.read_csv(META_PATH)
            st.subheader("📋 Beispiel-Metadaten")
            st.dataframe(df_meta.head())
        else:
            df_meta = None

        # Kleine 2D-PCA-Visualisierung
        try:
            from sklearn.decomposition import PCA
            import plotly.express as px

            st.subheader("📉 PCA 2D – Song-Embedding Space (Ausschnitt)")

            n = min(500, X.shape[0])
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X[:n])

            if df_meta is not None and "tag" in df_meta.columns:
                genres = df_meta["tag"].astype(str).values[:n]
            else:
                genres = y[:n].astype(str)

            df_plot = pd.DataFrame(
                {"pc1": coords[:, 0], "pc2": coords[:, 1], "genre": genres}
            )

            fig = px.scatter(
                df_plot,
                x="pc1",
                y="pc2",
                color="genre",
                opacity=0.5,
                title="Songs im Embedding-Space (PCA 2D, Ausschnitt)",
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"PCA-Visualisierung konnte nicht erstellt werden: {e}")

    st.info(
        "Alle oben gezeigten Resultate basieren auf den im Notebook gespeicherten Artefakten "
        "(`models/word2vec_lyrics.model`, `data/features/song_embeddings.npy`, ...). "
        "Die Streamlit-App lädt sie nur und visualisiert sie – es wird **kein Modell in der App neu trainiert**."
    )

# -------------------------------------------------------
# Seite 6 – Model Evaluation
# -------------------------------------------------------
elif page == "6️⃣ Model Evaluation":
    import os
    import json
    import numpy as np
    import pandas as pd

    st.title("6️⃣ Model Evaluation – Vergleich der Modelle")

    st.markdown(
        """
        **Model Evaluation: Genius Song Lyrics Subset (1%)**  
        Dataset: 34'049 Songs | 26'408 Artists | 6 Genres  

        **Embeddings:**  
        - Word2Vec (self-trained)  
        - TF-IDF (character-level n-grams)  
        - SentenceTransformer (MiniLM)  

        **Classifier:**  
        - LinearSVC  
        - Logistic Regression  
        - Random Forest  

        **Purpose:**  
        Mehrere Modelle zur automatischen Genre-Klassifikation vergleichen – basierend auf
        unterschiedlichen Textrepräsentationen (Embeddings) und Klassifikatoren.  

        Ausgewertet werden:  
        - Accuracy & Balanced Accuracy  
        - F1-Macro  
        - Klassifikationsberichte (im Notebook)  
        - Normalisierte Confusion Matrices (als PNG gespeichert)  

        Alle Trainings- und Auswertungsschritte laufen im Notebook  
        `model-evaluation.ipynb`.  
        Die App lädt nur die fertigen Artefakte aus dem Ordner `models/` und
        `documentation/model_evaluation/`.
        """
    )

    # =========================
    # 1. Notebook-Dokumentation (kurz)
    # =========================
    st.header("1. Notebook-Workflow (kurz dokumentiert)")

    st.subheader("1.1 Imports, Daten laden & Label-Encoding")
    st.code(
        """import numpy as np
import pandas as pd
import ast
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
from gensim.models import Word2Vec

df = pd.read_csv("data/clean/data.csv")

df["tokens"] = df["tokens"].apply(ast.literal_eval)
texts = df["tokens"]
labels = df["tag"]

CM_DIR = "documentation/model_evaluation"
os.makedirs(CM_DIR, exist_ok=True)

# Label-Encoding für Genres
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)""",
        language="python",
    )

    st.subheader("1.2 Word2Vec-Embedding + Klassifikation")
    st.markdown(
        """
        - Training eines **Word2Vec**-Modells auf den Trainings-Tokens  
        - Embedding pro Dokument = Durchschnitt aller Wortvektoren  
        - Training von drei Klassifikatoren:  
            - LinearSVC  
            - Logistic Regression  
            - Random Forest  
        - Für jeden Klassifikator wird eine **normalisierte Confusion Matrix** geplottet  
          und als PNG nach `documentation/model_evaluation/` gespeichert.
        """
    )
    st.code(
        """w2v = Word2Vec(
    sentences=X_train_tokens,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10,
    seed=42
)

def embed_sentence(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

X_train_w2v = np.vstack([embed_sentence(toks, w2v) for toks in X_train_tokens])
X_test_w2v  = np.vstack([embed_sentence(toks, w2v) for toks in X_test_tokens)

# Beispiel: LinearSVC + Confusion Matrix speichern
clf_w2v_svc = LinearSVC(class_weight="balanced", max_iter=10000)
clf_w2v_svc.fit(X_train_w2v, y_train)
y_pred_w2v_svc = clf_w2v_svc.predict(X_test_w2v)

cm = confusion_matrix(y_test, y_pred_w2v_svc,
                      labels=label_encoder.transform(label_encoder.classes_))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure()
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Normalized Confusion Matrix – Word2Vec + LinearSVC")
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, "cm_w2v_LinearSVC.png"), dpi=150)
plt.show()""",
        language="python",
    )

    st.subheader("1.3 TF-IDF-Embedding + Klassifikation")
    st.markdown(
        """
        - TF-IDF auf **Character n-grams (3–5)**  
        - Gleiche drei Klassifikatoren wie bei Word2Vec  
        - Confusion Matrices ebenfalls als PNG gespeichert  
          (`cm_tfidf_*.png`).
        """
    )
    st.code(
        """X_train_texts_char = X_train_texts.apply(lambda toks: " ".join(toks))
X_test_texts_char  = X_test_texts.apply(lambda toks: " ".join(toks))

tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    max_df=0.9,
)

X_train_tfidf = tfidf.fit_transform(X_train_texts_char)
X_test_tfidf  = tfidf.transform(X_test_texts_char)

clf_tfidf_svc = LinearSVC(class_weight="balanced")
clf_tfidf_svc.fit(X_train_tfidf, y_train)
y_pred_tfidf_svc = clf_tfidf_svc.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred_tfidf_svc,
                      labels=label_encoder.transform(label_encoder.classes_))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure()
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Normalized Confusion Matrix – TF-IDF + LinearSVC")
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, "cm_tfidf_LinearSVC.png"), dpi=150)
plt.show()""",
        language="python",
    )

    st.subheader("1.4 Transformer-Embedding (SentenceTransformer MiniLM) + Klassifikation")
    st.markdown(
        """
        - Verwendung von **SentenceTransformer all-MiniLM-L6-v2** (`device="cpu"`)  
        - Embeddings werden direkt aus den vollständigen Song-Texten erzeugt  
        - Wieder drei Klassifikatoren  
        - Confusion Matrices als `cm_st_*.png` gespeichert.
        """
    )
    st.code(
        """model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

X_train_sent = [" ".join(toks) for toks in X_train_texts]
X_test_sent  = [" ".join(toks) for toks in X_test_texts]

X_train_emb_st = model.encode(
    X_train_sent,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=False,
    convert_to_tensor=True,
)

X_test_emb_st = model.encode(
    X_test_sent,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=False,
    convert_to_tensor=True,
)

X_train_emb_st = X_train_emb_st.tolist()
X_test_emb_st  = X_test_emb_st.tolist()

clf_st_svc = LinearSVC(class_weight="balanced", max_iter=10000)
clf_st_svc.fit(X_train_emb_st, y_train)
y_pred_st_svc = clf_st_svc.predict(X_test_emb_st)

cm = confusion_matrix(y_test, y_pred_st_svc,
                      labels=np.arange(len(label_encoder.classes_)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure()
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title("SentenceTransformer (MiniLM) + LinearSVC")
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, "cm_st_LinearSVC.png"), dpi=150)
plt.show()""",
        language="python",
    )

    st.subheader("1.5 Speichern des finalen Modells & der Evaluationsergebnisse")
    st.code(
        """# bestes Modell (MiniLM + LinearSVC) speichern
os.makedirs("models", exist_ok=True)
joblib.dump(clf_st_svc, "models/clf_st_svc.joblib")
joblib.dump(label_encoder, "models/label_encoder.joblib")

# eval_results.json + confusion_matrix_best.npy
# (Metriken und Confusion Matrix des besten Modells)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

eval_file = MODELS_DIR / "eval_results.json"
eval_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

cm_path = MODELS_DIR / "confusion_matrix_best.npy"
np.save(cm_path, cm_best)""",
        language="python",
    )

    st.info(
        "Oben ist der Notebook-Workflow für die Modelle grob dokumentiert. "
        "Im nächsten Abschnitt lädt die App nur die gespeicherten Ergebnisse "
        "(`models/eval_results.json`, `documentation/model_evaluation/cm_*.png`, …) "
        "und visualisiert sie."
    )

    # =================================================== #
    # 2. Notebook-Resultate in der App
    # =================================================== #
    st.header("2. Notebook-Resultate – Modellvergleich & Confusion Matrices")

    MODELS_DIR = "models"
    CM_DIR = "documentation/model_evaluation"

    def load_eval_results():
        path = os.path.join(MODELS_DIR, "eval_results.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    eval_results = load_eval_results()
    if eval_results is None:
        st.info(
            "Keine `eval_results.json` gefunden.\n\n"
            "Bitte führe zuerst `model-evaluation.ipynb` aus und speichere die "
            "Ergebnisse als `models/eval_results.json`."
        )
    else:
        # bestes Modell-Label aus JSON
        best_model_name = eval_results.get("best_model")

        # JSON → DataFrame (alle Modelle außer 'best_model')
        rows = []
        for model_name, vals in eval_results.items():
            if model_name == "best_model":
                continue
            row = {
                "model": model_name,
                "embedding": vals.get("embedding", ""),
                "classifier": vals.get("classifier", ""),
                "accuracy": vals.get("accuracy", None),
                "balanced_accuracy": vals.get("balanced_accuracy", None),
                "f1_macro": vals.get("f1_macro", None),
            }
            rows.append(row)

        if not rows:
            st.warning("Keine Modell-Metriken in `eval_results.json` gefunden.")
        else:
            df_eval = pd.DataFrame(rows)
            df_eval = df_eval.sort_values(by="f1_macro", ascending=False)

            st.subheader("2.1 Übersicht über alle Modelle")

            def highlight_best(row):
                if best_model_name is not None and row["model"] == best_model_name:
                    return ["background-color: #C7F6FF; color:black; font-weight: bold"] * len(row)
                return [""] * len(row)

            styled = (
                df_eval.style
                .format(
                    {
                        "accuracy": "{:.3f}",
                        "balanced_accuracy": "{:.3f}",
                        "f1_macro": "{:.3f}",
                    },
                    na_rep="-",
                )
                .apply(highlight_best, axis=1)
            )
            st.dataframe(styled, use_container_width=True)

            st.markdown("---")
            st.subheader("2.2 F1-Macro nach Modell")
            st.bar_chart(df_eval.set_index("model")["f1_macro"])

            # --- Detailansicht pro Modell ---
            st.markdown("---")
            st.subheader("2.3 Details zu einem Modell (inkl. Confusion Matrix)")

            model_names = df_eval["model"].tolist()
            default_index = 0
            if best_model_name in model_names:
                default_index = model_names.index(best_model_name)

            model_choice = st.selectbox(
                "Modell auswählen:",
                model_names,
                index=default_index,
            )

            row = df_eval[df_eval["model"] == model_choice].iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Accuracy",
                f"{row['accuracy']:.3f}" if pd.notna(row["accuracy"]) else "-",
            )
            c2.metric(
                "Balanced Accuracy",
                f"{row['balanced_accuracy']:.3f}"
                if pd.notna(row["balanced_accuracy"])
                else "-",
            )
            c3.metric(
                "F1-Macro",
                f"{row['f1_macro']:.3f}" if pd.notna(row["f1_macro"]) else "-",
            )

            st.write(f"**Embedding:** {row['embedding']}")
            st.write(f"**Classifier:** {row['classifier']}")

            # Confusion-Matrix-Bild für das ausgewählte Modell
            st.markdown("---")
            st.subheader("Confusion Matrix – ausgewähltes Modell")

            cm_img_path = os.path.join(CM_DIR, f"cm_{row['model']}.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_column_width=200)
            else:
                st.info(
                    f"Keine Confusion-Matrix-Grafik für `{row['model']}` gefunden.\n\n"
                    f"Erwarte Datei: `{cm_img_path}`."
                )

            # --- Bestes Modell extra hervorheben ---
            if best_model_name is not None:
                st.markdown("---")
                st.subheader("2.4 🎯 Bestes Modell (laut F1-Macro)")

                best_row_df = df_eval[df_eval["model"] == best_model_name]
                if not best_row_df.empty:
                    br = best_row_df.iloc[0]
                    st.write(
                        f"- **Name:** `{best_model_name}`  \n"
                        f"- **Embedding:** {br['embedding']}  \n"
                        f"- **Classifier:** {br['classifier']}  \n"
                        f"- **Accuracy:** {br['accuracy']:.3f}  \n"
                        f"- **F1-macro:** {br['f1_macro']:.3f}"
                        + (
                            f"  \n- **Balanced Accuracy:** {br['balanced_accuracy']:.3f}"
                            if pd.notna(br["balanced_accuracy"])
                            else ""
                        )
                    )

                    # Confusion-Matrix-Bild des besten Modells
                    best_cm_img = os.path.join(CM_DIR, f"cm_{best_model_name}.png")
                    st.subheader("Confusion Matrix – bestes Modell")
                    if os.path.exists(best_cm_img):
                        st.image(best_cm_img, use_column_width=200)
                    else:
                        st.info(
                            f"Keine Confusion-Matrix-Grafik für das beste Modell "
                            f"`{best_model_name}` gefunden.\n\n"
                            f"Erwarte Datei: `{best_cm_img}`."
                        )

    st.info(
        "Alle Metriken und Confusion Matrices stammen aus dem Notebook "
        "`model-evaluation.ipynb`. In der App findet **kein erneutes Training** statt – "
        "es werden ausschließlich die gespeicherten Artefakte geladen und visualisiert."
    )


# -------------------------------------------------------
# Seite 7 – Text Classification
# -------------------------------------------------------
elif page == "7️⃣ Text Classification":
    st.title("7️⃣ Text Classification – Genre Vorhersage")

    st.markdown("""
    **Text Classification: Genius Song Lyrics (1%)**  
    **Dataset:** 34'049 Songs · 26'408 Artists · 6 Genres  
    Genres: *Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous*  

    **Purpose:**  
    Verwendung des im Notebook `model-evaluation.ipynb` gewählten **besten Modells**,
    um neue Songtexte automatisch einem Genre zuzuordnen.  
    Dieses Kapitel dient als **Prototyp** für eine interaktive Text-Classification-Demo:
    - Klassifikation **einzelner Lyrics**
    - Klassifikation **mehrerer Lyrics (Batch)**
    """)

    st.markdown("""
    **Ausgewähltes Modell:**  
    > SentenceTransformer (**all-MiniLM-L6-v2**) + **LinearSVC**  
    """)

    # -----------------------------
    # 1. Imports and Setup – Doku
    # -----------------------------
    st.header("1. Imports and Setup (Dokumentation Notebook)")

    st.subheader("1.1 Import Libraries")
    st.code(
        """import joblib
import numpy as np
from sentence_transformers import SentenceTransformer""",
        language="python",
    )

    st.subheader("1.2 Load Trained Model and Label Encoder")
    st.markdown("""
    Laden des im Kapitel *Model Evaluation* gewählten **finalen Klassifikationsmodells**
    (SentenceTransformer + LinearSVC) sowie des `LabelEncoder` für die Genres.
    """)
    st.code(
        """#%% md
## 1.2 Load Trained Model and Label Encoder
#%%
# Load classifier and label encoder
clf_st_svc = joblib.load("models/clf_st_svc.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# Load SentenceTransformer model
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

print("Model and label encoder loaded.")
print("Genres:", list(label_encoder.classes_))""",
        language="python",
    )

    # -----------------------------
    # 2. Classification – Doku
    # -----------------------------
    st.header("2. Classification (Notebook-Prototyp)")

    st.subheader("2.1 Classification of one Lyric")
    st.markdown("""
    Beispiel: Ein einzelner Songtext wird gereinigt, mit **MiniLM** eingebettet
    und über den **LinearSVC** klassifiziert.
    """)
    st.code(
        """#%% md
## 2.1 Classification of one Lyric
#%%
lyrics = \"\"\" 
Yeah I'm driving through the city late at night,
lights low, bass loud, trouble on my mind...
\"\"\"
#%%
lyrics_clean = lyrics.strip()
#%%
embedding_tensor = st_model.encode(
    [lyrics_clean],
    batch_size=16,
    show_progress_bar=False,
    convert_to_numpy=False,
    convert_to_tensor=True,
)

# convert to python list
embedding = embedding_tensor.tolist()
#%%
pred_idx = clf_st_svc.predict(embedding)[0]
pred_genre = label_encoder.inverse_transform([pred_idx])[0]

print("Predicted genre:", pred_genre)""",
        language="python",
    )

    st.subheader("2.2 Classification of more Lyrics")
    st.markdown("""
    Im zweiten Schritt werden mehrere kurze Beispiel-Lyrics in einem Rutsch
    klassifiziert, um das Modellverhalten zu demonstrieren.
    """)
    st.code(
        """#%% md
## 2.2 Classification of more Lyrics
#%%
texts = [
    "Yeah, I'm riding through the city with my homies late at night...",
    "Baby, I miss you every single day, I can't get you off my mind...",
    "Whiskey on the dashboard, small town lights and dusty roads...",
    "The crowd is roaring, the drums are loud, the stage is burning..."
]
#%%
emb = st_model.encode(
    [t.strip() for t in texts],
    convert_to_numpy=False,
    convert_to_tensor=True,
    show_progress_bar=False,
)
emb_list = emb.tolist()
#%%
pred_idx = clf_st_svc.predict(emb_list)
pred_genres = label_encoder.inverse_transform(pred_idx)

for t, g in zip(texts, pred_genres):
    print(t[:80] + "...")
    print("→", g)
    print("-" * 50)""",
        language="python",
    )

    st.subheader("2.3 Interpretation (Notebook)")
    st.markdown("""
    Die im Notebook gezeigten Vorhersagen wirken **intuitiv**:

    - *„City + homies + late night“* → eher **Rock**  
      (könnte auch Rap sein, aber die Stimmung ist eher „rebellisch/rockig“)
    - *„I miss you every single day“* → **Country**  
      (klassisches Heartbreak-Thema)
    - *„Whiskey + dusty roads + small town“* → eindeutig **Country**
    - *„Crowd, drums, stage is burning“* → **Pop**  
      (klare Stadion-/Performance-Energie)

    **Fazit:**  
    Das Modell weist Genres auf Basis kurzer Texte den typischen lyrischen Themen
    sehr plausibel zu. Selbst knappe Ausschnitte reichen, um stilistische Hinweise
    sinnvoll zu nutzen.
    """)

    st.markdown("---")

    # =================================================== #
    # 🔮 Interaktive Demo – basierend auf gespeichertem Modell
    # =================================================== #
    st.header("🔮 Interaktive Demo – Genre-Vorhersage für neue Lyrics")

    MODEL_PATH = "models/clf_st_svc.joblib"
    ENCODER_PATH = "models/label_encoder.joblib"

    # Cache-Funktion, damit das Modell nicht bei jedem Rerun neu geladen wird
    @st.cache_resource
    def load_text_classifier():
        from sentence_transformers import SentenceTransformer
        import joblib

        clf = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return clf, le, st_model

    # Prüfen, ob Dateien existieren
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        st.error(
            "Klassifikationsmodell oder LabelEncoder nicht gefunden.\n\n"
            f"Erwartete Dateien:\n"
            f"- `{MODEL_PATH}`\n"
            f"- `{ENCODER_PATH}`\n\n"
            "Bitte zuerst das Notebook **model-evaluation.ipynb** vollständig ausführen."
        )
    else:
        try:
            clf, label_encoder, st_model = load_text_classifier()
            st.success("SentenceTransformer + LinearSVC erfolgreich geladen.")
            st.write("**Genres:**", ", ".join(label_encoder.classes_))
        except Exception as e:
            st.error(f"Fehler beim Laden des Modells: {e}")
            st.stop()

        # -----------------------------
        # 2.1 Interaktive Single-Lyric-Klassifikation
        # -----------------------------
        st.subheader("2.1 Einzelnen Songtext klassifizieren")

        default_lyric = (
            "Yeah I'm driving through the city late at night, "
            "lights low, bass loud, trouble on my mind..."
        )

        user_lyric = st.text_area(
            "Gib hier deinen Songtext ein:",
            value=default_lyric,
            height=160,
        )

        if st.button("🎧 Genre vorhersagen"):
            text_clean = user_lyric.strip()
            if not text_clean:
                st.warning("Bitte zuerst einen Songtext eingeben.")
            else:
                with st.spinner("Embedding berechnen & Genre vorhersagen..."):
                    emb = st_model.encode(
                        [text_clean],
                        batch_size=16,
                        show_progress_bar=False,
                        convert_to_numpy=False,
                        convert_to_tensor=True,
                    ).tolist()

                    pred_idx = clf.predict(emb)[0]
                    pred_genre = label_encoder.inverse_transform([pred_idx])[0]

                st.success(f"Vorhergesagtes Genre: **{pred_genre}**")

        st.markdown("---")

        # -----------------------------
        # 2.2 Batch-Klassifikation mehrerer Lyrics
        # -----------------------------
        st.subheader("2.2 Mehrere Lyrics auf einmal klassifizieren")

        st.markdown("""
        Gib mehrere Songtexte ein, **einer pro Zeile**.  
        Kurze Fragments reichen bereits, das Modell arbeitet mit Kontextsignalen.
        """)

        batch_text = st.text_area(
            "Mehrere Lyrics (eine Zeile = ein Text):",
            value=(
                "Yeah, I'm riding through the city with my homies late at night...\n"
                "Baby, I miss you every single day, I can't get you off my mind...\n"
                "Whiskey on the dashboard, small town lights and dusty roads...\n"
                "The crowd is roaring, the drums are loud, the stage is burning..."
            ),
            height=180,
        )

        if st.button("📚 Alle Zeilen klassifizieren"):
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            if not lines:
                st.warning("Bitte mindestens eine nicht-leere Zeile eingeben.")
            else:
                with st.spinner("Embeddings berechnen & Genres vorhersagen..."):
                    emb_batch = st_model.encode(
                        lines,
                        convert_to_numpy=False,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    ).tolist()

                    pred_idx = clf.predict(emb_batch)
                    pred_genres = label_encoder.inverse_transform(pred_idx)

                df_results = pd.DataFrame(
                    {
                        "text": [t[:80] + ("..." if len(t) > 80 else "") for t in lines],
                        "genre": pred_genres,
                    }
                )
                st.dataframe(df_results, use_container_width=True)

        st.info(
            "Diese Seite nutzt **genau das im Notebook trainierte Modell** "
            "(SentenceTransformer *all-MiniLM-L6-v2* + LinearSVC). "
            "In der App wird **nichts neu trainiert**, sondern nur geladen und angewendet."
        )


# -------------------------------------------------------
# Seite 8 – Text Generation
# -------------------------------------------------------
elif page == "8️⃣ Text-Generierung":
    st.title("8️⃣ Text-Generierung – Markov Chain Lyrics")

    st.markdown("""
    **Lyrics Generation: Genius Song Lyrics (1%)**  
    Dataset: 34'049 Songs · 26'408 Artists · 6 Genres  
    Genres: Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous  

    **Purpose:**  
    Generierung neuer, stilkonsistenter Songtexte mithilfe eines einfachen
    **Markov-Chain-Modells**, das auf den bestehenden Lyrics trainiert wird.

    Das Notebook unterstützt:
    - Generierung aus dem **gesamten Korpus**
    - **Genre-spezifische** Lyrics (z. B. nur *country*, nur *rap*)
    """)

    # =========================
    # 1. Imports and Setup – DOKU
    # =========================
    st.header("1. Imports and Setup")

    st.subheader("1.1 Import Libraries and Load Data")
    st.markdown("""
    Zunächst werden `pandas` zum Laden des Datensatzes und `markovify`
    für das Markov-Chain-Modell importiert.  
    Anschließend wird der bereinigte Datensatz geladen und ein erster Blick
    auf die relevanten Spalten (`lyrics`, `tag`) geworfen.
    """)
    st.code(
        """import pandas as pd
import markovify

#%%
df = pd.read_csv("data/clean/data.csv")

print(df.shape)
df[["lyrics", "tag"]].head()""",
        language="python",
    )

    st.subheader("1.2 Data Preparation")
    st.markdown("""
    Alle vorhandenen Lyrics werden gesammelt und zu einem großen Text-Korpus
    zusammengefügt, der als Trainingsbasis für das Markov-Modell dient.
    """)
    st.code(
        """#%% md
## 1.2 Data Preparation
#%%
all_lyrics = df["lyrics"].dropna().tolist()

# Join all lyrics into one big text
corpus_text = "\\n".join(all_lyrics)""",
        language="python",
    )

    # =========================
    # 2. Markov chain model – DOKU
    # =========================
    st.header("2. Markov Chain Model")

    st.subheader("2.1 Build Model")
    st.markdown("""
    Aus dem gesamten Textkorpus wird ein **Markov-Chain-Modell** mit
    `state_size=2` gebaut.  

    Zur Generierung einzelner Zeilen wird  
    `model.make_short_sentence(max_chars=90, tries=100)` verwendet:

    - `make_short_sentence` erzeugt einen **gültigen Satz** mit maximal `max_chars` Zeichen.  
    - Besser geeignet für kurze, lyrics-ähnliche Zeilen als `make_sentence()`.  
    - `tries=100` steuert, wie viele Versuche unternommen werden, bevor aufgegeben wird.  

    So bleiben die generierten Zeilen **knapp** und erinnern an typische Songtext-Zeilen.
    """)
    st.code(
        """#%% md
# 2. Markov chain model
## 2.1 Build Model
#%%
text_model_all = markovify.Text(corpus_text, state_size=2)""",
        language="python",
    )

    st.subheader("2.2 Generate a few lines")
    st.markdown("Generiert einige Beispiel-Zeilen aus dem **gesamten Korpus**.")
    st.code(
        """#%% md
## 2.2 Generate a few lines
#%%
print("=== Generated lyrics (full corpus) ===\\n")
for _ in range(10):
    line = text_model_all.make_short_sentence(max_chars=90, tries=100)
    if line:
        print(line)""",
        language="python",
    )

    st.subheader("2.3 Genre-specific Lyrics")
    st.markdown("""
    Für eine **genre-spezifische** Generierung wird zunächst ein Subset der
    Lyrics nach Tag (`df["tag"]`) gefiltert und daraus ein neues Markov-Modell
    gebaut. Die Funktion `generate_markov_lyrics` kapselt dieses Verhalten.
    """)
    st.code(
        """#%% md
## 2.3 Generate Genre-specific Lyrics
#%%
def generate_markov_lyrics(genre=None, num_lines=10):
    \"\"\"Generate Markov-based lyrics from the full corpus or a specific genre.\"\"\"
    if genre is None:
        subset = df["lyrics"].dropna().tolist()
        label = "full corpus"
    else:
        subset = df[df["tag"] == genre]["lyrics"].dropna().tolist()
        label = f"genre: {genre}"

    corpus_text = "\\n".join(subset)
    model = markovify.Text(corpus_text, state_size=2)

    print(f"=== Generated lyrics ({label}) ===\\n")
    for _ in range(num_lines):
        line = model.make_short_sentence(max_chars=90, tries=100)
        if line:
            print(line)

# Beispiel: 10 Zeilen nur aus Country-Songs
generate_markov_lyrics(genre="country", num_lines=10)""",
        language="python",
    )

    st.subheader("2.4 Lyrics with Verse and Chorus")
    st.markdown("""
    Für komplexere Songstrukturen werden Hilfsfunktionen definiert:

    - `generate_line` – eine einzelne Zeile mit Markovify  
    - `generate_verse` – mehrere Zeilen als **Strophe**  
    - `generate_chorus` – **Refrain** mit teilweise wiederkehrenden Zeilen  
    - `generate_song` – baut einen Song aus `[Verse 1]`, `[Chorus]`, `[Verse 2]`
    """)
    st.code(
        """#%% md
## 2.4 Generate Lyrics with Verse and Chorus
#%%
def generate_line(model, max_tries=100):
    line = model.make_short_sentence(max_chars=90, tries=100)
    return line if line else ""

#%%
def generate_verse(model, num_lines=8):
    lines = []
    for _ in range(num_lines):
        line = generate_line(model)
        if line:
            lines.append(line)
    return lines

#%%
def generate_chorus(model, num_lines=4):
    lines = []
    base_line = generate_line(model)
    if not base_line:
        base_line = "La la la"

    for i in range(num_lines):
        if i % 2 == 0:
            lines.append(base_line)
        else:
            line = generate_line(model)
            lines.append(line if line else base_line)
    return lines

#%%
def generate_song(model):
    verse1 = generate_verse(model)
    chorus = generate_chorus(model)
    verse2 = generate_verse(model)

    print("[Verse 1]")
    print("\\n".join(verse1))
    print("\\n[Chorus]")
    print("\\n".join(chorus))
    print("\\n[Verse 2]")
    print("\\n".join(verse2))

#%%
genre = "country"
subset = df[df["tag"] == genre]["lyrics"].dropna().tolist()
text_model_genre = markovify.Text("\\n".join(subset), state_size=2)

generate_song(text_model_genre)""",
        language="python",
    )

    st.info(
        "Oben ist der komplette Markov-Workflow aus dem Notebook dokumentiert. "
        "Im nächsten Abschnitt gibt es eine kleine interaktive Lyrics-Generierung "
        "direkt in der App."
    )

    # =================================================== #
    # 🎤 Interaktive Markov-Lyrics in der App
    # =================================================== #
    st.header("🎤 Interaktive Lyrics-Generierung")

    # Für die App: Daten laden (falls nicht global)
    import pandas as pd
    import markovify

    @st.cache_data
    def load_lyrics_df():
        return pd.read_csv("data/clean/data.csv")

    df_markov = load_lyrics_df()

    # Auswahl: Genre oder kompletter Korpus
    all_genres = sorted(df_markov["tag"].dropna().unique().tolist())
    genre_options = ["(Full corpus)"] + all_genres

    st.subheader("1) Einstellungen")

    col1, col2 = st.columns(2)
    with col1:
        selected_genre = st.selectbox(
            "Genre für das Markov-Modell:",
            genre_options,
            index=0,
        )
    with col2:
        num_lines = st.slider("Anzahl Zeilen für einfache Generierung:", 3, 20, 8)

    st.markdown("---")
    st.subheader("2) Einfache Zeilen generieren")

    if st.button("Zeilen generieren"):
        if selected_genre == "(Full corpus)":
            subset = df_markov["lyrics"].dropna().tolist()
            label = "full corpus"
        else:
            subset = df_markov[df_markov["tag"] == selected_genre]["lyrics"].dropna().tolist()
            label = f"genre: {selected_genre}"

        if not subset:
            st.warning(f"Keine Lyrics für Auswahl **{label}** gefunden.")
        else:
            corpus_text = "\n".join(subset)
            model = markovify.Text(corpus_text, state_size=2)

            lines = []
            for _ in range(num_lines):
                line = model.make_short_sentence(max_chars=90, tries=100)
                if line:
                    lines.append(line)

            st.markdown(f"**Generated lyrics ({label})**:")
            if lines:
                st.text("\n".join(lines))
            else:
                st.warning("Es konnten keine Zeilen generiert werden.")

    st.markdown("---")
    st.subheader("3) Song mit Vers & Chorus generieren")

    verse_lines = st.slider("Zeilen pro Vers:", 4, 16, 8)
    chorus_lines = st.slider("Zeilen im Chorus:", 2, 8, 4)

    if st.button("Song generieren (Verse + Chorus)"):
        if selected_genre == "(Full corpus)":
            subset = df_markov["lyrics"].dropna().tolist()
            label = "full corpus"
        else:
            subset = df_markov[df_markov["tag"] == selected_genre]["lyrics"].dropna().tolist()
            label = f"genre: {selected_genre}"

        if not subset:
            st.warning(f"Keine Lyrics für Auswahl **{label}** gefunden.")
        else:
            corpus_text = "\n".join(subset)
            model = markovify.Text(corpus_text, state_size=2)

            def gen_line(m, max_tries=100):
                line = m.make_short_sentence(max_chars=90, tries=100)
                return line if line else ""

            def gen_verse(m, n):
                out = []
                for _ in range(n):
                    line = gen_line(m)
                    if line:
                        out.append(line)
                return out

            def gen_chorus(m, n):
                out = []
                base = gen_line(m)
                if not base:
                    base = "La la la"
                for i in range(n):
                    if i % 2 == 0:
                        out.append(base)
                    else:
                        line = gen_line(m)
                        out.append(line if line else base)
                return out

            verse1 = gen_verse(model, verse_lines)
            chorus = gen_chorus(model, chorus_lines)
            verse2 = gen_verse(model, verse_lines)

            st.markdown(f"**Generated song ({label})**")
            st.text(
                "[Verse 1]\n" +
                "\n".join(verse1) +
                "\n\n[Chorus]\n" +
                "\n".join(chorus) +
                "\n\n[Verse 2]\n" +
                "\n".join(verse2)
            )

    st.info(
        "Die Markov-Kette nutzt nur lokale Übergangswahrscheinlichkeiten, "
        "erzeugt aber trotzdem stilistische Muster, die an die Original-Lyrics erinnern. "
        "Für wirklich kohärente Songs wären neuronale Sprachmodelle nötig – "
        "hier geht es um eine leichtgewichtige, erklärbare Demo."
    )
