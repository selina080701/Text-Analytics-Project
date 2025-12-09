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

        st.subheader("👀 Vorschau")
        st.dataframe(df_real.head())

        st.subheader("🎵 Genre-Verteilung")
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

    st.subheader("1.1 Load Dataset")
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

    - Metadaten-Tags wie **`[Intro]`, `[Verse]`, `[Hook]`**
    - Zeilenumbrüche **`\\n`**
    - Mehrfache bzw. führende/abschließende Leerzeichen  

    Diese Elemente erschweren spätere NLP-Analysen und müssen daher bereinigt werden.
    """)

    preview_text = """
    0    [Intro]\\nBitch I'm clean\\nTwo sticks like Chow...
    1    My old girl left me on her old bull shit\\nSo I...
    2    [Intro: spoken]\\nAvast there matey haha\\nIf a ...
    3    Just throw a glimpse under the shell\\nGhostly ...
    4    [Verse 1]\\nI miss the taste of a sweeter life\\...
    Name: lyrics, dtype: object
    """

    st.code(preview_text, language="text")

    st.subheader("2.2 Define and Apply Cleaning Function")
    st.markdown("""
    ### 2.2 Define and Apply Cleaning Function

    Die Cleaning-Funktion bereitet den Text in drei Schritten vor:

    1. **Entfernen aller Inhalte zwischen eckigen Klammern**  
       `re.sub(r'\\[.*?\\]', '', text)`  
       Dieser Ausdruck löscht jeden Abschnitt, der zwischen `[` und `]` steht – inklusive des enthaltenen Textes. Dadurch verschwinden z. B. Annotationen, Quellen, Zeitstempel oder andere Meta-Informationen.

    2. **Ersetzen von Zeilenumbrüchen durch Leerzeichen**  
       `text.replace('\\n', ' ')`  
       Zeilenumbrüche werden in Leerzeichen umgewandelt, damit der Text eine durchgehende Linie bildet und besser weiterverarbeitet werden kann.

    3. **Reduzieren mehrfacher Leerzeichen & finales Formatieren**  
       `re.sub(r'\\s+', ' ', text).strip()`  
       Mehrere aufeinanderfolgende Leerzeichen werden zu einem einzigen zusammengefasst.  
       Gleichzeitig entfernt `.strip()` führende und nachfolgende Leerzeichen.  
       Das Ergebnis ist ein sauberer, kompakter Text ohne unnötige Abstände.

    **Endresultat:** Der Text ist frei von Klammerinhalten, hat keine Zeilenumbrüche mehr und enthält nur noch einheitliche Leerzeichen – optimal zur weiteren Analyse oder NLP-Verarbeitung.
    """)
    st.code(
        """
def clean_lyrics(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        """
df = df.drop(columns=['lyrics'])
df = df.rename(columns={'lyrics_clean': 'lyrics'})""",
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
        """
df = df.drop(columns=["lyrics"])

df = df.rename(columns={"lyrics_clean": "lyrics"})""",
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
        display_cols = [
            c for c in [
                "title",
                "tag",
                "artist",
                "year",
                "views",
                "features",
                "id",
                "language_cld3",
                "language_ft",
                "language",
                "lyrics"
            ]
            if c in df_clean.columns
        ]
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

    st.markdown("""
    Es wird das im **Kapitel 2** bereinigte Datenset geladen, das bereits  
    von Metadaten-Tags und Zeilenumbrüchen befreite Lyrics enthält.
    """)
    st.code(
        """df = pd.read_csv('data/clean/lyrics_subset_1pct_clean.csv')

df = df[df["language_cld3"] == "en"]""",
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
        """
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
        """STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","than","that","those","these","this",
    "to","of","in","on","for","with","as","at","by","from","into","over","under","up","down",
    "is","am","are","was","were","be","been","being","do","does","did","doing","have","has","had",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their",
    "not","no","yes","yeah","y'all","yall","im","i'm","i’d","i'd","i’ll","i'll","youre","you're","dont","don't",
    "cant","can't","ill","i’ll","id","i'd","ive","i’ve","ya","oh","ooh","la","na","nah"
}

def filtered_tokens(text):
    tokens = preprocess_text(text)
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 1]

filtered_tokens("This is a test!")""",
        language="python"
    )

    st.subheader("2.3 Visualisierung der häufigsten Wörter")
    st.markdown("""
    Zur Veranschaulichung werden die häufigsten Wörter **vor** und **nach**
    dem Entfernen der Stopwörter gegenübergestellt.  

    Die Plots zeigen deutlich, dass das Entfernen von Stopwörtern die Verteilung stark verändert:
    Das häufigste Wort **nach** dem Filtern taucht in den ursprünglichen Top-15 gar nicht mehr auf.
    """)

    FIG_DIR = "documentation/tokenization"
    img = os.path.join(FIG_DIR, "top_15_words.png")

    if os.path.exists(img):
        st.image(img, use_container_width=200)
    else:
        st.warning("⚠️ Das Bild konnte nicht gefunden werden.")

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

    st.subheader("1.1 Load Dataset")
    st.markdown("""
        Laden des final bereinigten Datensatzes (`data/clean/data.csv`) und
        Rückkonvertierung der Spalten `words` und `tokens` von String-Repräsentationen
        zu echten Python-Listen (mittels `ast.literal_eval`).
        """)
    st.code(
        """
df = pd.read_csv('data/clean/data.csv')

for col in ["words", "tokens"]:
    if isinstance(df[col].iloc[0], str):
        df[col] = df[col].apply(ast.literal_eval)
""",
        language="python",
    )

    st.subheader("1.2 Descriptive Statistics")
    st.markdown("""
        Zuerst wird die **Genre-Verteilung** analysiert und als Balkendiagramm geplottet.
        Anschließend werden **Text- und Token-Statistiken** berechnet
        (total, min, avg, max) und jeweils als kleine Übersichtsgrafik gespeichert.
        """)
    st.code(
        """
print("\\nGENRE DISTRIBUTION")
print("=" * 60)
category_counts = df['tag'].value_counts().sort_values(ascending=False)

for tag,count in category_counts.items():
    pct = (count / len(df)) * 100
    print(f"{tag}: {count:,} songs ({pct:.2f}%)")
""",
        language="python",
    )


    # -----------------------------
    # 2. Word-Level Analysis
    # -----------------------------
    st.header("2. Word-Level Analysis")

    st.subheader("2.1 Vocabulary Statistics")
    st.markdown("""
        Bestimmung der Vokabulargrösse, Gesamtzahl der Worttokens und Type–Token Ratio (TTR).
        Nun werfen wir einen genaueren Blick auf die Texte und Wörter, indem wir das Vokabular analysieren, 
        Zipfs Gesetz untersuchen, seltene Wörter (Hapaxlegomena) identifizieren und verschiedene Kategoriestatistiken untersuchen.
        """)
    st.code(
        """
all_tokens = [token for tokens in df["words"] for token in tokens]


word_counts = Counter(all_tokens)
vocab_size = len(word_counts)
type_token_ratio = vocab_size / len(all_tokens)

""",
        language="python",
    )
    preview_text = """
VOCABULARY STATISTICS
============================================================
Total word tokens:          10,596,323
Unique words (vocabulary):  127,659
Type-token ratio:           0.0120
    """
    st.code(preview_text, language="text")
    st.markdown("""
    Im Durchschnitt kommt jedes Wort etwa 100 Mal im Datensatz vor, was auf einen hohen Wiederholungsgrad hindeutet.
    Das type-token ratio (TTR) von 0,012 ist relativ niedrig, was zu erwarten war, 
    da der Korpus aus Songtexten besteht – einem Genre, das sich durch wiederkehrende Wörter, 
    Refrains und eine im Vergleich zu anderen Textarten begrenzte lexikalische Vielfalt auszeichnet.""")

    st.subheader("2.2 Zipf-Analyse")

    st.markdown("""
    Die Zipf-Law beschreibt eine fundamentale Eigenschaft natürlicher Sprache:
    Die Häufigkeit eines Wortes ist **umgekehrt proportional zu seinem Rang**
    in der sortierten Wortfrequenzliste.
    """)

    st.markdown("**Mathematische Form:**")
    st.latex(r"f(r) = \frac{C}{r^{\alpha}}")

    st.markdown(r"""
    **Bedeutung der Parameter:**

    - \( f(r) \) = Häufigkeit des Wortes mit Rang \( r \)
    - \( \alpha \) = Exponent bzw. Steigung (typischer Idealwert für natürliche Sprache ≈ 1.0)
    - \( C \) = Normierungskonstante

    Wenn \( \alpha = 1.0 \), dann gilt:

    - das Wort auf Rang 2 tritt **halb so häufig** auf wie das Wort auf Rang 1
    - Rang 3 tritt **ein Drittel so häufig** auf
    - Rang 4 **ein Viertel so häufig**, usw.

    Diese Potenzgesetz-Struktur ist erstaunlich stabil und findet sich in
    unterschiedlichsten Texten, Genres, Korpora und Sprachen wieder.
    """)

    st.code(
        """all_word_freq = Counter(words).most_common(100)
ranks = list(range(1, len(all_word_freq) + 1))
frequencies = [freq for word, freq in all_word_freq]


log_ranks_100 = np.log(ranks).reshape(-1, 1)
log_freq_100 = np.log(frequencies)

model = LinearRegression()
model.fit(log_ranks_100, log_freq_100)

r_squared = model.score(log_ranks_100, log_freq_100)
slope = model.coef_[0]
intercept = model.intercept_
coefficient_C = np.exp(intercept)""",
        language="python",
    )


    st.subheader("2.3 Hapax Legomena (Rare Words)")
    st.markdown("""
        Analyse der seltensten Wörter (Hapax Legomena, Count=1) und aller Wörter mit ≤5 Vorkommen.  
        Zusätzlich wird die Verteilung „Wie viele Wörter kommen X-mal vor?“ als Balkendiagramm gespeichert
        und Kennzahlen in `rare_words_stats.json` geschrieben.
        """)
    st.code(
        """
word_counts = Counter(words)
hapax = [word for word, count in word_counts.items() if count == 1]
hapax_pct = (len(hapax) / vocab_size) * 100

rare_2 = [word for word, count in word_counts.items() if count == 2]
rare_3_5 = [word for word, count in word_counts.items() if 3 <= count <= 5]
rare_le_5 = len(hapax) + len(rare_2) + len(rare_3_5)
""",
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
        """categories = df['tag'].unique()

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
""",
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
        """def ngrams(tokens, n):
    if n <= 0:
        return []
    iters = tee(tokens, n)
    for i, it in enumerate(iters):
        for _ in range(i):
            next(it, None)
    return zip(*iters)
    
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
)""",
        language="python",
    )

    st.subheader("3.2 N-Grams per Artist/Genre")
    st.markdown("""
        Für jede Gruppe (Artist / Genre) wird das jeweils häufigste N-Gramm (Uni/ Bi/ Trigram)
        bestimmt und die Top-20 bzw. Top-Listen visualisiert.  
        Die Resultate werden als `top20_ngrams_per_artist.png` und `top_ngrams_per_genre.png` gespeichert.
        """)
    st.code(
        """
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
""",
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
        preview_text = """
GENRE DISTRIBUTION
============================================================
pop: 14,100 songs (41.41%)
rap: 9,723 songs (28.56%)
rock: 6,375 songs (18.72%)
rb: 1,531 songs (4.50%)
misc: 1,450 songs (4.26%)
country: 870 songs (2.56%)
"""
        st.code(preview_text, language="text")

        if os.path.exists(img):
            st.image(img, use_container_width=200)

    # --------------------------------------------------------------------------
    # TEXT STATISTICS
    with tab_text:
        st.subheader("TEXT STATISTICS")
        img = os.path.join(FIG_DIR, "text_statistics.png")
        preview_text = """
TEXT STATISTICS
============================================================
Total lyrics (songs):     34,049
Total words:              10,596,323
Average words/lyric:      311.21
Shortest lyric:           8 words
Longest lyric:            17434 words
        """
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, use_container_width=200)

    # --------------------------------------------------------------------------
    # TOKEN STATISTICS
    with tab_token:
        st.subheader("TOKEN STATISTICS")
        img = os.path.join(FIG_DIR, "token_statistics.png")
        preview_text = """
TOKEN STATISTICS
============================================================
Total lyrics (songs):     34,049
Total tokens:             5,999,753
Unique tokens:            127,555
Average tokens/lyric:     176.21
Shortest lyric:           4 tokens
Longest lyric:            9578 tokens
        """
        st.markdown("""
        Die unteren Diagramme zeigen die Häufigkeit der 15 häufigsten Wörter vor und nach dem Entfernen von Stoppwörtern. 
        Wir können deutlich sehen, dass das Entfernen von Stoppwörtern einen signifikanten Unterschied macht: 
        Das häufigste Wort nach dem Filtern erscheint vor dem Entfernen der Stoppwörter nicht einmal unter den 15 häufigsten Wörtern.
        """)
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, use_container_width=200)

    # --------------------------------------------------------------------------
    # TOP 15 WORDS BEFORE/AFTER STOPWORDS

    with tab_top15:
        st.subheader("TOP 15 WORDS – BEFORE & AFTER STOPWORDS")
        img = os.path.join(FIG_DIR, "top15_words_before_after_stopwords.png")
        st.markdown("""
        Die Plots zeigen die 15 häufigsten Wörter vor und nach Stopwort-Entfernung.
        Das häufigste Wort nach dem Filtern erscheint nicht mehr in den ursprünglichen Top-15.
                  """)
        if os.path.exists(img):
            st.image(img, use_container_width=200)

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

        st.markdown("""Das angepasste Zipf-Law-Modell weist eine Steigung von −0,83 mit einem 
        Wert von 0,98 auf, was auf eine hervorragende Anpassung an die erwartete Verteilung hinweist.
        Obwohl die Steigung etwas flacher ist als die ideale −1,0, 
        deutet diese geringe Abweichung (0,17) darauf hin, dass die Häufigkeits-Rang-Beziehung in den Liedtexten dem Zipf-law sehr nahe kommt – 
        häufige Wörter werden viel häufiger verwendet als seltene, wie es typischerweise in Liedtexten zu beobachten ist.
            """)
        img = os.path.join(FIG_DIR, "zipf_loglog_and_top30.png")
        preview_text = """
ZIPF'S LAW ANALYSIS
============================================================
Fitted equation: f(r) = 788840.86 / r^0.828

Model parameters:
Slope (α):         -0.8284
R^2 (fit quality): 0.9845
Ideal Zipf slope:  -1.0000
Deviation:         0.1716
        """
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, caption="Zipf Log-Log + Top-30 Comparison", use_container_width=200)

    # --------------------------------------------------------------------------
    # RARE WORDS ANALYSIS
    with tab_rare:
        st.subheader("RARE WORDS ANALYSIS")

        stats = os.path.join(FIG_DIR, "rare_words_stats.json")
        if os.path.exists(stats):
            with open(stats, "r") as f:
                r = json.load(f)

        img = os.path.join(FIG_DIR, "rare_words_distribution.png")
        st.markdown("""
        Ein großer Teil des Vokabulars in den Liedtexten ist selten: 48,6 % sind Hapaxlegomena (kommen nur einmal vor) 
        und 74,3 % aller Wörter kommen fünfmal oder weniger vor. 
        Einige wenige Wörter werden häufig wiederholt, während die meisten Wörter einzigartig oder sehr selten sind.
        """)

        st.markdown("""
        Die Verteilung zeigt, dass viele Wörter nur einmal oder sehr selten in Songtexten vorkommen. 
        Das ist etwas überraschend, da man eher das Gegenteil erwarten würde: dass die meisten Wörter sehr häufig vorkommen und nur wenige Wörter selten. 
        Bei genauerer Betrachtung der Hapaxlegomena zeigt sich jedoch, dass es sich dabei oft um Wörter wie „shitforeal”, „denimits”, „matey”, „yohoho”, 'yohohoyohoho', ‚hahaher‘, ‚swabs‘, ‚bosun‘, ‚yed‘, ‚affydavy‘] 
        – also keine bedeutungsvollen Wörter im üblichen Sinne, sondern eher Zeichenfolgen oder erfundene Begriffe, die die Laute des Sängers nachahmen.
        """)
        st.code(
            f"""RARE WORDS ANALYSIS
        ============================================================
        Hapax legomena:         {r['hapax_count']:,}  ({r['hapax_pct']:.1f}% vocab)
        Rare words ≤5 times:    {r['rare_le_5']:,}  ({r['rare_le_5_pct']:.1f}% vocab)
        Examples:
          {r['example_hapax']}""",
            language="text",
        )
        if os.path.exists(img):
            st.image(img, caption="Word Frequency Distribution", use_container_width=200)

    # --------------------------------------------------------------------------
    # CATEGORY STATISTICS
    with tab_category:
        st.subheader("CATEGORY STATISTICS")
        img = os.path.join(FIG_DIR, "category_statistics.png")
        st.markdown("""
        Rap scheint über einen großen Wortschatz zu verfügen, was mit dem zuvor beobachteten Vorkommen seltener Wörter übereinstimmt. 
        Ausserdem kommen Zahlen in Rap-Songs häufiger vor als in anderen Genres. 
        Im Gegensatz dazu weisen Country-Songs tendenziell einen sehr kleinen Wortschatz auf. 
        In den meisten Genres ist die durchschnittliche Anzahl der Wörter pro Song ziemlich ähnlich, 
        obwohl Rap-Songs etwas länger sind und Songs, die als „Verschiedenes” (misc) klassifiziert sind, 
        deutlich länger sind – allerdings lässt sich diese Kategorie nicht ohne Weiteres als spezifisches Genre interpretieren.
        """)
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, use_container_width=200)

    # --------------------------------------------------------------------------
    # TOP 15 N-GRAMS
    with tab_ngrams:
        st.subheader("TOP 15 N-GRAMS")
        img = os.path.join(FIG_DIR, "top15_ngrams.png")
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, use_container_width=200)

    # --------------------------------------------------------------------------
    # TOP N-GRAMS PER ARTIST
    with tab_artist:
        st.subheader("TOP N-GRAMS PER ARTIST")
        img = os.path.join(FIG_DIR, "top20_ngrams_per_artist.png")
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, use_container_width=200)

    # --------------------------------------------------------------------------
    # TOP N-GRAMS PER GENRE
    with tab_genres:
        st.subheader("TOP N-GRAMS PER GENRE")
        img = os.path.join(FIG_DIR, "top_ngrams_per_genre.png")
        st.code(preview_text, language="text")
        if os.path.exists(img):
            st.image(img, use_container_width=200)



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
    st.header("1. Train Word2Vec Model")

    st.markdown("""
    **Ziel:** Lernen von Wortvektoren aus den Lyrics-Token mit `gensim.Word2Vec`.

    Wichtige Parameter:
    - `vector_size=50` → 50-dimensionale Embeddings (kompakt, schnell)
    - `window=5` → Kontextfenster von 5 Wörtern links/rechts
    - `min_count=2` → Wörter mit weniger als 2 Vorkommen werden ignoriert
    - `epochs=100` → 100 Trainingsdurchläufe für stabilere Vektoren
    """)
    st.code(
        """l

**Ziel:** Lernen von Wortvektoren aus den Lyrics-Token.
**Bibliothek:** `gensim.models.Word2Vec`




model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=2,
    workers=4,
    epochs=100
)
""",
        language="python",
    )
    preview_text = """
| Parameter       | Bedeutung |
|----------------|-----------|
| `sentences`    | Liste von Wortlisten (Token pro Song) |
| `vector_size`  | Dimension der Vektoren |
| `window`       | Kontextfenstergröße |
| `min_count`    | Minimalhäufigkeit für Wörter |
| `workers`      | Anzahl Threads |
| `epochs`       | Trainingsdurchläufe |
        """
    st.code(preview_text, language="text")
    st.markdown("""
        Die unteren Diagramme zeigen die Häufigkeit der 15 häufigsten Wörter vor und nach dem Entfernen von Stoppwörtern. 
        Wir können deutlich sehen, dass das Entfernen von Stoppwörtern einen signifikanten Unterschied macht: 
        Das häufigste Wort nach dem Filtern erscheint vor dem Entfernen der Stoppwörter nicht einmal unter den 15 häufigsten Wörtern.
        """)


    st.markdown("""
    **Ergebnis:**  
    Ein trainiertes Word2Vec-Modell, das jedes Wort als Punkt in einem **50-dimensionalen Raum**
    repräsentiert – Wörter mit ähnlichem Kontext liegen nah beieinander.
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
        """
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
        '''
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

df_songs = df.dropna(subset=["tokens", GENRE_COL]).copy()

df_songs["embedding"] = df_songs["tokens"].apply(
    lambda toks: get_song_vector(toks, model)
)

X = np.vstack(df_songs["embedding"].values)      
y = df_songs[GENRE_COL].astype(str).values        

print("Song embeddings shape:", X.shape)
print("Number of songs:", len(y))
print("Example genres:", y[:10])
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
        """
os.makedirs("data/features", exist_ok=True)

np.save("data/features/song_embeddings.npy", X)
np.save("data/features/song_labels.npy", y)

print("Saved song embeddings and labels to 'data/features/'")

meta_cols = [GENRE_COL]
for col in ["title", "artist", "id", "song_id"]:
    if col in df_songs.columns:
        meta_cols.append(col)

df_songs[meta_cols].to_csv("data/features/song_metadata.csv", index=False)
print("Saved song metadata to 'data/features/song_metadata.csv'")

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

    st.subheader("1.1 Label-Encoding")
    st.code(
        """
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
""",
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
""",
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
""",
        language="python",
    )

    st.subheader("1.5 Speichern des finalen Modells & der Evaluationsergebnisse")
    st.code(
        """
os.makedirs("models", exist_ok=True)
joblib.dump(clf_st_svc, "models/clf_st_svc.joblib")
joblib.dump(label_encoder, "models/label_encoder.joblib")

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

                # --------------------------------------------------
                # Hilfsfunktion: deutsche Modellbeschreibung
                # --------------------------------------------------
                def get_model_description_de(model_name: str) -> str:
                    name = model_name.lower()

                    if "w2v" in name:
                        return """
    ### 📌 Word2Vec – Zusammenfassung der Klassifikatoren

    **LinearSVC**
    - Accuracy: ~0.577  
    - Balanced Accuracy: ~0.508  
    LinearSVC liefert die stabilste Gesamtleistung. Dominante Genres (rap, pop) werden zuverlässig erkannt, und die faire Klassenverteilung ist am besten. Minderheitsgenres bleiben weiterhin schwierig.

    **Logistische Regression**
    - Accuracy: ~0.463  
    - Balanced Accuracy: ~0.551  
    Höchste Balanced Accuracy – sehr faire und ausgewogene Klassifikation über alle Genres hinweg. Allerdings sinkt die Gesamtgenauigkeit, da große Klassen schwieriger zu unterscheiden sind.

    **Random Forest**
    - Accuracy: ~0.648  
    - Balanced Accuracy: ~0.405  
    Höchste Accuracy, aber deutlichste Verzerrung zugunsten der Mehrheitsklassen (pop, rap). Sehr schwach für Minderheitsgenres.

    **Fazit (Word2Vec)**
    - Beste Gesamtperformance: **LinearSVC**  
    - Beste Fairness: **Logistische Regression**  
    - Höchste Accuracy, aber am wenigsten fair: **Random Forest**
    """

                    if "tfidf" in name:
                        return """
    ### 📌 TF-IDF – Zusammenfassung der Klassifikatoren

    **LinearSVC**
    - Accuracy: ~0.593  
    - Balanced Accuracy: ~0.458  
    Sehr gute Gesamtperformance mit stabiler Accuracy. Minderheitsgenres bleiben jedoch schwierig.

    **Logistische Regression**
    - Accuracy: ~0.551  
    - Balanced Accuracy: ~0.535  
    Beste Fairness über alle Genres – ausgewogene Klassifikation, besserer Recall für kleinere Klassen wie *misc* und *rb*.

    **Random Forest**
    - Accuracy: ~0.581  
    - Balanced Accuracy: ~0.405  
    Akzeptable Accuracy, aber deutliche Probleme bei Minderheitsgenres (insbesondere *country* und *rb*).

    **Fazit (TF-IDF)**
    - Beste Gesamtperformance: **LinearSVC**  
    - Beste Fairness: **Logistische Regression**  
    - Höchste Accuracy, aber am wenigsten fair: **Random Forest**
    """

                    if "st" in name or "transformer" in name or "minilm" in name:
                        return """
    ### 📌 Transformer (SentenceTransformer MiniLM) – Zusammenfassung

    **LinearSVC**
    - Accuracy: ~0.572  
    - Balanced Accuracy: ~0.515  
    Beste Gesamtbalance zwischen Genauigkeit und Fairness. Dominante Genres (rap, pop) werden zuverlässig erkannt, Minderheitsgenres profitieren von den reicheren Transformer-Embeddings.

    **Logistische Regression**
    - Accuracy: ~0.475  
    - Balanced Accuracy: ~0.543  
    Höchste Balanced Accuracy – sehr faire Verteilung über alle Klassen. Die Gesamtgenauigkeit ist etwas niedriger, vor allem wegen der schwierigen *pop*-Klasse.

    **Random Forest**
    - Accuracy: ~0.624  
    - Balanced Accuracy: ~0.343  
    Sehr hohe Accuracy, aber extrem schlechte Fairness gegenüber Minderheitsgenres (country, rb). Starker Bias zugunsten der Mehrheitsklassen.

    **Fazit (Transformer)**
    - Beste Gesamtperformance: **LinearSVC**  
    - Beste Fairness: **Logistische Regression**  
    - Höchste Accuracy, aber schlechteste Fairness: **Random Forest**
    """

                    return ""

                # --------------------------------------------------
                # 2.1 Übersicht über alle Modelle
                # --------------------------------------------------
                st.subheader("2.1 Übersicht über alle Modelle")

                # (hier könntest du z.B. df_eval anzeigen, wenn gewünscht)
                # st.dataframe(df_eval)

                st.markdown("---")
                st.subheader("2.2 F1-Macro nach Modell")
                st.bar_chart(df_eval.set_index("model")["f1_macro"])

                # --------------------------------------------------
                # 2.3 Details zu den Modellen (Tabs)
                # --------------------------------------------------
                st.markdown("---")
                st.subheader("2.3 Details zu den Modellen (inkl. Confusion Matrices)")


                # ---------- TAB-NAME FORMATIERUNG ----------
                def format_model_label(model_name: str) -> str:
                    name = model_name.lower()

                    # Embedding bestimmen
                    if "w2v" in name:
                        emb = "Word2Vec"
                    elif "tfidf" in name:
                        emb = "TF-IDF"
                    elif "st" in name or "transformer" in name or "minilm" in name:
                        emb = "Transformer (MiniLM)"
                    else:
                        emb = "?"

                    # Klassifikator bestimmen
                    if "svc" in name:
                        clf = "LinearSVC"
                    elif "logreg" in name or "logistic" in name:
                        clf = "Logistic Regression"
                    elif "rf" in name or "forest" in name:
                        clf = "Random Forest"
                    else:
                        clf = "?"

                    return f"{emb} – {clf}"


                # ---------- FIXE TAB-SORTIERUNG ----------
                sort_order = [
                    ("Word2Vec", "LinearSVC"),
                    ("Word2Vec", "Logistic Regression"),
                    ("Word2Vec", "Random Forest"),
                    ("TF-IDF", "LinearSVC"),
                    ("TF-IDF", "Logistic Regression"),
                    ("TF-IDF", "Random Forest"),
                    ("Transformer (MiniLM)", "LinearSVC"),
                    ("Transformer (MiniLM)", "Logistic Regression"),
                    ("Transformer (MiniLM)", "Random Forest"),
                ]

                # Modellnamen + Anzeigenamen
                models_with_labels = []
                for model_key in df_eval["model"]:
                    label = format_model_label(model_key)
                    models_with_labels.append((model_key, label))

                # Nach definierter Reihenfolge sortieren
                sorted_models = []
                for emb, clf in sort_order:
                    target = f"{emb} – {clf}"
                    for m_key, label in models_with_labels:
                        if label == target:
                            sorted_models.append((m_key, label))

                # Falls Modelle nicht gefunden → ignorieren
                if not sorted_models:
                    st.warning("Keine Modelle gefunden, um Tabs zu erzeugen.")
                else:
                    # ---------- TABS ERZEUGEN ----------
                    tabs = st.tabs([label for (_, label) in sorted_models])


                    # ---------- MODELL-BESCHREIBUNGEN (DEUTSCH) ----------
                    def get_model_description_de(model_name: str) -> str:
                        name = model_name.lower()

                        if "w2v" in name:
                            return """
                ### 📌 Word2Vec – Zusammenfassung der Klassifikatoren

                **LinearSVC**
                - Accuracy: ~0.577  
                - Balanced Accuracy: ~0.508  
                LinearSVC liefert die stabilste Gesamtleistung. Dominante Genres (rap, pop) werden zuverlässig erkannt, und die faire Klassenverteilung ist am besten.

                **Logistische Regression**
                - Accuracy: ~0.463  
                - Balanced Accuracy: ~0.551  
                Beste Fairness und höchste Balanced Accuracy über alle Genres hinweg.

                **Random Forest**
                - Accuracy: ~0.648  
                - Balanced Accuracy: ~0.405  
                Sehr hohe Accuracy, aber starker Bias zugunsten der Mehrheitsklassen.

                **Fazit**
                - Beste Gesamtperformance: **LinearSVC**  
                - Beste Fairness: **Logistische Regression**  
                - Höchste Accuracy, aber verzerrt: **Random Forest**
                """

                        if "tfidf" in name:
                            return """
                ### 📌 TF-IDF – Zusammenfassung der Klassifikatoren

                **LinearSVC**
                - Accuracy: ~0.593  
                - Balanced Accuracy: ~0.458  
                Sehr gute Gesamtperformance mit stabiler Accuracy.

                **Logistische Regression**
                - Accuracy: ~0.551  
                - Balanced Accuracy: ~0.535  
                Beste Fairness, besserer Recall für kleinere Genres.

                **Random Forest**
                - Accuracy: ~0.581  
                - Balanced Accuracy: ~0.405  
                Schwach bei Minderheitsgenres.

                **Fazit**
                - Beste Gesamtperformance: **LinearSVC**  
                - Beste Fairness: **Logistische Regression**  
                - Höchste Accuracy, aber verzerrt: **Random Forest**
                """

                        if "st" in name or "transformer" in name or "minilm" in name:
                            return """
                ### 📌 Transformer (MiniLM) – Zusammenfassung der Klassifikatoren

                **LinearSVC**
                - Accuracy: ~0.572  
                - Balanced Accuracy: ~0.515  
                Beste Balance zwischen Genauigkeit und Fairness.

                **Logistische Regression**
                - Accuracy: ~0.475  
                - Balanced Accuracy: ~0.543  
                Fairste und ausgewogenste Klassifikation.

                **Random Forest**
                - Accuracy: ~0.624  
                - Balanced Accuracy: ~0.343  
                Sehr hohe Accuracy, aber extrem geringe Fairness.

                **Fazit**
                - Beste Gesamtperformance: **LinearSVC**
                - Beste Fairness: **Logistische Regression**
                - Höchste Accuracy, aber schlechteste Fairness: **Random Forest**
                """
                        return ""


                    # ---------- TAB-INHALTE ----------
                    for i, (model_key, label) in enumerate(sorted_models):
                        with tabs[i]:
                            row = df_eval[df_eval["model"] == model_key].iloc[0]

                            # -- Metriken --
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Accuracy", f"{row['accuracy']:.3f}")
                            c2.metric("Balanced Accuracy", f"{row['balanced_accuracy']:.3f}")
                            c3.metric("F1-Macro", f"{row['f1_macro']:.3f}")

                            st.write(f"**Embedding:** {row['embedding']}")
                            st.write(f"**Classifier:** {row['classifier']}")

                            # -- Confusion Matrix --
                            st.markdown("---")
                            st.subheader("Confusion Matrix")
                            cm_img_path = os.path.join(CM_DIR, f"cm_{row['model']}.png")

                            if os.path.exists(cm_img_path):
                                st.image(cm_img_path, use_container_width=True)
                            else:
                                st.info(f"Keine Grafik gefunden: `{cm_img_path}`")

                            # -- Beschreibung --
                            desc = get_model_description_de(model_key)
                            if desc:
                                st.markdown("---")
                                st.markdown(desc)
                # --------------------------------------------------
                # 3. Finale Modellwahl (einmal, außerhalb der Tabs)
                # --------------------------------------------------
                st.markdown("---")
                st.subheader("3. Finale Modellwahl & Modellselektion")

                st.markdown("""
    Über alle drei Embedding-Strategien – **Word2Vec**, **TF-IDF** und **Transformer (MiniLM)** – zeigt sich ein konsistentes Muster:

    - **LinearSVC** liefert die stabilste Gesamtperformance, unabhängig vom Embedding.  
    - **Logistische Regression** verbessert systematisch die Klassenbalance und den Recall für Minderheitsgenres.  
    - **Random Forest** erreicht oft hohe Accuracy, ist aber deutlich zugunsten der Mehrheitsklassen verzerrt und erzielt eine niedrige Balanced Accuracy.

    ### 🎯 Final gewähltes Modell

    **SentenceTransformer (MiniLM) + LinearSVC**

    Dieses Modell bietet:

    - solide Accuracy (~0.57)  
    - die beste Balanced Accuracy unter den leistungsstarken Modellen (~0.52)  
    - gute Performance sowohl für dominante als auch für Minderheitsgenres  
    - robuste Generalisierung dank semantisch reichhaltiger Transformer-Embeddings  

    In Kombination mit **LinearSVC**, das sehr stabil auf hochdimensionalen Embeddings arbeitet, ergibt sich ein Modell, das eine gute Balance zwischen Performance und Fairness über alle Genres hinweg bietet.
    """)


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
    st.header("1. Load Trained Model and Label Encoder")

    st.markdown("""
    Laden des im Kapitel *Model Evaluation* gewählten **finalen Klassifikationsmodells**
    (SentenceTransformer + LinearSVC) sowie des `LabelEncoder` für die Genres.
    """)
    st.code(
        """
clf_st_svc = joblib.load("models/clf_st_svc.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
""",
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
        """
lyrics = \"\"\" 
Yeah I'm driving through the city late at night,
lights low, bass loud, trouble on my mind...
\"\"\"

lyrics_clean = lyrics.strip()

embedding_tensor = st_model.encode(
    [lyrics_clean],
    batch_size=16,
    show_progress_bar=False,
    convert_to_numpy=False,
    convert_to_tensor=True,
)

embedding = embedding_tensor.tolist()

pred_idx = clf_st_svc.predict(embedding)[0]
pred_genre = label_encoder.inverse_transform([pred_idx])[0]
""",
        language="python",
    )

    st.subheader("2.2 Classification of more Lyrics")
    st.markdown("""
    Im zweiten Schritt werden mehrere kurze Beispiel-Lyrics in einem Rutsch
    klassifiziert, um das Modellverhalten zu demonstrieren.
    """)
    st.code(
        """
texts = [
    "Yeah, I'm riding through the city with my homies late at night...",
    "Baby, I miss you every single day, I can't get you off my mind...",
    "Whiskey on the dashboard, small town lights and dusty roads...",
    "The crowd is roaring, the drums are loud, the stage is burning..."
]

emb = st_model.encode(
    [t.strip() for t in texts],
    convert_to_numpy=False,
    convert_to_tensor=True,
    show_progress_bar=False,
)
emb_list = emb.tolist()

pred_idx = clf_st_svc.predict(emb_list)
pred_genres = label_encoder.inverse_transform(pred_idx)

""",
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
        """
df = pd.read_csv("data/clean/data.csv")

df[["lyrics", "tag"]].head()""",
        language="python",
    )

    st.subheader("1.2 Data Preparation")
    st.markdown("""
    Alle vorhandenen Lyrics werden gesammelt und zu einem großen Text-Korpus
    zusammengefügt, der als Trainingsbasis für das Markov-Modell dient.
    """)
    st.code(
        """
all_lyrics = df["lyrics"].dropna().tolist()

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
        """
text_model_all = markovify.Text(corpus_text, state_size=2)""",
        language="python",
    )

    st.subheader("2.2 Generate a few lines")
    st.markdown("Generiert einige Beispiel-Zeilen aus dem **gesamten Korpus**.")
    st.code(
        """
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
        """
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
        """
def generate_line(model, max_tries=100):
    line = model.make_short_sentence(max_chars=90, tries=100)
    return line if line else ""

def generate_verse(model, num_lines=8):
    lines = []
    for _ in range(num_lines):
        line = generate_line(model)
        if line:
            lines.append(line)
    return lines

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
