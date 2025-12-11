import os
import streamlit as st

def show_tokenization_page():
    st.title("3️⃣ Kapitel 3 – Tokenization: Genius Song Lyrics Subset (1%)")

    st.markdown("""
    **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  
    **Genres:** Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous

    **Purpose:**  
    Vorbereitung der Textdaten für weitere Analysen, indem Songtexte in einzelne Tokens
    zerlegt und Stopwörter entfernt werden.  
    Diese Schritte sorgen für eine saubere und strukturierte Darstellung der Texte und bilden
    die Grundlage für nachgelagerte NLP- und Statistik-Analysen.
    """)

    st.info(
        "**Hinweis:** Dieser Abschnitt dokumentiert die Schritte aus dem zugehörigen Notebook "
        "`tokenization.ipynb`. Die Tokenisierung der Lyrics sowie das Entfernen von Stopwörtern "
        "wurden vollständig im Notebook ausgeführt. Die Streamlit-App lädt lediglich die dort "
        "erzeugten Daten und visualisiert ausgewählte Ergebnisse – ohne die Tokenisierung erneut "
        "durchzuführen."
    )

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

