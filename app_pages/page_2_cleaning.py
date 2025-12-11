import os
import pandas as pd
import re
import streamlit as st

def show_cleaning_page():
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
