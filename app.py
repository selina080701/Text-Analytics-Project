import streamlit as st

# ===============================================
# Grundeinstellungen der App
# ===============================================
st.set_page_config(
    page_title="Text Analytics on Song Lyrics",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Projekt-Überblick",
        "1. Daten laden (Subset)",
        "2. Data Cleaning",
        "3. Tokenization",
        "4. Statistische Analyse",
        "5. Word Embeddings",
    ]
)

# ===============================================
# Hilfs-Funktion für Sektionstitel
# ===============================================
def section_header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.markdown(f"### {subtitle}")
    st.markdown("---")


# ===============================================
# Seite: Projekt-Überblick
# ===============================================
if page == "Projekt-Überblick":
    section_header("🎵 Text Analytics on Song Lyrics", "Projekt-Überblick")

    st.markdown(
        """
        Dieses Projekt analysiert Songtexte aus dem **Genius Song Lyrics Dataset**.  
        Ziel ist es, typische NLP-Methoden auf Lyrics anzuwenden:

        **Schritte im Projekt:**
        1. Sampling eines Subsets  
        2. Cleaning und Normalisierung  
        3. Tokenisierung & Stopword Removal  
        4. Statistische Analyse  
        5. Word Embeddings & Visualisierung  
        """
    )

    with st.expander("ℹ️ Hinweise zur Dokumentation", expanded=False):
        st.info("Hier kannst du später Projektziele, Diagramme, Architektur, Workflow-Grafiken usw. ergänzen.")


# ==============================
# Seite: 1. Daten laden (Subset)
# ==============================
elif page == "1. Daten laden (Subset)":
    import os
    from datasets import load_dataset
    import pandas as pd

    st.title("📥 1. Daten laden (Subset)")
    st.markdown(
        """
        Dieses Modul lädt das **Genius Song Lyrics Dataset** von Hugging Face
        und erstellt daraus ein **zufälliges Subset** (z. B. 1 %), das wir
        direkt in der App anzeigen und optional als CSV speichern können.
        """
    )

    st.markdown("### 1️⃣ Einstellungen für das Subset")

    # Auswahl der Subset-Größe in Prozent
    subset_fraction = st.slider(
        "Wähle die Größe des Subsets in Prozent:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Es wird ein zufälliges Subset des Datasets mit dieser Größe erstellt."
    )

    # Pfad für optionales Speichern
    output_dir = "data_subsets"
    output_path = os.path.join(output_dir, f"lyrics_subset_{subset_fraction}pct.csv")

    col1, col2 = st.columns(2)
    with col1:
        load_button = st.button("🚀 Subset laden")
    with col2:
        save_checkbox = st.checkbox("Subset zusätzlich als CSV speichern", value=False)

    if load_button:
        with st.spinner("Lade Dataset von Hugging Face und erstelle Subset..."):
            # 1. Original-Dataset von Hugging Face laden
            dataset = load_dataset("sebastiandizon/genius-song-lyrics", split="train")

            # 2. Subset-Größe berechnen
            subset_size = int(len(dataset) * (subset_fraction / 100))

            # 3. Shuffle für Reproduzierbarkeit & Auswahl des Subsets
            dataset = dataset.shuffle(seed=42)
            dataset_small = dataset.select(range(subset_size))

            # 4. In pandas DataFrame konvertieren
            df = pd.DataFrame(dataset_small)

        st.success(f"✅ Subset erfolgreich geladen mit {len(df):,} Einträgen.")

        st.markdown("### 2️⃣ Vorschau auf das Subset")
        st.dataframe(df.head(), use_container_width=True)

        # Optional: lokal als CSV speichern
        if save_checkbox:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_path, index=False)
            st.info(f"💾 CSV gespeichert unter: `{output_path}`")

        # Download-Button direkt aus der App
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Subset als CSV herunterladen",
            data=csv_bytes,
            file_name=f"lyrics_subset_{subset_fraction}pct.csv",
            mime="text/csv",
        )

        # Kleine Zusatzinfo-Box
        with st.expander("📌 Technische Details (aus dem Notebook)"):
            st.markdown(
                """
                **Schritte intern:**
                ```python
                from datasets import load_dataset
                import pandas as pd
                import os

                dataset = load_dataset("sebastiandizon/genius-song-lyrics", split="train")

                subset_fraction = 1
                subset_size = int(len(dataset) * (subset_fraction / 100))

                dataset = dataset.shuffle(seed=42)
                dataset_small = dataset.select(range(subset_size))

                df = pd.DataFrame(dataset_small)
                ```
                """
            )
    else:
        st.info("⬅️ Wähle oben einen Prozentwert und klicke auf **„🚀 Subset laden“**, um zu starten.")

# ==============================
# Seite: 2. Data Cleaning
# ==============================
elif page == "2. Data Cleaning":
    import os
    import re
    import pandas as pd

    st.title("🧼 2. Data Cleaning")
    st.markdown(
        """
        In diesem Schritt werden die geladenen Songtexte **bereinigt**:

        - Vereinheitlichung (z. B. Kleinschreibung)
        - Entfernen von Tags wie `[Chorus]`, `[Verse 1]`, etc.
        - Entfernen von Sonderzeichen und mehrfachen Leerzeichen
        - Vergleich der Textlängen **vorher / nachher**
        """
    )

    st.markdown("### 1️⃣ Datengrundlage wählen")

    # Option A: CSV-Datei aus vorherigem Schritt (Standardpfad)
    default_dir = "data_subsets"
    csv_files = []
    if os.path.isdir(default_dir):
        csv_files = [f for f in os.listdir(default_dir) if f.endswith(".csv")]

    col1, col2 = st.columns(2)

    with col1:
        selected_csv = None
        if csv_files:
            selected_csv = st.selectbox(
                "Vorhandene Subset-CSV aus Ordner `data_subsets` auswählen:",
                options=csv_files,
                index=0,
            )

    with col2:
        uploaded_file = st.file_uploader(
            "Oder eigene CSV-Datei hochladen:",
            type=["csv"]
        )

    df = None

    # Daten laden
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV aus Upload geladen.")
    elif selected_csv is not None:
        csv_path = os.path.join(default_dir, selected_csv)
        df = pd.read_csv(csv_path)
        st.success(f"✅ CSV aus `{csv_path}` geladen.")
    else:
        st.info(
            "Bitte lade eine CSV-Datei hoch **oder** lege zuerst ein Subset in "
            "`data_subsets/` an (Schritt 1)."
        )

    if df is not None:
        st.markdown("### 2️⃣ Rohdaten-Vorschau")
        st.dataframe(df.head(), use_container_width=True)

        # Textspalte auswählen
        st.markdown("### 3️⃣ Textspalte für Cleaning auswählen")

        # Versuch, eine sinnvolle Default-Spalte vorzuschlagen
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if "lyrics" in df.columns:
            default_col = "lyrics"
        elif len(text_columns) > 0:
            default_col = text_columns[0]
        else:
            default_col = None

        if default_col is None:
            st.error(
                "In der CSV wurden keine Textspalten (dtype=object) gefunden. "
                "Bitte überprüfe die Datei."
            )
        else:
            text_col = st.selectbox(
                "Textspalte für die Bereinigung:",
                options=text_columns,
                index=text_columns.index(default_col),
                help="Typischerweise ist dies z. B. `lyrics` oder `text`."
            )

            st.markdown("### 4️⃣ Cleaning-Optionen")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                opt_lower = st.checkbox("Kleinschreibung", value=True)
                opt_strip_tags = st.checkbox("Tags wie [Chorus] entfernen", value=True)
            with col_b:
                opt_remove_punct = st.checkbox("Sonderzeichen entfernen", value=True)
                opt_remove_digits = st.checkbox("Zahlen entfernen", value=False)
            with col_c:
                opt_collapse_spaces = st.checkbox("Mehrfache Leerzeichen reduzieren", value=True)

            def clean_text(text: str) -> str:
                if not isinstance(text, str):
                    return ""

                # 1) Kleinschreibung
                if opt_lower:
                    text = text.lower()

                # 2) Tags wie [Chorus], [Verse 1], etc.
                if opt_strip_tags:
                    text = re.sub(r"\[.*?\]", " ", text)

                # 3) Sonderzeichen (alles außer Buchstaben/Zahlen/Leerzeichen)
                if opt_remove_punct:
                    text = re.sub(r"[^\w\s]", " ", text)

                # 4) Zahlen entfernen
                if opt_remove_digits:
                    text = re.sub(r"\d+", " ", text)

                # 5) Mehrfache Leerzeichen zu einem
                if opt_collapse_spaces:
                    text = re.sub(r"\s+", " ", text).strip()

                return text

            if st.button("🧼 Text bereinigen"):
                with st.spinner("Bereinige Texte..."):
                    df["text_raw"] = df[text_col].astype(str)
                    df["text_clean"] = df["text_raw"].apply(clean_text)

                    # Längen vor/nachher
                    df["len_raw"] = df["text_raw"].str.len()
                    df["len_clean"] = df["text_clean"].str.len()

                st.success("✅ Cleaning abgeschlossen.")

                st.markdown("### 5️⃣ Beispiel: Vorher / Nachher")
                st.dataframe(
                    df[["text_raw", "text_clean", "len_raw", "len_clean"]].head(10),
                    use_container_width=True
                )

                st.markdown("### 6️⃣ Statistik: Textlängen")

                col_len1, col_len2 = st.columns(2)
                with col_len1:
                    st.metric(
                        "Ø Länge (raw)",
                        f"{df['len_raw'].mean():.1f} Zeichen"
                    )
                with col_len2:
                    st.metric(
                        "Ø Länge (clean)",
                        f"{df['len_clean'].mean():.1f} Zeichen"
                    )

                # Download der bereinigten Daten
                cleaned_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Bereinigte Daten als CSV herunterladen",
                    data=cleaned_csv,
                    file_name="lyrics_cleaned.csv",
                    mime="text/csv",
                )

                with st.expander("📌 Hinweis zu den Cleaning-Schritten"):
                    st.markdown(
                        """
                        **Zusammenfassung der verwendeten Regeln:**

                        - `lower()` → alles in Kleinschreibung  
                        - Regex `\\[.*?\\]` → entfernt Blocke wie `[Chorus]`, `[Verse 1]`  
                        - Regex `[^\w\s]` → entfernt Sonderzeichen, lässt nur Buchstaben/Zahlen/Leerzeichen  
                        - Optional: Entfernen von Ziffern (`\\d+`)  
                        - Reduktion mehrfacher Leerzeichen zu einem Leerzeichen  
                        """
                    )
            else:
                st.info("Klicke auf **„🧼 Text bereinigen“**, um die Cleaning-Schritte auszuführen.")


# ==============================
# Seite: 3. Tokenization
# ==============================
elif page == "3. Tokenization":
    import os
    import re
    import pandas as pd
    from collections import Counter

    st.title("✂️ 3. Tokenization & Stopwords")
    st.markdown(
        """
        In diesem Schritt werden die bereinigten Songtexte in **Tokens** zerlegt
        und häufige Wörter analysiert.

        Basierend auf dem Notebook `tokenization.ipynb`:
        - Laden des **bereinigten Datasets**
        - Erzeugen einer einfachen Token-Liste pro Song (`words`)
        - Entfernen von Stopwörtern → `tokens`
        - Berechnung von Wort- und Tokenanzahl pro Song
        - Häufigste Wörter vor / nach Stopword-Entfernung
        """
    )

    st.markdown("### 1️⃣ Datensatz für die Tokenization wählen")

    # Standard-Ordner aus dem Notebook
    default_dir = "data/clean"
    csv_files = []
    if os.path.isdir(default_dir):
        csv_files = [f for f in os.listdir(default_dir) if f.endswith(".csv")]

    col1, col2 = st.columns(2)

    with col1:
        selected_csv = None
        if csv_files:
            selected_csv = st.selectbox(
                "Bereinigte CSV aus Ordner `data/clean` auswählen:",
                options=csv_files,
                index=0,
                help="z. B. `lyrics_subset_1pct_clean.csv` aus dem Cleaning-Notebook."
            )

    with col2:
        uploaded_file = st.file_uploader(
            "Oder eigene (bereinigte) CSV-Datei hochladen:",
            type=["csv"]
        )

    df = None

    # Daten laden
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV aus Upload geladen.")
    elif selected_csv is not None:
        csv_path = os.path.join(default_dir, selected_csv)
        df = pd.read_csv(csv_path)
        st.success(f"✅ CSV aus `{csv_path}` geladen.")
    else:
        st.info(
            "Bitte wähle eine CSV-Datei aus `data/clean/` oder lade eine Datei hoch.\n"
            "Diese sollte bereits aus Schritt 2 (Data Cleaning) stammen."
        )

    if df is not None:
        st.markdown("### 2️⃣ Überblick über den Datensatz")

        # Optional: Filter auf englische Songs, wie im Notebook
        if "language_cld3" in df.columns:
            only_en = st.checkbox(
                "Nur englische Songs verwenden (language_cld3 == 'en')",
                value=True
            )
            if only_en:
                df = df[df["language_cld3"] == "en"]

        st.write(f"**Form des DataFrames:** {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
        if "artist" in df.columns:
            st.write(f"**Anzahl Artists:** {df['artist'].nunique()}")

        st.dataframe(df.head(), use_container_width=True)

        # Textspalte auswählen
        st.markdown("### 3️⃣ Textspalte für die Tokenization auswählen")

        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if "lyrics" in df.columns:
            default_text_col = "lyrics"
        elif len(text_columns) > 0:
            default_text_col = text_columns[0]
        else:
            default_text_col = None

        if default_text_col is None:
            st.error(
                "Es wurden keine geeigneten Textspalten gefunden. "
                "Bitte prüfe, ob eine Spalte mit Lyrics/Text vorhanden ist."
            )
        else:
            text_col = st.selectbox(
                "Textspalte:",
                options=text_columns,
                index=text_columns.index(default_text_col),
                help="Typischerweise `lyrics`."
            )

            st.markdown("### 4️⃣ Tokenization-Einstellungen (wie im Notebook)")

            col_a, col_b = st.columns(2)
            with col_a:
                lowercase = st.checkbox("Kleinschreibung (lowercase)", value=True)
            with col_b:
                letters_only = st.checkbox(
                    "Nur Buchstaben (a–z) behalten",
                    value=True,
                    help="Entspricht dem Regex-Filter `[^a-z\\s]` aus dem Notebook."
                )

            # Funktionen aus dem Notebook nachgebaut
            def preprocess_text(text, lowercase: bool = True):
                """Clean and tokenize text (wie in tokenization.ipynb)"""
                if not isinstance(text, str):
                    return []
                if lowercase:
                    text = text.lower()
                if letters_only:
                    text = re.sub(r"[^a-z\s]", "", text)
                tokens = text.split()
                return tokens

            STOPWORDS = {
                "the","a","an","and","or","but","if","then","so","than","that","those","these","this",
                "to","of","in","on","for","with","as","at","by","from","into","over","under","up","down",
                "is","am","are","was","were","be","been","being","do","does","did","doing","have","has","had",
                "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their",
                "not","no","yes","yeah","y'all","yall","im","i'm","i’d","i'd","i’ll","i'll","youre","you're","dont","don't",
                "cant","can't","ill","i’ll","id","i'd","ive","i’ve","ya","oh","ooh","la","na","nah"
            }

            def filtered_tokens(text):
                """Filter Stopwörter (wie im Notebook)"""
                tokens = preprocess_text(text)
                return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 1]

            top_n = st.slider(
                "Anzahl der Top-Wörter für die Statistik:",
                min_value=5,
                max_value=30,
                value=15,
                step=1
            )

            if st.button("🚀 Tokenization starten"):
                with st.spinner("Tokenisiere Texte und berechne Statistiken..."):
                    # 1) Roh-Tokens ("words")
                    df["words"] = df[text_col].apply(preprocess_text)
                    df["word_count"] = df["words"].apply(len)

                    # 2) Tokens ohne Stopwörter ("tokens")
                    df["tokens"] = df[text_col].apply(filtered_tokens)
                    df["token_count"] = df["tokens"].apply(len)

                st.success("✅ Tokenization abgeschlossen.")

                st.markdown("### 5️⃣ Beispiel: Tokens pro Song")
                preview_cols = ["words", "word_count", "tokens", "token_count"]
                meta_cols = []
                for c in ["title", "artist"]:
                    if c in df.columns:
                        meta_cols.append(c)

                show_cols = meta_cols + preview_cols
                st.dataframe(df[show_cols].head(10), use_container_width=True)

                st.markdown("### 6️⃣ Häufigste Wörter (vor / nach Stopword-Entfernung)")

                # Flatten-Listen für alle Wörter
                all_words = [t for row in df["words"] for t in row]
                all_tokens = [t for row in df["tokens"] for t in row]

                word_counts_raw = Counter(all_words).most_common(top_n)
                word_counts_filtered = Counter(all_tokens).most_common(top_n)

                df_raw = pd.DataFrame(word_counts_raw, columns=["word", "count"])
                df_filtered = pd.DataFrame(word_counts_filtered, columns=["word", "count"])

                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader(f"Top {top_n} Wörter (roh)")
                    st.dataframe(df_raw, use_container_width=True)
                    st.bar_chart(df_raw.set_index("word"))

                with col_right:
                    st.subheader(f"Top {top_n} Wörter (ohne Stopwörter)")
                    st.dataframe(df_filtered, use_container_width=True)
                    st.bar_chart(df_filtered.set_index("word"))

                st.markdown("### 7️⃣ Datensatz mit Tokens speichern")
                tokenized_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Tokenisierten Datensatz als CSV herunterladen",
                    data=tokenized_csv,
                    file_name="lyrics_tokenized.csv",
                    mime="text/csv",
                )

                with st.expander("📌 Technische Details (aus dem Notebook)"):
                    st.markdown(
                        """
                        Im ursprünglichen Notebook wurden u. a. folgende Schritte ausgeführt:

                        ```python
                        import re
                        from collections import Counter

                        def preprocess_text(text, lowercase=True):
                            if not isinstance(text, str):
                                return []
                            if lowercase:
                                text = text.lower()
                            text = re.sub(r"[^a-z\\s]", "", text)
                            tokens = text.split()
                            return tokens

                        df["words"] = df["lyrics"].apply(preprocess_text)
                        df["word_count"] = df["words"].apply(len)

                        def filtered_tokens(text):
                            tokens = preprocess_text(text)
                            return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 1]

                        df["tokens"] = df["lyrics"].apply(filtered_tokens)
                        df["token_count"] = df["tokens"].apply(len)
                        ```
                        """
                    )
            else:
                st.info("Klicke auf **„🚀 Tokenization starten“**, um Tokens zu erzeugen und Statistiken zu sehen.")

# ==============================
# Seite: 4. Statistische Analyse
# ==============================
elif page == "4. Statistische Analyse":
    import os
    import ast
    import numpy as np
    import pandas as pd
    from collections import Counter
    from itertools import islice

    st.title("📊 4. Statistische Analyse")
    st.markdown(
        """
        In diesem Modul werden die **tokenisierten Lyrics** statistisch ausgewertet.

        Typische Fragestellungen:
        - Wie lang sind Songs im Schnitt (Wörter / Tokens)?
        - Wie verteilen sich Genres im Datensatz?
        - Welche Artists haben besonders viele Songs?
        - Welche Wörter/N-Gramme kommen häufig vor?
        """
    )

    # --------------------------
    # 1️⃣ Datensatz wählen
    # --------------------------
    st.markdown("### 1️⃣ Datensatz wählen")

    default_dir = "data/clean"
    csv_files = []
    if os.path.isdir(default_dir):
        csv_files = [f for f in os.listdir(default_dir) if f.endswith(".csv")]

    col1, col2 = st.columns(2)

    with col1:
        selected_csv = None
        if csv_files:
            selected_csv = st.selectbox(
                "CSV aus Ordner `data/clean` auswählen:",
                options=csv_files,
                index=0,
                help="z. B. die Datei aus Schritt 3 (Tokenization)."
            )

    with col2:
        uploaded_file = st.file_uploader(
            "Oder eigene (tokenisierte) CSV-Datei hochladen:",
            type=["csv"]
        )

    df = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV aus Upload geladen.")
    elif selected_csv is not None:
        csv_path = os.path.join(default_dir, selected_csv)
        df = pd.read_csv(csv_path)
        st.success(f"✅ CSV aus `{csv_path}` geladen.")
    else:
        st.info(
            "Bitte wähle eine CSV-Datei aus `data/clean/` oder lade eine Datei hoch.\n"
            "Die Datei sollte idealerweise Spalten wie `word_count`, `token_count`, `words`, `tokens`, `tag`, `artist` enthalten."
        )

    if df is not None:
        # -----------------------------------
        # 2️⃣ Grundlegende Übersicht
        # -----------------------------------
        st.markdown("### 2️⃣ Überblick über den Datensatz")

        # words/tokens evtl. von String → Liste umwandeln
        for col in ["words", "tokens"]:
            if col in df.columns and len(df[col]) > 0 and isinstance(df[col].iloc[0], str):
                try:
                    df[col] = df[col].apply(ast.literal_eval)
                except Exception:
                    pass  # falls schon Listen vorhanden sind

        st.write(f"**Zeilen (Songs):** {len(df):,}")
        if "artist" in df.columns:
            st.write(f"**Anzahl Artists:** {df['artist'].nunique():,}")
        if "tag" in df.columns:
            st.write(f"**Anzahl Genres (Tags):** {df['tag'].nunique():,}")

        with st.expander("📄 Datensatz-Vorschau"):
            st.dataframe(df.head(), use_container_width=True)

        # -----------------------------------
        # 3️⃣ Grundlegende Statistiken
        # -----------------------------------
        st.markdown("### 3️⃣ Statistiken zu Wort- & Tokenanzahl")

        if "word_count" in df.columns or "token_count" in df.columns:
            col_wc, col_tc = st.columns(2)

            if "word_count" in df.columns:
                with col_wc:
                    st.subheader("Word Count (roh)")
                    st.metric("Ø Wörter pro Song", f"{df['word_count'].mean():.1f}")
                    st.metric("Median Wörter pro Song", f"{df['word_count'].median():.1f}")
                    st.metric("Max Wörter pro Song", f"{df['word_count'].max():.0f}")

                    # Histogramm über Binning
                    wc = df["word_count"].clip(upper=df["word_count"].quantile(0.99))
                    bins = np.linspace(wc.min(), wc.max(), 21)
                    wc_binned = pd.cut(wc, bins=bins)
                    wc_hist = wc_binned.value_counts().sort_index()
                    hist_df = pd.DataFrame({"bin": wc_hist.index.astype(str), "count": wc_hist.values}).set_index("bin")
                    st.bar_chart(hist_df)

            if "token_count" in df.columns:
                with col_tc:
                    st.subheader("Token Count (ohne Stopwörter)")
                    st.metric("Ø Tokens pro Song", f"{df['token_count'].mean():.1f}")
                    st.metric("Median Tokens pro Song", f"{df['token_count'].median():.1f}")
                    st.metric("Max Tokens pro Song", f"{df['token_count'].max():.0f}")

                    tc = df["token_count"].clip(upper=df["token_count"].quantile(0.99))
                    bins = np.linspace(tc.min(), tc.max(), 21)
                    tc_binned = pd.cut(tc, bins=bins)
                    tc_hist = tc_binned.value_counts().sort_index()
                    hist_df2 = pd.DataFrame({"bin": tc_hist.index.astype(str), "count": tc_hist.values}).set_index("bin")
                    st.bar_chart(hist_df2)
        else:
            st.warning("Es wurden keine Spalten `word_count` oder `token_count` gefunden – bitte stelle sicher, dass die Datei aus Schritt 3 stammt.")

        # -----------------------------------
        # 4️⃣ Genre-Verteilung & Artist-Stats
        # -----------------------------------
        st.markdown("### 4️⃣ Verteilungen nach Genre und Artist")

        col_g, col_a = st.columns(2)

        if "tag" in df.columns:
            with col_g:
                st.subheader("Genre-Verteilung (Tag)")
                genre_counts = df["tag"].value_counts().sort_values(ascending=False)
                genre_df = pd.DataFrame({"genre": genre_counts.index, "count": genre_counts.values}).set_index("genre")
                st.dataframe(genre_df)
                st.bar_chart(genre_df)

        if "artist" in df.columns and "token_count" in df.columns:
            with col_a:
                st.subheader("Top Artists nach Anzahl Songs")

                artist_stats = (
                    df.groupby("artist")
                    .agg(
                        songs=("artist", "count"),
                        avg_tokens=("token_count", "mean")
                    )
                    .sort_values("songs", ascending=False)
                    .head(20)
                )
                st.dataframe(artist_stats)
                st.bar_chart(artist_stats[["songs"]])

        # -----------------------------------
        # 5️⃣ Häufigste Tokens (global)
        # -----------------------------------
        if "tokens" in df.columns:
            st.markdown("### 5️⃣ Häufigste Tokens im gesamten Dataset")

            top_n = st.slider(
                "Anzahl der Top-Tokens:",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            )

            # Flatten aller Tokens
            all_tokens = []
            for row in df["tokens"]:
                if isinstance(row, list):
                    all_tokens.extend(row)

            token_counts = Counter(all_tokens)
            most_common_tokens = token_counts.most_common(top_n)
            token_df = pd.DataFrame(most_common_tokens, columns=["token", "count"]).set_index("token")

            st.dataframe(token_df)
            st.bar_chart(token_df)

        # -----------------------------------
        # 6️⃣ N-Gramm-Analyse nach Genre
        # -----------------------------------
        st.markdown("### 6️⃣ N-Gramm-Analyse (z. B. pro Genre)")

        if "tokens" in df.columns and "tag" in df.columns:
            n_val = st.selectbox("N-Gramm-Länge (n):", options=[1, 2, 3], index=0)
            group_level = st.selectbox(
                "Gruppierung für N-Gramme:",
                options=["tag", "artist"],
                index=0,
                help="In vielen Fällen ist `tag` = Genre sinnvoll."
            )

            max_groups = st.slider(
                "Max. Anzahl Gruppen für die Anzeige:",
                min_value=3,
                max_value=15,
                value=6
            )

            def most_common_ngram_for_group(group_df: pd.DataFrame, label_col: str, n: int) -> pd.DataFrame:
                """
                Für jede Gruppe (z. B. Genre oder Artist) das häufigste n-Gramm berechnen.
                """
                rows = []
                for label, sub in group_df.groupby(label_col):
                    c = Counter()
                    for toks in sub["tokens"]:
                        if isinstance(toks, list) and len(toks) >= n:
                            # n-Gramme bilden
                            for i in range(len(toks) - n + 1):
                                ngram = tuple(toks[i:i+n])
                                c[ngram] += 1
                    if c:
                        top_ngram, cnt = c.most_common(1)[0]
                        rows.append({
                            label_col: label,
                            "ngram": " ".join(top_ngram),
                            "count": cnt,
                            "songs": len(sub)
                        })
                    else:
                        rows.append({
                            label_col: label,
                            "ngram": None,
                            "count": 0,
                            "songs": len(sub)
                        })
                return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

            if st.button("🔍 N-Gramme berechnen"):
                with st.spinner("Berechne häufigste N-Gramme pro Gruppe..."):
                    subset = df.copy()
                    # Optional: nur Gruppen mit Mindestanzahl Songs berücksichtigen
                    group_counts = subset[group_level].value_counts()
                    valid_groups = group_counts[group_counts >= 20].index  # z. B. mind. 20 Songs
                    subset = subset[subset[group_level].isin(valid_groups)]

                    ngram_df = most_common_ngram_for_group(subset, label_col=group_level, n=n_val)
                    ngram_df = ngram_df.head(max_groups)

                st.success("✅ N-Gramm-Auswertung abgeschlossen.")

                st.write(f"**Top {len(ngram_df)} Gruppen (`{group_level}`) mit häufigsten {n_val}-Grammen:**")
                st.dataframe(ngram_df, use_container_width=True)

                # Balkendiagramm
                if not ngram_df.empty:
                    plot_df = ngram_df.copy()
                    plot_df["label"] = plot_df[group_level] + " — " + plot_df["ngram"].fillna("")
                    plot_df = plot_df.set_index("label")[["count"]]
                    st.bar_chart(plot_df)
        else:
            st.info(
                "Für die N-Gramm-Analyse werden die Spalten `tokens` und `tag` benötigt.\n"
                "Stelle sicher, dass der Datensatz aus Schritt 3 (Tokenization) stammt und Genres enthält."
            )

# ==============================
# Seite: 5. Word Embeddings
# ==============================
elif page == "5. Word Embeddings":
    import os
    import ast
    import numpy as np
    import pandas as pd
    from gensim.models import Word2Vec
    from sklearn.decomposition import PCA
    import plotly.express as px

    st.title("🧠 5. Word Embeddings")
    st.markdown(
        """
        In diesem Modul werden **Word Embeddings** auf Basis der tokenisierten Lyrics erzeugt
        und interaktiv untersucht.

        Typische Fragen:
        - Welche Wörter liegen semantisch nahe beieinander?
        - Lassen sich Themen / Genres im Vektorraum erkennen?
        - Welche Songs (Dokumente) sind sich im „Embedding-Space“ ähnlich?
        """
    )

    # ---------------------------------
    # 1️⃣ Datensatz mit Tokens wählen
    # ---------------------------------
    st.markdown("### 1️⃣ Datensatz mit Tokens wählen")

    default_dir = "data/clean"
    csv_files = []
    if os.path.isdir(default_dir):
        csv_files = [f for f in os.listdir(default_dir) if f.endswith(".csv")]

    col1, col2 = st.columns(2)

    with col1:
        selected_csv = None
        if csv_files:
            selected_csv = st.selectbox(
                "CSV aus Ordner `data/clean` auswählen:",
                options=csv_files,
                index=0,
                help="z. B. die tokenisierte Datei aus Schritt 3/4."
            )

    with col2:
        uploaded_file = st.file_uploader(
            "Oder eigene (tokenisierte) CSV-Datei hochladen:",
            type=["csv"]
        )

    df = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV aus Upload geladen.")
    elif selected_csv is not None:
        csv_path = os.path.join(default_dir, selected_csv)
        df = pd.read_csv(csv_path)
        st.success(f"✅ CSV aus `{csv_path}` geladen.")
    else:
        st.info(
            "Bitte wähle eine CSV-Datei aus `data/clean/` oder lade eine Datei hoch.\n"
            "Die Datei sollte eine Spalte `tokens` enthalten (Liste von Wörtern pro Song)."
        )

    if df is not None:
        # tokens-Spalte in echte Listen umwandeln (falls als String gespeichert)
        if "tokens" not in df.columns:
            st.error("In der geladenen CSV wurde keine Spalte `tokens` gefunden.")
        else:
            if len(df["tokens"]) > 0 and isinstance(df["tokens"].iloc[0], str):
                try:
                    df["tokens"] = df["tokens"].apply(ast.literal_eval)
                except Exception:
                    st.warning("Konnte `tokens` nicht sicher in Listen umwandeln – bitte Format prüfen.")

            st.markdown("### 2️⃣ Datensatz-Vorschau")
            preview_cols = []
            for col in ["title", "artist", "tag", "tokens"]:
                if col in df.columns:
                    preview_cols.append(col)
            st.dataframe(df[preview_cols].head(10), use_container_width=True)

            # ---------------------------------
            # 2️⃣ Word2Vec-Training konfigurieren
            # ---------------------------------
            st.markdown("### 3️⃣ Word2Vec-Training konfigurieren")

            col_left, col_right = st.columns(2)
            with col_left:
                vector_size = st.slider("Vektordimension (vector_size)", 10, 200, 50, 10)
                window = st.slider("Kontextfenster (window)", 2, 10, 5, 1)
                min_count = st.slider("min_count (Minimale Wortfrequenz)", 1, 20, 5, 1)

            with col_right:
                sg_model = st.selectbox(
                    "Trainingsmodus",
                    options=[("CBOW (sg=0)", 0), ("Skip-gram (sg=1)", 1)],
                    format_func=lambda x: x[0],
                    index=1,
                )[1]
                epochs = st.slider("Trainingsepochen", 5, 50, 15, 5)

                max_sentences = st.slider(
                    "Max. Anzahl Songs fürs Training (Sampling)",
                    min_value=100,
                    max_value=min(10000, len(df)),
                    value=min(2000, len(df)),
                    step=100,
                    help="Zur Beschleunigung: nicht den kompletten Datensatz verwenden."
                )

            # Sampling der Sätze
            df_train = df.sample(n=max_sentences, random_state=42) if len(df) > max_sentences else df.copy()
            sentences = [toks for toks in df_train["tokens"] if isinstance(toks, list) and len(toks) > 0]

            st.write(f"➡️ Verwendete Songs fürs Training: **{len(sentences):,}**")

            if "w2v_model" not in st.session_state:
                st.session_state["w2v_model"] = None
                st.session_state["w2v_vocab_size"] = 0

            if st.button("🎓 Word2Vec trainieren"):
                if not sentences:
                    st.error("Keine gültigen Token-Listen gefunden – bitte Datensatz prüfen.")
                else:
                    with st.spinner("Trainiere Word2Vec-Modell..."):
                        model = Word2Vec(
                            sentences=sentences,
                            vector_size=vector_size,
                            window=window,
                            min_count=min_count,
                            sg=sg_model,
                            workers=4,
                            epochs=epochs,
                        )
                    st.session_state["w2v_model"] = model
                    st.session_state["w2v_vocab_size"] = len(model.wv)

                    st.success(f"✅ Modell trainiert. Vokabulargröße: **{len(model.wv):,} Wörter**")

            model = st.session_state.get("w2v_model", None)

            # ---------------------------------
            # 3️⃣ Ähnlichkeitssuche für Wörter
            # ---------------------------------
            st.markdown("### 4️⃣ Wort-Ähnlichkeiten erkunden")

            if model is None:
                st.info("Trainiere zuerst ein Word2Vec-Modell, um diese Sektion zu nutzen.")
            else:
                vocab = list(model.wv.key_to_index.keys())
                example_word = vocab[0] if vocab else ""
                query_word = st.text_input(
                    "Wort für Ähnlichkeitssuche:",
                    value=example_word,
                    help="Muss im Vokabular des Modells vorkommen."
                )
                topn = st.slider("Anzahl ähnlicher Wörter", 3, 20, 10, 1)

                if st.button("🔍 Ähnliche Wörter anzeigen"):
                    if query_word not in model.wv:
                        st.error(f"Das Wort **`{query_word}`** ist nicht im Vokabular.")
                    else:
                        sims = model.wv.most_similar(query_word, topn=topn)
                        sim_df = pd.DataFrame(sims, columns=["word", "similarity"])
                        st.subheader(f"Ähnliche Wörter zu: `{query_word}`")
                        st.dataframe(sim_df, use_container_width=True)

                        # Kleine 2D-PCA-Visualisierung der Nachbarschaft
                        words_for_plot = [query_word] + [w for w, _ in sims]
                        vectors = np.array([model.wv[w] for w in words_for_plot])
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(vectors)
                        plot_df = pd.DataFrame(
                            {
                                "x": coords[:, 0],
                                "y": coords[:, 1],
                                "word": words_for_plot,
                            }
                        )
                        fig = px.scatter(
                            plot_df,
                            x="x",
                            y="y",
                            text="word",
                            title=f"2D-PCA Nachbarschaft von '{query_word}'",
                        )
                        fig.update_traces(textposition="top center")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

            # ---------------------------------
            # 4️⃣ Visualisierung von Wort-Embeddings
            # ---------------------------------
            st.markdown("### 5️⃣ 2D-Visualisierung von Wort-Embeddings")

            if model is not None and len(model.wv) > 0:
                top_vocab_n = st.slider(
                    "Anzahl Wörter für die Visualisierung (häufigste Wörter)",
                    min_value=20,
                    max_value=min(300, len(model.wv)),
                    value=min(100, len(model.wv)),
                    step=10,
                )

                # Häufigste Wörter aus Vokabular
                # In neueren gensim Versionen: index_to_key nach Frequenz sortiert
                words_vis = model.wv.index_to_key[:top_vocab_n]
                vecs = np.array([model.wv[w] for w in words_vis])

                pca = PCA(n_components=2)
                coords = pca.fit_transform(vecs)
                plot_df2 = pd.DataFrame(
                    {"x": coords[:, 0], "y": coords[:, 1], "word": words_vis}
                )

                fig2 = px.scatter(
                    plot_df2,
                    x="x",
                    y="y",
                    text="word",
                    title="2D-PCA der Word Embeddings (Top-Wörter)",
                )
                fig2.update_traces(textposition="top center")
                fig2.update_layout(height=600)
                st.plotly_chart(fig2, use_container_width=True)
            elif model is None:
                st.info("Sobald ein Modell trainiert ist, kannst du hier die Einbettungen visualisieren.")

            # ---------------------------------
            # 5️⃣ Dokument-Embeddings (Songs) & Raum
            # ---------------------------------
            st.markdown("### 6️⃣ Dokument-Embeddings (Songs) im Vektorraum")

            if model is not None and "tokens" in df.columns:
                # Erzeuge Dokument-Embeddings als Durchschnitt der Wortvektoren
                def doc_embedding(tokens):
                    vecs = [model.wv[t] for t in tokens if t in model.wv]
                    if not vecs:
                        return None
                    return np.mean(vecs, axis=0)

                # Optionale Filter
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    filter_by_genre = False
                    selected_tag = None
                    if "tag" in df.columns:
                        filter_by_genre = st.checkbox("Nach Genre filtern", value=False)
                        if filter_by_genre:
                            tags = df["tag"].dropna().unique().tolist()
                            if tags:
                                selected_tag = st.selectbox("Genre wählen:", options=tags)

                with col_f2:
                    max_docs = st.slider(
                        "Max. Anzahl Songs für die Visualisierung",
                        min_value=30,
                        max_value=min(500, len(df)),
                        value=min(150, len(df)),
                        step=10,
                    )

                df_docs = df.copy()
                if filter_by_genre and selected_tag is not None:
                    df_docs = df_docs[df_docs["tag"] == selected_tag]

                if len(df_docs) > max_docs:
                    df_docs = df_docs.sample(n=max_docs, random_state=42)

                # Embeddings berechnen
                with st.spinner("Berechne Dokument-Embeddings..."):
                    emb_list = []
                    idx_list = []
                    for idx, row in df_docs.iterrows():
                        toks = row["tokens"]
                        if isinstance(toks, list) and len(toks) > 0:
                            emb = doc_embedding(toks)
                            if emb is not None:
                                emb_list.append(emb)
                                idx_list.append(idx)

                if not emb_list:
                    st.warning(
                        "Konnte keine Dokument-Embeddings erzeugen – möglicherweise sind zu wenige Tokens im Vokabular."
                    )
                else:
                    emb_arr = np.vstack(emb_list)
                    df_use = df_docs.loc[idx_list]

                    labels = None
                    # Titel bevorzugt, sonst Artist, sonst Index
                    for col in ["title", "artist", "tag"]:
                        if col in df_use.columns:
                            labels = df_use[col].astype(str).tolist()
                            break
                    if labels is None:
                        labels = df_use.index.astype(str).tolist()

                    pca_docs = PCA(n_components=2)
                    coords_docs = pca_docs.fit_transform(emb_arr)

                    plot_df_docs = pd.DataFrame(
                        {
                            "x": coords_docs[:, 0],
                            "y": coords_docs[:, 1],
                            "label": labels,
                        }
                    )

                    if "tag" in df_use.columns:
                        plot_df_docs["tag"] = df_use["tag"].astype(str).values
                    else:
                        plot_df_docs["tag"] = "unknown"

                    fig_docs = px.scatter(
                        plot_df_docs,
                        x="x",
                        y="y",
                        color="tag",
                        hover_name="label",
                        title="Dokument-Embeddings (Songs) in 2D (PCA)",
                    )
                    fig_docs.update_layout(height=650)
                    st.plotly_chart(fig_docs, use_container_width=True)

                    st.info(
                        "Jeder Punkt entspricht einem Song. Nähe im Plot deutet auf ähnliche Wortverwendung/Themen hin.\n"
                        "Farben repräsentieren ggf. Genres (`tag`)."
                    )

            # ---------------------------------
            # 6️⃣ Export (optional)
            # ---------------------------------
            st.markdown("### 7️⃣ Optional: Wort-Embeddings exportieren")

            if model is not None:
                if st.button("⬇️ Wort-Embeddings als CSV exportieren"):
                    words = model.wv.index_to_key
                    vectors = np.array([model.wv[w] for w in words])
                    emb_df = pd.DataFrame(vectors, index=words)
                    emb_df.index.name = "word"
                    csv_bytes = emb_df.to_csv().encode("utf-8")
                    st.download_button(
                        label="CSV herunterladen",
                        data=csv_bytes,
                        file_name="word_embeddings.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Trainiere zuerst ein Modell, um Embeddings exportieren zu können.")
