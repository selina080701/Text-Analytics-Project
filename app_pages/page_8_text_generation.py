import os
import pandas as pd
import streamlit as st
import markovify

def show_text_generation_page():
    st.title("8️⃣ Kapitel 8 - Lyrics Generation: Genius Song Lyrics (1%)")

    st.markdown(""" 
    **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  
    **Genres:** Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous

    **Purpose:**  
    Generierung neuer, stilkonsistenter Songtexte mithilfe eines einfachen
    **Markov-Chain-Modells**, das auf den bestehenden Lyrics trainiert wird.

    Unterstützte Optionen:
    - Generierung aus dem **kompletten Datensatz**
    - **Genre-spezifische** Generierung
    """)

    st.info(
        "**Hinweis:** Dieser Abschnitt dokumentiert die Schritte aus dem zugehörigen Notebook "
        "`text-generation.ipynb`. Die grundlegende Vorgehensweise zur Markov-basierten "
        "Lyrics-Generierung wurde dort entwickelt. Die Streamlit-App übernimmt diese Logik "
        "und ermöglicht eine interaktive Generierung neuer Songzeilen und Songs."
    )

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

    st.subheader("2.1 Kurze Einordnung: Markov-Ketten")

    st.markdown("""
    Markov-Modelle arbeiten nur mit lokalen Übergangswahrscheinlichkeiten: Die nächste Zeile hängt also immer nur 
    vom aktuellen Zustand bzw. den letzten Wörtern ab. Trotz dieser Einfachheit entstehen oft stilistische Muster, 
    die an die Original-Lyrics erinnern.

    Für wirklich kohärente, inhaltlich konsistente Songs wären allerdings komplexere neuronale 
    Sprachmodelle nötig. In diesem Kapitel geht es bewusst um eine leichtgewichtige, gut 
    erklärbare Demo.
    """)

    st.subheader("2.2 Build Model")
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

    st.subheader("2.3 Generate a few lines")
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

    st.subheader("2.4 Genre-specific Lyrics")
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

    st.subheader("2.5 Lyrics with Verse and Chorus")
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

    # =================================================== #
    # 🎤 Interaktive Markov-Lyrics in der App
    # =================================================== #
    st.header("🎤 Interaktive Lyrics-Generierung")

    # Für die App: Daten laden (falls nicht global)
    import pandas as pd
    import markovify
    from pathlib import Path

    @st.cache_data
    def load_lyrics_df():
        base_dir = Path(__file__).resolve().parents[1]
        data_path = base_dir / "data" / "clean" / "data.csv.gz"
        return pd.read_csv(data_path, compression="gzip")

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

