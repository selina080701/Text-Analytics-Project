import os
import pandas as pd
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

def show_text_classification_page():
    st.title("7️⃣ Kapitel 7 - Text Classification: Genius Song Lyrics (1%)")

    st.markdown(""" 
    **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  
    **Genres:** Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous

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

    st.info(
        "**Hinweis:** Dieser Abschnitt basiert auf dem zugehörigen Notebook "
        "`text-classification.ipynb`. Das dort geladene und vorbereitete Modell wird in "
        "der Streamlit-App lediglich angewendet, um neue Lyrics zu klassifizieren – "
        "ohne erneutes Training."
    )

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
