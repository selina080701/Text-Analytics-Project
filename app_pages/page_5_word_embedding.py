import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from gensim.models import Word2Vec
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

def show_word_embedding_page():
    import pandas as pd

    st.title("5️⃣ Kapitel 5 – Word Embedding: Genius Song Lyrics Subset (1%)")

    st.markdown("""
    **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  
    **Genres:** Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous
    
    **Purpose:**  
    Erstellung und Exploration von **Word Embeddings** für Songtexte.  

    Auf Basis der tokenisierten Lyrics werden:
    - ein **Word2Vec-Modell** trainiert,
    - semantische Beziehungen zwischen Wörtern untersucht,
    - Songtexte über TF-IDF + Word2Vec auf **Dokument-Embeddings** abgebildet
    - und diese im Embedding-Space analysiert.

    """)

    st.info(
        "**Hinweis:** Dieser Abschnitt dokumentiert die Schritte aus dem zugehörigen Notebook "
        "`word-embedding.ipynb`. Das Word2Vec-Modell und die Song-Embeddings wurden dort "
        "trainiert und als Dateien gespeichert. Die Streamlit-App lädt diese Inhalte lediglich "
        "und visualisiert die Ergebnisse – ohne das Modell erneut zu trainieren."
    )

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
        """
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


    df = pd.DataFrame({
        "Parameter": [
            "sentences", "vector_size", "window",
            "min_count", "workers", "epochs"
        ],
        "Bedeutung": [
            "Liste von Wortlisten (Token pro Song)",
            "Dimension der Vektoren",
            "Kontextfenstergrösse",
            "Minimalhäufigkeit für Wörter",
            "Anzahl Threads",
            "Trainingsdurchläufe"
        ]
    })

    st.dataframe(df)

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
    anschliessend Visualisierung im 3D-Raum und per PCA.
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
