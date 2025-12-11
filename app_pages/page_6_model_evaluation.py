import os
import json
import numpy as np
import pandas as pd
import streamlit as st

def show_model_evaluation_page():
    import os
    import json
    import numpy as np
    import pandas as pd

    st.title("6️⃣ Kapitel 6 - Model Evaluation: Genius Song Lyrics Subset (1%)")

    st.markdown(
        """
        **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  
        **Genres:** Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous
        
        **Purpose:**  
        Mehrere Modelle zur automatischen Genre-Klassifikation vergleichen – basierend auf
        unterschiedlichen Textrepräsentationen (Embeddings) und Klassifikatoren.  

        **Embeddings:**  
        - Word2Vec (self-trained)  
        - TF-IDF (character-level n-grams)  
        - SentenceTransformer (MiniLM)  

        **Classifier:**  
        - LinearSVC  
        - Logistic Regression  
        - Random Forest  

        Ausgewertet werden:  
        - Accuracy & Balanced Accuracy  
        - F1-Macro  
        - Klassifikationsberichte (im Notebook)  
        - Normalisierte Confusion Matrices (als PNG gespeichert)  

        """
    )

    st.info(
        "**Hinweis:** Dieser Abschnitt dokumentiert die Schritte aus dem zugehörigen Notebook "
        "`model-evaluation.ipynb`. Das Training der Modelle sowie die Berechnung aller Metriken "
        "und Confusion Matrices wurden vollständig im Notebook durchgeführt und als Ergebnisse "
        "gespeichert. Die Streamlit-App lädt diese Ergebnisse "
        "ausschließlich und visualisiert sie – ohne die Modelle erneut zu trainieren."
    )

    # =========================
    # 1. Notebook-Dokumentation (kurz)
    # =========================
    st.header("1. Preparation")
    st.subheader("1.1 Load Dataset")

    st.markdown("""
    Laden des final bereinigten Datensatzes (`data/clean/data.csv`) und
    Konvertierung der Spalte `tokens` von einer String-Repräsentation in echte Python-Listen
    (mittels `ast.literal_eval`).

    Anschließend erfolgt das **Label-Encoding**: Die Genre-Bezeichnungen (Strings wie *"rap"*, *"rock"*, *"rb"*) werden 
    mit einem `LabelEncoder` in ganze Zahlen umgewandelt, da Klassifikationsmodelle numerische Labels benötigen.
    """)

    st.code(
        """
df = pd.read_csv("data/clean/data.csv")

df["tokens"] = df["tokens"].apply(ast.literal_eval)
texts = df["tokens"]
labels = df["tag"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
        """,
        language="python",
    )

    st.subheader("1.2 Train-Test-Split")
    st.markdown("""
    Der Datensatz wird in einen **Trainings-** und einen **Testsplit** aufgeteilt. Dabei werden 80 % der Daten zum 
    Trainieren der Modelle verwendet, die restlichen 20 % dienen zur unabhängigen Evaluation.

    Durch `stratify=y_encoded` wird sichergestellt, dass alle Genres im gleichen Verhältnis in beiden Splits vertreten 
    sind – wichtig bei **unausgeglichenen Klassen**.
    """)
    st.code(
        """
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)""",
        language="python",
    )

    # =========================
    # 2. Embeddings & Modelle
    # =========================

    st.header("2. Embeddings & Modelle")
    st.subheader("2.1 Word2Vec")

    # -------------------------
    # 2.1.1 Embedding erzeugen
    # -------------------------
    st.markdown("### 2.1.1 Embedding erzeugen")
    st.markdown("""
    Für die erste Embedding-Strategie wird ein **Word2Vec-Modell** auf den Token-Sequenzen
    des Trainingssplits trainiert. Word2Vec lernt für jedes Wort einen dichten Vektor,
    der semantische Ähnlichkeiten abbildet (z. B. ähnliche Wörter → ähnliche Vektoren).

    Um jedes Dokument (Songtext) als festen Embedding-Vektor darzustellen,
    werden die Wortvektoren gemittelt (**Mean Word Embedding**).  
    Dies erzeugt einen robusten, kompakten Repräsentationsvektor pro Song.
    """)

    st.code(
        """
w2v = Word2Vec(
    sentences=X_train_tokens,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10,
    seed=42,
)

def embed_sentence(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

X_train_w2v = np.vstack([embed_sentence(toks, w2v) for toks in X_train_tokens])
X_test_w2v  = np.vstack([embed_sentence(toks, w2v) for toks in X_test_tokens])""",
        language="python",
    )

    # -------------------------
    # 2.1.2 Klassifikation
    # -------------------------
    st.markdown("### 2.1.2 Klassifikation auf Word2Vec")
    st.markdown("""
    Auf Basis der erzeugten Word2Vec-Embeddings werden drei unterschiedliche
    Klassifikationsmodelle trainiert.  
    Alle Modelle erhalten `class_weight="balanced"`, um die ungleich verteilten Genres auszugleichen
    und Minderheitsklassen nicht zu benachteiligen.

    Im Folgenden werden die drei Modelle jeweils einzeln gezeigt.
    """)

    st.markdown("""
    **LinearSVC** ist ein lineares SVM-Modell und eignet sich gut für hochdimensionale Text-Embeddings.
    """)

    st.code(
        """
clf_w2v_svc = LinearSVC(class_weight="balanced", max_iter=10000)
clf_w2v_svc.fit(X_train_w2v, y_train)
y_pred_w2v_svc = clf_w2v_svc.predict(X_test_w2v)""",
        language="python",
    )

    st.markdown("""
    Die **Logistische Regression** ist ein einfaches, stabiles lineares Modell und funktioniert gut bei unausgeglichenen Klassen.
    """)

    st.code(
        """
clf_w2v_logreg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
)
clf_w2v_logreg.fit(X_train_w2v, y_train)
y_pred_w2v_logreg = clf_w2v_logreg.predict(X_test_w2v)""",
        language="python",
    )

    st.markdown("""
    Der **Random Forest** ist ein nichtlineares Ensemblemodell. Er kann komplexe Muster erfassen, skaliert aber weniger gut mit hochdimensionalen Text-Embeddings.
    """)

    st.code(
        """
clf_w2v_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
clf_w2v_rf.fit(X_train_w2v, y_train)
y_pred_w2v_rf = clf_w2v_rf.predict(X_test_w2v)""",
        language="python",
    )

    st.subheader("2.2 TF-IDF")

    # -------------------------
    # 2.2.1 Embedding erzeugen
    # -------------------------
    st.markdown("### 2.2.1 Embedding erzeugen")
    st.markdown("""
    Für die zweite Embedding-Strategie wird **TF-IDF** auf Zeichen-n-Grammen angewendet.  
    Die Token-Sequenzen werden dazu wieder zu Strings zusammengefügt, anschließend wird ein
    TF-IDF-Vektorraum auf **Character n-grams (3–5)** gelernt.

    Damit lassen sich charakteristische Schreibweisen, Silbenmuster und typische Endungen pro Genre erfassen.
    """)

    st.code(
        """
X_train_texts_char = X_train_texts.apply(lambda toks: " ".join(toks))
X_test_texts_char  = X_test_texts.apply(lambda toks: " ".join(toks))

tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    max_df=0.9,
)

X_train_tfidf = tfidf.fit_transform(X_train_texts_char)
X_test_tfidf  = tfidf.transform(X_test_texts_char)""",
        language="python",
    )

    # -------------------------
    # 2.2.2 Klassifikation auf TF-IDF
    # -------------------------
    st.markdown("### 2.2.2 Klassifikation auf TF-IDF")
    st.markdown("""
    Auf den TF-IDF-Features werden erneut drei Klassifikationsmodelle trainiert:
    **LinearSVC**, **Logistic Regression** und **Random Forest**, jeweils mit
    `class_weight="balanced"`.
    """)

    # LinearSVC
    st.markdown("""
    **LinearSVC** eignet sich auch hier gut für die hochdimensionalen TF-IDF-Vektoren.
    """)
    st.code(
        """
clf_tfidf_svc = LinearSVC(class_weight="balanced")
clf_tfidf_svc.fit(X_train_tfidf, y_train)
y_pred_tfidf_svc = clf_tfidf_svc.predict(X_test_tfidf)""",
        language="python",
    )

    # Logistische Regression
    st.markdown("""
    Die **Logistische Regression** dient als weiteres lineares Basismodell auf TF-IDF.
    """)
    st.code(
        """
clf_tfidf_logreg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
)
clf_tfidf_logreg.fit(X_train_tfidf, y_train)
y_pred_tfidf_logreg = clf_tfidf_logreg.predict(X_test_tfidf)""",
        language="python",
    )

    # Random Forest
    st.markdown("""
    Der **Random Forest** bildet die nichtlineare Vergleichsbasis auf TF-IDF-Features.
    """)
    st.code(
        """
clf_tfidf_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
clf_tfidf_rf.fit(X_train_tfidf, y_train)
y_pred_tfidf_rf = clf_tfidf_rf.predict(X_test_tfidf)""",
        language="python",
    )

    st.subheader("2.3 Transformer (SentenceTransformer MiniLM)")

    # -------------------------
    # 2.3.1 Embedding erzeugen
    # -------------------------
    st.markdown("### 2.3.1 Embedding erzeugen")
    st.markdown("""
    Als dritte Embedding-Strategie wird ein **SentenceTransformer** verwendet:
    `all-MiniLM-L6-v2` erzeugt semantische Satz- bzw. Dokument-Embeddings direkt aus den
    vollständigen Songtexten.

    Dazu werden die Token-Sequenzen wieder zu Strings zusammengefügt und mit dem
    SentenceTransformer zu dichten Vektoren enkodiert.
    """)

    st.code(
        """
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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

# Tensor → Python-Listen (für die Sklearn-Modelle)
X_train_emb_st = X_train_emb_st.tolist()
X_test_emb_st  = X_test_emb_st.tolist()""",
        language="python",
    )

    # -------------------------
    # 2.3.2 Klassifikation auf Transformer-Embeddings
    # -------------------------
    st.markdown("### 2.3.2 Klassifikation auf Transformer-Embeddings")
    st.markdown("""
    Auf den Transformer-Embeddings werden erneut drei Klassifikationsmodelle trainiert:
    **LinearSVC**, **Logistic Regression** und **Random Forest**.
    """)

    # LinearSVC
    st.markdown("""
    **LinearSVC** dient hier als robustes lineares Modell auf den semantischen Embeddings.
    """)
    st.code(
        """
clf_st_svc = LinearSVC(class_weight="balanced", max_iter=10000)
clf_st_svc.fit(X_train_emb_st, y_train)
y_pred_st_svc = clf_st_svc.predict(X_test_emb_st)""",
        language="python",
    )

    # Logistische Regression
    st.markdown("""
    Die **Logistische Regression** wird als zweites lineares Vergleichsmodell verwendet.
    """)
    st.code(
        """
clf_st_logreg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
)
clf_st_logreg.fit(X_train_emb_st, y_train)
y_pred_st_logreg = clf_st_logreg.predict(X_test_emb_st)""",
        language="python",
    )

    # Random Forest
    st.markdown("""
    Für den **Random Forest** werden die Embeddings in NumPy-Arrays konvertiert.
    """)
    st.code(
        """
X_train_st_rf = np.asarray(X_train_emb_st)
X_test_st_rf  = np.asarray(X_test_emb_st)

clf_st_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

clf_st_rf.fit(X_train_st_rf, y_train)
y_pred_st_rf = clf_st_rf.predict(X_test_st_rf)""",
        language="python",
    )

    # =========================
    # 3. Save
    # =========================

    st.subheader("3. Speichern des finalen Modells & der Evaluationsergebnisse")
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

    # =================================================== #
    # 📁 NOTEBOOK-RESULTATE – Word Embeddings
    # =================================================== #
    st.markdown("---")
    st.header("📁 Notebook-Resultate – Word Embeddings")

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
                # 2.1 Übersicht über alle Modelle
                # --------------------------------------------------
                st.subheader("Übersicht über alle Modelle")

                st.dataframe(
                    df_eval[
                        ["model", "embedding", "classifier", "accuracy", "balanced_accuracy", "f1_macro"]
                    ].reset_index(drop=True)
                    .style.format(
                        {
                            "accuracy": "{:.3f}",
                            "balanced_accuracy": "{:.3f}",
                            "f1_macro": "{:.3f}",
                        }
                    )
                )

                st.markdown("---")
                st.subheader("F1-Macro nach Modell")
                st.bar_chart(df_eval.set_index("model")["f1_macro"])

                # --------------------------------------------------
                # 2.3 Details zu den Modellen (Tabs)
                # --------------------------------------------------
                st.markdown("---")
                st.subheader("Details zu den Modellen (inkl. Confusion Matrices)")


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
                st.subheader("Finale Modellwahl & Modellselektion")

                st.markdown("""
    Über alle drei Embedding-Strategien – **Word2Vec**, **TF-IDF** und **Transformer (MiniLM)** – zeigt sich ein konsistentes Muster:

    - **LinearSVC** liefert die stabilste Gesamtperformance, unabhängig vom Embedding.  
    - **Logistische Regression** verbessert systematisch die Klassenbalance und den Recall für Minderheitsgenres.  
    - **Random Forest** erreicht oft hohe Accuracy, ist aber deutlich zugunsten der Mehrheitsklassen verzerrt und erzielt eine niedrige Balanced Accuracy.

    #### 🎯 Final gewähltes Modell

    **SentenceTransformer (MiniLM) + LinearSVC**

    Dieses Modell bietet:

    - solide Accuracy (~0.57)  
    - die beste Balanced Accuracy unter den leistungsstarken Modellen (~0.52)  
    - gute Performance sowohl für dominante als auch für Minderheitsgenres  
    - robuste Generalisierung dank semantisch reichhaltiger Transformer-Embeddings  

    In Kombination mit **LinearSVC**, das sehr stabil auf hochdimensionalen Embeddings arbeitet, ergibt sich ein Modell, das eine gute Balance zwischen Performance und Fairness über alle Genres hinweg bietet.
    """)
