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
