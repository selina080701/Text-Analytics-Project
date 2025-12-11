import os
import json
import numpy as np
import pandas as pd
import streamlit as st

def show_statistical_analysis_page():
    st.title("4️⃣ Kapitel 4 – Statistical Analysis: Genius Song Lyrics Subset (1%)")

    st.markdown("""
            **Dataset:** 34'049 Songs | 26'408 Artists | 6 Genres  
            **Genres:** Rap / Hip-Hop · Rock · Pop · R&B · Country · Miscellaneous

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

    st.info(
        "**Hinweis:** Dieser Abschnitt dokumentiert die Schritte aus dem zugehörigen Notebook "
        "`statistical-analysis.ipynb`. Alle statistischen Auswertungen und Visualisierungen "
        "wurden vollständig im Notebook berechnet und als PNG/JSON gespeichert. "
        "Die Streamlit-App lädt diese Inhalte lediglich und zeigt sie an – ohne die Analysen "
        "erneut auszuführen."
    )

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

