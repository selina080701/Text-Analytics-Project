# utils/page_config.py

from app_pages.page_1_subset import show_subset_page
from app_pages.page_2_cleaning import show_cleaning_page
from app_pages.page_3_tokenization import show_tokenization_page
from app_pages.page_4_statistical_analysis import show_statistical_analysis_page
from app_pages.page_5_word_embedding import show_word_embedding_page
from app_pages.page_6_model_evaluation import show_model_evaluation_page
from app_pages.page_7_text_classification import show_text_classification_page
from app_pages.page_8_text_generation import show_text_generation_page

PAGE_CONFIG = {
    "1️⃣ Datensubset laden": show_subset_page,
    "2️⃣ Daten bereinigen": show_cleaning_page,
    "3️⃣ Tokenisierung": show_tokenization_page,
    "4️⃣ Statistische Analyse": show_statistical_analysis_page,
    "5️⃣ Word Embedding": show_word_embedding_page,
    "6️⃣ Model Evaluation": show_model_evaluation_page,
    "7️⃣ Text Classification": show_text_classification_page,
    "8️⃣ Text-Generierung": show_text_generation_page,
}
