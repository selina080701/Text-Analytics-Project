# Project: Text Analytics on Song Lyrics

**Authors:** Serge Pellegatta, Selina Steiner

**GitHub Repository:** [https://github.com/selina080701/Text-Analytics-Project](https://github.com/selina080701/Text-Analytics-Project)

---

## Project Description

This project is part of the *Text Analytics* module. Students work in teams on a practical project where learned methods are applied to solve an innovative, data-driven question. The task involves:

- Identifying a concrete challenge suitable for text analytics  
- Preparing relevant data sources  
- Selecting and evaluating appropriate models  
- Implementing a functional Python prototype  

This project is based on the [Genius Song Lyrics Dataset](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics) from Hugging Face.

---

## Notebooks

### 1. `load-data-subset.ipynb`
- **Purpose:** Load a smaller subset of the full Genius Song Lyrics dataset from Hugging Face and save it locally.
- **Details:** Allows downloading a lightweight subset (e.g., 1%, 5%, 10%) to reduce memory and storage usage.
- **Output:** Saves the raw subset CSV files in `data/raw/`.

### 2. `data-cleaning.ipynb`
- **Purpose:** Clean and preprocess song lyrics for analysis.
- **Details:** Removes metadata tags (e.g., `[Intro]`, `[Verse]`), line breaks, and extra spaces. Renames the cleaned lyrics column to `lyrics`.
- **Output:** Saves the cleaned CSV files in `data/clean/`.

### 3. `statistical-analysis.ipynb`
- **Purpose:** Explore patterns and distributions in the cleaned song lyrics across genres and artists.
- **Details:** Focuses on word frequencies, stylistic differences, and similarity structures.
- **Input:** Uses cleaned CSVs from `data/clean/`.

---

## Folder Structure

- `data/raw/` : Raw subsets of the dataset (e.g., 1%, 5%)  
- `data/clean/` : Cleaned versions of the subsets  
- `load-data-subset.ipynb` : Notebook to load and save raw subsets  
- `data-cleaning.ipynb` : Notebook to clean the raw lyrics  
- `statistical-analysis.ipynb` : Notebook to perform analysis on cleaned data  
- `requirements.txt` : Python dependencies for the project  

---

## Setup

1. **Create a virtual environment**
```bash
python3 -m venv .venv
```

2. **Activate the virtual environment**
* macOS / Linux:
```bash
source .venv/bin/activate
```
* Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Git for Jupyter Notebooks (optional, avoids merge conflicts)**
```bash
# Install nbdime
pip install nbdime

# Enable Git integration globally
nbdime config-git --enable --global
```

5. **(Optional) Clear notebook outputs before committing**
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <notebook>.ipynb
```