# config.py
from pathlib import Path

# Basis-Verzeichnis (Projekt-Root)
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_CLEAN_DIR = BASE_DIR / "data" / "clean"
MODELS_DIR = BASE_DIR / "models"

# Ordner sicherstellen
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
