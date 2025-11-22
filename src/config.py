import os
from pathlib import Path

PLANET_API_KEY = os.getenv("PL_API_KEY")  # desde tu .env


# Ruta a la carpeta raíz del proyecto (sube un nivel desde src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_QUADS = DATA_DIR / "quads"
DATA_MODELS = DATA_DIR / "models"

# Si quieres verlas cuando importas:
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_RAW:", DATA_RAW)
    print("DATA_PROCESSED:", DATA_PROCESSED)
    print("DATA_QUADS:", DATA_QUADS)
    print("DATA_MODELS:", DATA_MODELS)
