import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
PLANET_API_KEY = os.environ["PLANET_API_KEY"]
print("API KEY Successfully loaded")

BASEMAPS_API_URL = "https://api.planet.com/basemaps/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_QUADS = DATA_DIR / "quads"
DATA_MODELS = DATA_DIR / "models"

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_RAW:", DATA_RAW)
    print("DATA_PROCESSED:", DATA_PROCESSED)
    print("DATA_QUADS:", DATA_QUADS)
    print("DATA_MODELS:", DATA_MODELS)
