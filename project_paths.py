"""
project_paths.py
----------------
Central project-root paths shared across the app, storage, and pipeline.

Using absolute paths here prevents the app from creating separate SQLite,
profile, and Qdrant state depending on which directory the command was run
from (for example project root vs. app/).
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROFILES_DIR = PROJECT_ROOT / "profiles"
MATERIALS_DB_PATH = DATA_DIR / "materials.db"
QDRANT_STORAGE_DIR = PROJECT_ROOT / "qdrant_storage"
