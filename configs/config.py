from pathlib import Path 
import yaml


ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
DATA_PROC = ROOT / "data" / "processed"