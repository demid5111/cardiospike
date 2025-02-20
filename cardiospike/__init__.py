import multiprocessing
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

NUM_CORES = multiprocessing.cpu_count()

ROOT_DIR = Path(__file__).parents[0].parents[0]

PACKAGE_DIR = ROOT_DIR / "cardiospike"


DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
SMART_MODEL_PATH = DATA_DIR / "web_model.joblib"
TEST_PATH = DATA_DIR / "test.csv"
WELLTORY_PATH = DATA_DIR / "welltory.csv"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

CHECKPOINTS_DIR: str = os.environ.get("CHECKPOINTS_DIR"), str(DATA_DIR / "checkpoints")  # type: ignore
LOGS_DIR: str = os.environ.get("LOGS_DIR"), str(DATA_DIR / "logs")  # type: ignore


DB_COLORS = {
    "violet": "#804bf2",
    "blue": "#1940ff",
    "dirty-blue": "#2d8ca7",
    "turquoise": "#68e4b2",
    "green": "#9fe54a",
    "yellow": "#ffbe00",
    "red": "#ff1c60",
}

API_E2E_ARTIFACTS_DIR = ROOT_DIR / 'tests' / 'artifacts'
EVALUATION_REPORTS_DIR = DATA_DIR / 'evaluation_reports'
EVALUATION_REPORTS_PATH = EVALUATION_REPORTS_DIR / 'full_report.csv'
