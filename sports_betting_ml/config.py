import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = str(PROJECT_ROOT / "models")
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "alerts").mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DATA_RAW_PATH = str(DATA_DIR / "raw" / "nba_games.csv")
DATA_PROCESSED_PATH = str(DATA_DIR / "processed" / "nba_with_features.pkl")
ALERTS_OUTPUT_PATH = str(DATA_DIR / "alerts" / "value_bets.json")
SENT_BETS_FILE = str(DATA_DIR / "alerts" / "sent_bets.txt")

ODDS_API_KEY = "5b7d9e3ee259ce582e0165a2aca78ccb"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORTS = ["basketball_nba"]
REGIONS = ["us"]
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"

TELEGRAM_BOT_TOKEN = "8786034020:AAFi0Qolviy2-mmC8StpFUe5FSE7mZ6Vv2U"
TELEGRAM_CHAT_ID = "6423161858"

# ============================================
# VALUE BET DETECTION - SEUIL EV (MODE SIGNAL PUR)
# ============================================
EV_THRESHOLD = 1.0  # EV minimum en % pour déclencher un signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOGS_DIR / 'system.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)