import json
import logging
from pathlib import Path
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_for_json(obj):
    """Convertit récursivement TOUS les types numpy"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def export_alerts_to_json(alerts: list[dict], path: str = config.ALERTS_OUTPUT_PATH) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean_alerts = [clean_for_json(alert) for alert in alerts]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_alerts, f, ensure_ascii=False, indent=2)
    logger.info("%d alerte(s) exportée(s) vers %s", len(alerts), path)
    return path


def notify(alerts: list[dict]) -> None:
    if not alerts:
        logger.info("Aucun value bet détecté sur ce scan.")
        return
    export_alerts_to_json(alerts)
    logger.info("✅ Alertes sauvegardées dans data/alerts/value_bets.json")