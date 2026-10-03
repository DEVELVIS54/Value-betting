# sports_betting_ml/run_all.py
"""
Lanceur unique : poller Telegram (arrière-plan) + scans périodiques + règlement quotidien.

Usage : python run_all.py                      (scan toutes les 6 h, règlement toutes les 24 h)
        python run_all.py --interval 300       (scan toutes les 5 h)
Ctrl+C arrête tout proprement.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def log(msg: str) -> None:
    print(f"[run_all {datetime.now():%d/%m %H:%M}] {msg}", flush=True)


def run(args: list) -> int:
    return subprocess.run([PY, *args], cwd=ROOT).returncode


def start_poller() -> subprocess.Popen:
    log("Démarrage du poller Telegram")
    return subprocess.Popen([PY, "src/telegram_alerts.py"], cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=360, help="minutes entre deux scans (3 crédits chacun)")
    ap.add_argument("--delay", type=int, default=0, help="minutes avant le premier scan")
    ap.add_argument("--settle-hours", type=float, default=24, help="heures entre deux règlements (2 crédits)")
    args = ap.parse_args()

    poller = start_poller()
    next_scan = time.time() + args.delay * 60
    next_settle = time.time() + args.settle_hours * 3600
    log(f"Scan toutes les {args.interval} min | règlement toutes les {args.settle_hours:g} h | Ctrl+C pour arrêter")

    try:
        while True:
            if poller.poll() is not None:
                log("Poller arrêté, redémarrage")
                poller = start_poller()

            now = time.time()
            if now >= next_scan:
                log("Mise a jour de l'historique")
                code_hist = run(["update_history.py"])
                if code_hist != 0:
                    log(f"Historique en échec (code {code_hist}) : nouvel essai dans 60 s")
                    time.sleep(60)
                    code_hist = run(["update_history.py"])
                if code_hist != 0:
                    log(f"Historique toujours en échec (code {code_hist}) : scan ignoré (aucun crédit dépensé), "
                        "voir logs\\update_history.log")
                else:
                    log("Scan des cotes")
                    code = run(["main.py", "scan"])
                    if code == 3:
                        log("Quota insuffisant : scan ignoré, nouvel essai au prochain cycle")
                    elif code != 0:
                        log(f"Scan en erreur (code {code})")
                next_scan = time.time() + args.interval * 60

            if now >= next_settle:
                log("Règlement des résultats")
                run(["main.py", "settle"])
                next_settle = time.time() + args.settle_hours * 3600

            time.sleep(20)
    except KeyboardInterrupt:
        log("Arrêt demandé")
    finally:
        poller.terminate()
        try:
            poller.wait(timeout=10)
        except subprocess.TimeoutExpired:
            poller.kill()
        log("Terminé")


if __name__ == "__main__":
    main()