# sports-betting-ml

Système d'investissement algorithmique +EV pour Football, NBA, NFL (pre-match uniquement).

## Arborescence

```
sports-betting-ml/
├── .env.example              # Variables d'environnement (clé API, webhook n8n)
├── requirements.txt
├── config.py                  # Configuration centrale (sports, marchés, Kelly, chemins)
├── main.py                    # Pipeline complet : fetch -> prédiction -> détection -> alerte
├── data/
│   ├── raw/                   # Snapshots JSON des cotes + CSV historiques
│   └── alerts/                # value_bets.json généré à chaque scan
├── models/                    # Modèles XGBoost entraînés (.joblib)
├── notebooks/                 # Exploration de données (facultatif)
└── src/
    ├── odds_fetcher.py        # Client The Odds API (marchés groupés h2h+spreads+totals)
    ├── model_train.py         # Entraînement XGBoost sur CSV historiques
    ├── kelly.py                # Fractional Kelly (dimensionnement des mises)
    ├── value_bet_detector.py  # Comparaison cote théorique IA vs cote bookmaker
    └── notifier.py             # Export JSON + webhook n8n (paris simples uniquement)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis renseigner ODDS_API_KEY
```

## Utilisation

1. Entraîner le modèle : `python -m src.model_train`
2. Lancer le pipeline complet : `python main.py`

## Règles strictes respectées

- Paris **simples uniquement**, aucun combiné (`config.ALLOW_COMBINED_BETS = False`)
- Mode **pre-match uniquement**, aucun rafraîchissement live
- **1 appel API = 1 sport**, marchés groupés (`h2h,spreads,totals`) pour préserver le quota de 500 req/mois
- Mise dimensionnée en **Kelly fractionné** (Kelly/4 par défaut, `config.KELLY_FRACTION`)
