# Architecture des Rolling Averages & Anti Data-Leakage

## Problème corrigé : Data Leakage

### ❌ AVANT (Incorrect)
```
Entraînement du modèle :
Match Arsenal vs Chelsea (22/08/2026)
  - HS (tirs) = 18
  - HST (tirs cadrés) = 7
  - Résultat = "H" (victoire Arsenal)
  
⚠️ PROBLÈME : le modèle apprend que "HS=18 → victoire"
Mais ces statistiques sont du MÊME MATCH qu'il prédisait !
C'est comme apprendre à prédire l'examen en lisant la RÉPONSE.
```

### ✅ APRÈS (Correct)
```
Entraînement du modèle :
Match Arsenal vs Chelsea (22/08/2026)
  - Arsenal_HS_rolling_5 = moyenne des tirs des 5 matchs antérieurs
                          = (12 + 15 + 14 + 16 + 13) / 5 = 14.0
  - Arsenal_HST_rolling_5 = moyenne des tirs cadrés des 5 matchs antérieurs
                           = (4 + 5 + 6 + 5 + 4) / 5 = 4.8
  - Résultat = "H"
  
✓ CORRECT : le modèle apprend que "forme_récente_Arsenal=14 → victoire probable"
Les statistiques utilisées sont ANTÉRIEURES au match prédit.
```

---

## Pipeline Technique

### 1. **load_and_prepare_dataset_football()** (`model_train.py`)

Charger le CSV brut et calculer les rolling averages avec `.shift(1)` :

```python
# Exemple : calcul pour Arsenal (domicile)
df.sort_values("Date")

# Moyenne mobile sur 5 matchs
df["home_HS_rolling_5"] = df.groupby("HomeTeam")["HS"].rolling(5, min_periods=1).mean()

# ANTI-LEAKAGE : décaler d'1 ligne
# Le match N voit uniquement les moyennes des matchs N-1 à N-5
df["home_HS_rolling_5"] = df["home_HS_rolling_5"].shift(1)

# Supprimer les NaN créés par shift
df = df.dropna()
```

**Résultat après shift :**
```
Date          Match                  HS  home_HS_rolling_5
2026-08-01    Arsenal vs Liverpool   12  NaN              (aucun historique)
2026-08-08    Arsenal vs Chelsea     15  12.0             (moyenne de 1 seul match : ok)
2026-08-15    Arsenal vs Man City    18  13.5             (moyenne des 2 précédents)
2026-08-22    Arsenal vs Tottenham   16  14.0             (moyenne des 3 précédents)
2026-08-29    Arsenal vs Brighton    14  15.25            (moyenne des 4 précédents)
2026-09-05    Arsenal vs Newcastle   13  14.6             (moyenne des 5 précédents ← fenêtre complète)
```

**Important** : match du 2026-08-08 voit `HS_rolling_5 = 12.0` (1 seul match, pas 5).
Mais ce 12 vient de **avant** le 2026-08-08, donc pas de leakage.

---

### 2. **Features finales pour XGBoost**

Colonnes utilisées pour entraîner le modèle :

**FOOTBALL** (8 features par match) :
```
home_HS_rolling_5      (tirs domicile, moyenne 5 matchs)
home_HST_rolling_5     (tirs cadrés domicile)
home_HC_rolling_5      (corners domicile)
home_FTHG_rolling_5    (buts marqués domicile)

away_AS_rolling_5      (tirs extérieur)
away_AST_rolling_5     (tirs cadrés extérieur)
away_AC_rolling_5      (corners extérieur)
away_FTAG_rolling_5    (buts marqués extérieur)
```

**BASKETBALL** (8 features par match) :
```
home_HomePoints_rolling_5
home_HomeFG_Pct_rolling_5
home_HomeRebounds_rolling_5
home_HomeTurnovers_rolling_5

away_AwayPoints_rolling_5
away_AwayFG_Pct_rolling_5
away_AwayRebounds_rolling_5
away_AwayTurnovers_rolling_5
```

---

### 3. **Extract Features pour Prédiction Live** (`predictor.py`)

Quand l'API The Odds API signale : **"Arsenal vs Chelsea demain 15:00"**

```python
# 1. Charger l'historique pré-calculé
df_history = pd.read_pickle("data/raw/E0_with_rolling.pkl")

# 2. Récupérer la DERNIÈRE ligne d'Arsenal (domicile)
arsenal_last = df_history[df_history["HomeTeam"] == "Arsenal"].tail(1)
#
# Date          HomeTeam  home_HS_rolling_5  home_HST_rolling_5  ...
# 2026-09-21    Arsenal   13.8               5.2                ...

# 3. Récupérer la DERNIÈRE ligne de Chelsea (extérieur)
chelsea_last = df_history[df_history["AwayTeam"] == "Chelsea"].tail(1)
#
# Date          AwayTeam  away_AS_rolling_5  away_AST_rolling_5  ...
# 2026-09-20    Chelsea   10.4               4.1                ...

# 4. Créer le vecteur de features pour XGBoost
features = {
    "home_HS_rolling_5": 13.8,    # forme d'Arsenal
    "home_HST_rolling_5": 5.2,
    ...
    "away_AS_rolling_5": 10.4,    # forme de Chelsea
    "away_AST_rolling_5": 4.1,
    ...
}

# 5. Prédire
probas = model.predict_proba([features])
# → [0.25, 0.35, 0.40]  (classeur 0=victoire Chelsea, 1=nul, 2=victoire Arsenal)
```

**Aucun leakage** : les données utilisées sont antérieures au match à prédire.

---

## Flux Complet : De l'Entraînement à la Prédiction

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1 : OFFLINE TRAINING (une seule fois)                    │
├─────────────────────────────────────────────────────────────────┤
│ $ python -m src.model_train                                     │
│                                                                  │
│ 1. Load E0.csv (10,000 matchs)                                  │
│    Colonnes brutes : HS, AS, HST, AST, HC, AC, FTHG, FTAG, FTR │
│                                                                  │
│ 2. Compute rolling_5 avec .shift(1) (ANTI-LEAKAGE)              │
│    Nouvelles colonnes : home_HS_rolling_5, away_AST_rolling_5..│
│                                                                  │
│ 3. Train XGBoost sur rolling averages uniquement                │
│    Features : 8 colonnes (rolling_5)                            │
│    Cible : FTR_encoded (H=2, D=1, A=0)                          │
│                                                                  │
│ 4. Save model → models/xgb_football.joblib (100 KB)             │
│ 5. Save history → data/raw/E0_with_rolling.pkl (1 MB)           │
│                                                                  │
│ Output: Accuracy ~52% | Log Loss ~0.95                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2 : ONLINE PREDICTION (toutes les heures/jours)          │
├─────────────────────────────────────────────────────────────────┤
│ $ python main.py                                                │
│                                                                  │
│ 1. Fetch The Odds API → cotes pour matchs à venir               │
│    Exemple : "Arsenal vs Chelsea | cote 1.85"                   │
│                                                                  │
│ 2. Load model + history                                         │
│    model = joblib.load("models/xgb_football.joblib")            │
│    df_history = pd.read_pickle("data/raw/E0_with_rolling.pkl")  │
│                                                                  │
│ 3. Extract features from df_history                             │
│    arsenal_rolling = df_history[team=="Arsenal"].tail(1)        │
│    chelsea_rolling = df_history[team=="Chelsea"].tail(1)        │
│                                                                  │
│ 4. Predict probabilities                                        │
│    probas = model.predict_proba([features])                     │
│    → home_prob=0.55, draw_prob=0.25, away_prob=0.20            │
│                                                                  │
│ 5. Detect value bets (+EV)                                      │
│    theo_odds = 1 / 0.55 = 1.82                                  │
│    bookmaker_odds = 1.85                                        │
│    If 1.85 > 1.82 → VALUE BET DETECTED                          │
│                                                                  │
│ 6. Calculate Kelly stake                                        │
│    stake = (0.55 * 1.85 - 1) / (1.85 - 1) × 0.25 = 3.2% capital│
│                                                                  │
│ 7. Export → data/alerts/value_bets.json + webhook n8n           │
│                                                                  │
│ Output: {"match": "Arsenal vs Chelsea", "ev": "+2.8%", ...}     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Vérification de l'Anti-Leakage

**Checklist** :

- ✅ `.shift(1)` appliqué après chaque `.rolling()` ?
- ✅ Les NaN initiaux supprimés avec `.dropna()` ?
- ✅ Entraînement sur `home_HS_rolling_5`, pas sur `HS` brut ?
- ✅ Prédiction : extraction de la DERNIÈRE ligne de l'historique pour chaque équipe ?
- ✅ Les prédictions n'utilisent que les données antérieures au match ?

**Si oui à tous** → Pas de data leakage. 🎯

---

## Exemple Numérique Complet

### Entraînement (Arsenal, matchs de juillet 2026) :

```
Date        Adversaire  HS  HST  Résultat  HS_rolling_5  HST_rolling_5  FTR_encoded
2026-07-01  Liverpool   12  4    D         NaN           NaN            1
2026-07-08  Chelsea     15  6    H         12.0          4.0            2     ← shift(1)
2026-07-15  Man City    18  7    A         13.5          5.0            0
2026-07-22  Brighton    14  5    H         15.0          6.0            2
2026-07-29  Everton     16  6    D         15.25         5.75           1     ← shift(1)
2026-08-05  Newcastle   13  4    H         14.6          5.4            2     ← shift(1)
```

**Entraînement XGBoost** :
- Feature 1: 12.0 (Arsenal_HS_rolling_5 en 2026-07-08) → Cible: D (nul)
- Feature 2: 13.5 (Arsenal_HS_rolling_5 en 2026-07-15) → Cible: A (perte)
- Feature 3: 15.0 (Arsenal_HS_rolling_5 en 2026-07-22) → Cible: H (victoire)

**Aucune leakage** : les features viennent de **avant** le match, pas de celui-ci.

### Prédiction (Arsenal vs Chelsea, 2026-09-22) :

```
Historique : dernière ligne d'Arsenal (en tant que domicile)
Date        Adversaire  HS  HST  HS_rolling_5  HST_rolling_5
2026-09-21  Brentford   13  5    13.8          5.2

Historique : dernière ligne de Chelsea (en tant qu'extérieur)
Date        Adversaire  AS  AST  AS_rolling_5  AST_rolling_5
2026-09-20  Fulham      10  4    10.4          4.1

Features pour XGBoost :
{
    "home_HS_rolling_5": 13.8,
    "home_HST_rolling_5": 5.2,
    "away_AS_rolling_5": 10.4,
    "away_AST_rolling_5": 4.1,
    ...
}

Prédiction : P(domicile) = 0.55, P(nul) = 0.25, P(extérieur) = 0.20

✓ Pas de leakage : 13.8 tirs en moyenne ne vient pas du match Arsenal-Chelsea 2026-09-22,
  mais de la forme récente d'Arsenal avant ce match.
```

---

## Conclusion

**Le `.shift(1)` est la clé.**

Sans lui : data leakage → modèle truqué sur test, échoue en production.
Avec lui : modèle robuste → fiable en temps réel sur des matchs futurs.
