# Anti-Data-Leakage : Architecture Complète

## Problème Résolu

**Data Leakage (Triche)** : Si le modèle entraîne directement sur les statistiques brutes d'un match (tirs, buts, fautes du match en question), il connaît déjà le contexte que le match qu'il est censé *prédire*.

### Scénario problématique
```
Modèle apprend sur :
  Match N : HS=15, HST=6, FTHG=3 → Résultat=Victoire Domicile

Puis on lui demande : "Prédit Match N+1 : HS=?, HST=?, FTHG=?"
→ Le modèle ne peut PAS utiliser les stats brutes car il ne les a pas en avance.
```

---

## Solution : Rolling Averages (Moyennes Mobiles)

### Concept
Remplacer les données brutes d'un match par la **forme actuelle** de l'équipe (moyenne de ses 5 derniers matchs).

```
Match N-4 : Arsenal 3-1 Spurs | HS=18, HST=7
Match N-3 : Arsenal 2-0 Brighton | HS=16, HST=6
Match N-2 : Arsenal 1-1 Chelsea | HS=14, HST=4
Match N-1 : Arsenal 4-2 West Ham | HS=20, HST=9
Match N   : Arsenal vs Newcastle   | HS=?, HST=?
          ↓ (ce qu'on prédit)      ↓ (on utilise ça)
          
Moyenne Mobile Arsenal (5 derniers) = HS_avg = (18+16+14+20)/4 = 17
                                       HST_avg = (7+6+4+9)/4 = 6.5
```

### Protection Anti-Leakage : shift(1)

```python
# Calculer la moyenne mobile
rolling_col = df.groupby("HomeTeam")["HS"].rolling(window=5).mean()

# CRITIQU E: shift(1) décale d'un rang
df["HS_avg"] = rolling_col.shift(1)
```

**Résultat** :
```
Match N-4 : HS_avg = NaN   (pas de 5 matchs antérieurs)
Match N-3 : HS_avg = 18.0  (moyenne de Match N-4 seul)
Match N-2 : HS_avg = 17.0  (moyenne Matches N-4, N-3)
Match N-1 : HS_avg = 16.3  (moyenne Matches N-4 à N-2)
Match N   : HS_avg = 17.0  (moyenne Matches N-5 à N-1)   ← **CE QU'ON UTILISE**
```

Le modèle qui prédit Match N n'a accès qu'aux stats de N-5 à N-1. **Pas de triche.**

---

## Architecture Complète

### 1. **ENTRAÎNEMENT** (Offline - une seule fois)

```
src/model_train.py
  ↓
  1. Charger E0.csv (10,000+ matchs historiques)
  2. Appeler prepare_training_data_football(df)
     ↓
     src/data_transform.py::compute_rolling_averages_football()
       • Trier par Date
       • Grouper par HomeTeam / AwayTeam
       • Calculer rolling(window=5).mean() + shift(1)
       • Résultat : 10 colonnes de "Forme"
  3. Entraîner XGBoost UNIQUEMENT sur ces colonnes de Forme
  4. Sauvegarder :
     - models/xgb_model_football.joblib
     - models/features_football.joblib (liste des 10 features)
```

**Colonnes utilisées pour l'entraînement** :
```
HOME :                          AWAY :
- home_shots_avg                - away_shots_avg
- home_shots_on_target_avg      - away_shots_on_target_avg
- home_goals_avg                - away_goals_avg
- home_corners_avg              - away_corners_avg
- home_fouls_avg                - away_fouls_avg
```

**Colonnes JAMAIS utilisées** : HS, HST, FTHG, HC, HF brutes (données du match lui-même).

---

### 2. **INFÉRENCE** (Online - à chaque scan)

```
main.py → run_pipeline()
  ↓
  ÉTAPE 1 : OddsFetcher.get_odds_for_all_sports()
    ↓ Récupère : Arsenal vs Newcastle (match futur)
    
  ÉTAPE 2 : FootballPredictor.predict_match("Arsenal", "Newcastle")
    ↓
    src/prediction_inference.py::FootballPredictor
      • Charger le modèle entraîné
      • Charger l'historique E0.csv
      • _compute_current_rolling_avg("Arsenal", "home")
        ↓ Récupère les 5 DERNIERS matchs d'Arsenal à domicile dans l'historique
        ↓ Calcule : HS_avg, HST_avg, FTHG_avg, HC_avg, HF_avg
      • _compute_current_rolling_avg("Newcastle", "away")
        ↓ Même chose pour Newcastle à l'extérieur
      • Construire le vecteur [17.0, 6.5, 2.3, 5.2, 11.1, ...]
      • Appeler model.predict_proba(X)
      • Retourner : {prob_home: 0.65, prob_draw: 0.20, prob_away: 0.15}
    
  ÉTAPE 3 : Croiser avec cotes The Odds API
    ↓ Arsenal@1.85 (The Odds API) vs Probabilité=0.65 (modèle)
    ↓ Cote Théorique = 1/0.65 = 1.54
    ↓ 1.85 > 1.54 ? OUI → Value Bet (+EV)
    
  ÉTAPE 4 : Calculer Kelly et notifier
    ↓ EV = (0.65 * 1.85) - 1 = 0.20 (+20%)
    ↓ Mise Kelly = ((0.65 * 1.85 - 1) / (1.85 - 1)) * 0.25 = 2.1% du bankroll
```

---

## Flux de Données Clés

### Différence ENTRAÎNEMENT vs INFÉRENCE

| Phase | Source | Features | Cible | Résultat |
|-------|--------|----------|-------|----------|
| **ENTRAÎNEMENT** | E0.csv (passé) | rolling_avg (5 derniers matchs) | FTR (résultat réel) | Modèle .joblib |
| **INFÉRENCE** | The Odds API (futur) | rolling_avg (5 derniers matchs historiques) | ??? (c'est ce qu'on prédit) | Proba {H, D, A} |

### Anti-leakage en image
```
ENTRAÎNEMENT :
─────────────────────────────────────────────
Match 1 (2010) : HS_raw=12, rolling_avg=NaN
Match 2 (2010) : HS_raw=15, rolling_avg=12 ← feature
Match 3 (2010) : HS_raw=18, rolling_avg=13.5 ← feature
...
Match N (2026) : HS_raw=20, rolling_avg=17 ← feature + FTR=H (cible connue)
─────────────────────────────────────────────
Le modèle apprend : "Si rolling_avg ≈ 17, alors FTR tend vers H"

INFÉRENCE :
─────────────────────────────────────────────
Match Futur : Arsenal vs Newcastle (lundi)
  → Historique : rolling_avg(Arsenal_home) = 17.2
  → Modèle : "Je prédis Victoire Arsenal (0.65)"
─────────────────────────────────────────────
Pas de contamination : on n'a PAS utilisé les stats du match futur.
```

---

## Fichiers Clés

| Fichier | Rôle | Input | Output |
|---------|------|-------|--------|
| `data_transform.py` | Calcul rolling averages | CSV brut | DF avec colonnes Forme + anti-leakage |
| `model_train.py` | Entraîner XGBoost | DF transformé + FTR | xgb_model_football.joblib |
| `prediction_inference.py` | Prédire matchs futurs | Historique + noms équipes | {prob_home, prob_draw, prob_away} |
| `value_bet_detector.py` | Comparer IA vs bookmaker | Probabilités IA + cotes API | Alerte value bet |
| `main.py` | Orchestrer tout | API The Odds API | JSON alerts + webhook n8n |

---

## Commandes

### Entraîner le modèle (une fois)
```bash
python -m src.model_train
```
→ Génère `models/xgb_model_football.joblib` + `models/features_football.joblib`

### Prédire un match (test)
```bash
python -m src.prediction_inference
```
→ Teste la prédiction sur Arsenal vs City, Liverpool vs Chelsea, etc.

### Lancer le pipeline complet (en production)
```bash
python main.py
```
→ Fetch → Prédiction → Value Bets → Notification

---

## Garanties Anti-Leakage

✅ **Le modèle n'utilise JAMAIS** :
- Les tirs du match à prédire (HS, AS bruts)
- Les buts du match à prédire (FTHG, FTAG bruts)
- Les corners du match à prédire (HC, AC bruts)

✅ **Le modèle utilise UNIQUEMENT** :
- La moyenne de ces stats sur les 5 matchs ANTÉRIEURS
- Décalée d'un rang (shift(1)) pour anti-leakage strict

✅ **Résultat** :
- Prédictions saines et non biaisées
- Applicable en temps réel sans contamination
- Robustesse testée sur train/test splits réalistes

---

## Troubleshooting

**Q: Le modèle a 52% d'accuracy. C'est bon ?**
A: Pour le sport, oui. L'accuracy brute n'est pas le KPI. Le +EV sur The Odds API l'est.

**Q: Pourquoi shift(1) et pas shift(0) ?**
A: shift(0) ne décale pas → data leakage. shift(1) décale d'un match → anti-leakage.

**Q: Les rolling averages peuvent être NaN ?**
A: Oui, pour les 5 premiers matchs de chaque équipe. On les supprime (.dropna()).

**Q: Comment adapter pour d'autres sports ?**
A: Copier `compute_rolling_averages_football()` et remplacer les colonnes
   (HS → points marqués, HST → % FG, etc.).
