"""
Transformation des données brutes en features de "Forme" (Rolling Averages).
Anti-data-leakage strict : le modèle qui prédit le Match N n'a accès qu'aux stats des matchs N-5 à N-1.
"""

import logging
from typing import Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# FOOTBALL (Premier League et autres ligues)
# ============================================================================

def compute_rolling_averages_football(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Calcule les moyennes mobiles pour le football.
    
    Colonnes en entrée (football-data.co.uk) :
    - Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HST, AST, HC, AC, HF, AF, FTR
    
    Colonnes en sortie (features de "Forme") :
    - home_shots_avg, away_shots_avg
    - home_shots_on_target_avg, away_shots_on_target_avg
    - home_goals_avg, away_goals_avg
    - home_corners_avg, away_corners_avg
    - home_fouls_avg, away_fouls_avg
    
    Anti-data-leakage : utilise .shift(1) après rolling() pour décaler d'un match.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Initialiser les colonnes de features
    features_to_compute = {
        "home": {
            "HS": "home_shots_avg",
            "HST": "home_shots_on_target_avg",
            "FTHG": "home_goals_avg",
            "HC": "home_corners_avg",
            "HF": "home_fouls_avg",
        },
        "away": {
            "AS": "away_shots_avg",
            "AST": "away_shots_on_target_avg",
            "FTAG": "away_goals_avg",
            "AC": "away_corners_avg",
            "AF": "away_fouls_avg",
        },
    }
    
    df_transformed = df.copy()
    
    # ===== HOME TEAM ROLLING AVERAGES =====
    for raw_col, feature_col in features_to_compute["home"].items():
        # Calculer la moyenne mobile par équipe à domicile (HomeTeam)
        rolling_col = (
            df_transformed.groupby("HomeTeam")[raw_col]
            .rolling(window=horizon, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # ANTI-DATA-LEAKAGE : shift(1) pour que Match N n'ait accès qu'aux moyennes de N-5 à N-1
        df_transformed[feature_col] = rolling_col.shift(1)
    
    # ===== AWAY TEAM ROLLING AVERAGES =====
    for raw_col, feature_col in features_to_compute["away"].items():
        # Calculer la moyenne mobile par équipe à l'extérieur (AwayTeam)
        rolling_col = (
            df_transformed.groupby("AwayTeam")[raw_col]
            .rolling(window=horizon, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # ANTI-DATA-LEAKAGE : shift(1)
        df_transformed[feature_col] = rolling_col.shift(1)
    
    # Supprimer les NaN (premiers matchs de chaque équipe)
    df_transformed = df_transformed.dropna(subset=list(features_to_compute["home"].values()) + 
                                                    list(features_to_compute["away"].values()))
    
    logger.info("Rolling averages (football) calculés pour %d matchs (horizon=%d)", 
                len(df_transformed), horizon)
    return df_transformed


# ============================================================================
# BASKETBALL (NBA)
# ============================================================================

def compute_rolling_averages_basketball(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Calcule les moyennes mobiles pour le basket (NBA).
    
    Colonnes attendues en entrée (format NBA stats) :
    - Date, HomeTeam, AwayTeam, HomePTS, AwayPTS, Home_FG_Pct, Away_FG_Pct,
      HomeRebounds, AwayRebounds, HomeTurnovers, AwayTurnovers
    
    Colonnes en sortie (features de "Forme") :
    - home_pts_avg, away_pts_avg
    - home_fg_pct_avg, away_fg_pct_avg
    - home_rebounds_avg, away_rebounds_avg
    - home_turnovers_avg, away_turnovers_avg
    
    Anti-data-leakage : shift(1) après rolling().
    """
    df = df.sort_values("Date").reset_index(drop=True)
    
    features_to_compute = {
        "home": {
            "HomePTS": "home_pts_avg",
            "Home_FG_Pct": "home_fg_pct_avg",
            "HomeRebounds": "home_rebounds_avg",
            "HomeTurnovers": "home_turnovers_avg",
        },
        "away": {
            "AwayPTS": "away_pts_avg",
            "Away_FG_Pct": "away_fg_pct_avg",
            "AwayRebounds": "away_rebounds_avg",
            "AwayTurnovers": "away_turnovers_avg",
        },
    }
    
    df_transformed = df.copy()
    
    # ===== HOME TEAM ROLLING AVERAGES =====
    for raw_col, feature_col in features_to_compute["home"].items():
        rolling_col = (
            df_transformed.groupby("HomeTeam")[raw_col]
            .rolling(window=horizon, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df_transformed[feature_col] = rolling_col.shift(1)
    
    # ===== AWAY TEAM ROLLING AVERAGES =====
    for raw_col, feature_col in features_to_compute["away"].items():
        rolling_col = (
            df_transformed.groupby("AwayTeam")[raw_col]
            .rolling(window=horizon, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df_transformed[feature_col] = rolling_col.shift(1)
    
    # Supprimer les NaN
    df_transformed = df_transformed.dropna(subset=list(features_to_compute["home"].values()) + 
                                                    list(features_to_compute["away"].values()))
    
    logger.info("Rolling averages (basket) calculés pour %d matchs (horizon=%d)", 
                len(df_transformed), horizon)
    return df_transformed


# ============================================================================
# FONCTION GÉNÉRIQUE DE PRÉPARATION POUR L'ENTRAÎNEMENT
# ============================================================================

def prepare_training_data_football(
    df: pd.DataFrame, 
    horizon: int = 5
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Pipeline complète pour le football :
    1. Calculer les rolling averages
    2. Encoder la cible (FTR)
    3. Retourner le DataFrame prêt à l'entraînement + liste des features
    
    :return: (df_prêt_pour_xgboost, liste_des_feature_columns)
    """
    # Étape 1 : Rolling averages
    df_transformed = compute_rolling_averages_football(df, horizon=horizon)
    
    # Étape 2 : Encoder FTR (H=2, D=1, A=0)
    df_transformed["FTR_encoded"] = df_transformed["FTR"].map({"H": 2, "D": 1, "A": 0})
    df_transformed = df_transformed.dropna(subset=["FTR_encoded"])
    
    # Étape 3 : Lister les features (UNIQUEMENT les rolling averages, pas les données brutes)
    feature_columns = [
        "home_shots_avg", "away_shots_avg",
        "home_shots_on_target_avg", "away_shots_on_target_avg",
        "home_goals_avg", "away_goals_avg",
        "home_corners_avg", "away_corners_avg",
        "home_fouls_avg", "away_fouls_avg",
    ]
    
    logger.info("Dataset prêt pour entraînement: %d matchs, %d features", 
                len(df_transformed), len(feature_columns))
    return df_transformed, feature_columns


def prepare_training_data_basketball(
    df: pd.DataFrame, 
    horizon: int = 5
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Pipeline complète pour le basket :
    1. Calculer les rolling averages
    2. Encoder la cible (Win/Loss : 1/0)
    3. Retourner le DataFrame prêt à l'entraînement + liste des features
    
    :return: (df_prêt_pour_xgboost, liste_des_feature_columns)
    """
    # Étape 1 : Rolling averages
    df_transformed = compute_rolling_averages_basketball(df, horizon=horizon)
    
    # Étape 2 : Encoder la cible (Win=1 si HomePTS > AwayPTS, Loss=0)
    df_transformed["HomeWin_encoded"] = (df_transformed["HomePTS"] > df_transformed["AwayPTS"]).astype(int)
    df_transformed = df_transformed.dropna(subset=["HomeWin_encoded"])
    
    # Étape 3 : Lister les features
    feature_columns = [
        "home_pts_avg", "away_pts_avg",
        "home_fg_pct_avg", "away_fg_pct_avg",
        "home_rebounds_avg", "away_rebounds_avg",
        "home_turnovers_avg", "away_turnovers_avg",
    ]
    
    logger.info("Dataset prêt pour entraînement (basket): %d matchs, %d features", 
                len(df_transformed), len(feature_columns))
    return df_transformed, feature_columns


if __name__ == "__main__":
    # Exemple d'utilisation (test avec football)
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs/sports-betting-ml")
    
    df_raw = pd.read_csv("data/raw/E0.csv")
    df_transformed, features = prepare_training_data_football(df_raw)
    print("\nColonnes du dataset final :")
    print(df_transformed[features].head())
    print(f"\nTarget distribution :\n{df_transformed['FTR_encoded'].value_counts()}")
