"""
Pipeline d'INFÉRENCE : récupère les cotes en direct (The Odds API) et prédit les matchs à venir.
Utilise les rolling averages calculés à partir de l'historique récent.
"""

import logging
from typing import Optional, Dict, List

import joblib
import pandas as pd
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FootballPredictor:
    """Prédiction pour le football avec anti-data-leakage."""
    
    def __init__(self, 
                 model_path: str = f"{config.MODELS_DIR}/xgb_model_football.joblib",
                 features_path: str = f"{config.MODELS_DIR}/features_football.joblib",
                 historical_csv: str = f"{config.RAW_DATA_DIR}/E0.csv"):
        """
        :param model_path: chemin vers le modèle XGBoost entraîné
        :param features_path: chemin vers la liste des features
        :param historical_csv: chemin vers le CSV historique (pour calculer les rolling averages)
        """
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(features_path)
        
        # Charger et préparer l'historique
        self.df_historical = pd.read_csv(historical_csv)
        self.df_historical = self.df_historical.sort_values("Date").reset_index(drop=True)
        
        logger.info("Modèle football chargé: %d features", len(self.feature_columns))
        logger.info("Historique chargé: %d matchs", len(self.df_historical))
    
    def _compute_current_rolling_avg(self, team: str, home_or_away: str, horizon: int = 5) -> Dict[str, float]:
        """
        Calcule la moyenne mobile ACTUALISÉE pour une équipe donnée.
        
        :param team: nom de l'équipe (ex: "Arsenal")
        :param home_or_away: "home" ou "away"
        :param horizon: fenêtre mobile (5 derniers matchs)
        :return: dict avec les moyennes mobiles actualisées
        """
        if home_or_away == "home":
            team_matches = self.df_historical[self.df_historical["HomeTeam"] == team].copy()
            stats_cols = {
                "HS": "home_shots_avg",
                "HST": "home_shots_on_target_avg",
                "FTHG": "home_goals_avg",
                "HC": "home_corners_avg",
                "HF": "home_fouls_avg",
            }
        else:  # away
            team_matches = self.df_historical[self.df_historical["AwayTeam"] == team].copy()
            stats_cols = {
                "AS": "away_shots_avg",
                "AST": "away_shots_on_target_avg",
                "FTAG": "away_goals_avg",
                "AC": "away_corners_avg",
                "AF": "away_fouls_avg",
            }
        
        if len(team_matches) < horizon:
            logger.warning("Équipe %s a moins de %d matchs. Utiliser les données disponibles.", team, horizon)
        
        # Prendre les derniers matchs pour cette équipe
        recent_matches = team_matches.tail(horizon)
        
        rolling_avgs = {}
        for raw_col, feature_col in stats_cols.items():
            avg = recent_matches[raw_col].mean() if len(recent_matches) > 0 else 0.0
            rolling_avgs[feature_col] = avg
        
        return rolling_avgs
    
    def predict_match(self, 
                      home_team: str, 
                      away_team: str,
                      match_id: str = None) -> Optional[Dict]:
        """
        Prédit l'issue d'un match à venir.
        
        :param home_team: nom de l'équipe à domicile
        :param away_team: nom de l'équipe à l'extérieur
        :param match_id: identifiant optionnel du match (pour tracking)
        :return: dict avec {match_id, home_team, away_team, predictions, probabilities}
        """
        try:
            # Récupérer les rolling averages pour chaque équipe
            home_avgs = self._compute_current_rolling_avg(home_team, "home")
            away_avgs = self._compute_current_rolling_avg(away_team, "away")
            
            # Construire le vecteur de features (dans l'ordre exact du modèle)
            feature_dict = {**home_avgs, **away_avgs}
            
            # Vérifier que toutes les features sont présentes
            missing_features = [f for f in self.feature_columns if f not in feature_dict]
            if missing_features:
                logger.error("Features manquantes pour %s vs %s: %s", home_team, away_team, missing_features)
                return None
            
            # Construire le DataFrame d'entrée (1 ligne, features dans le bon ordre)
            X = pd.DataFrame([[feature_dict[f] for f in self.feature_columns]], 
                            columns=self.feature_columns)
            
            # Prédiction
            prediction = self.model.predict(X)[0]  # 0=Away, 1=Draw, 2=Home
            probabilities = self.model.predict_proba(X)[0]  # [P(Away), P(Draw), P(Home)]
            
            result = {
                "match_id": match_id or f"{home_team}_{away_team}",
                "home_team": home_team,
                "away_team": away_team,
                "prediction": ["Away", "Draw", "Home"][prediction],
                "prediction_code": prediction,
                "prob_away": round(probabilities[0], 4),
                "prob_draw": round(probabilities[1], 4),
                "prob_home": round(probabilities[2], 4),
                "home_avg_stats": home_avgs,
                "away_avg_stats": away_avgs,
            }
            
            logger.info("Prédiction: %s vs %s → %s (H=%.1f%%, D=%.1f%%, A=%.1f%%)",
                       home_team, away_team, result["prediction"],
                       result["prob_home"] * 100,
                       result["prob_draw"] * 100,
                       result["prob_away"] * 100)
            
            return result
        
        except Exception as e:
            logger.error("Erreur prédiction %s vs %s: %s", home_team, away_team, str(e))
            return None


class BasketballPredictor:
    """Prédiction pour le basket avec anti-data-leakage."""
    
    def __init__(self,
                 model_path: str = f"{config.MODELS_DIR}/xgb_model_basketball.joblib",
                 features_path: str = f"{config.MODELS_DIR}/features_basketball.joblib",
                 historical_csv: str = f"{config.RAW_DATA_DIR}/nba_stats.csv"):
        """
        :param model_path: chemin vers le modèle XGBoost entraîné
        :param features_path: chemin vers la liste des features
        :param historical_csv: chemin vers le CSV historique NBA
        """
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(features_path)
        
        # Charger et préparer l'historique
        self.df_historical = pd.read_csv(historical_csv)
        self.df_historical = self.df_historical.sort_values("Date").reset_index(drop=True)
        
        logger.info("Modèle basket chargé: %d features", len(self.feature_columns))
        logger.info("Historique chargé: %d matchs", len(self.df_historical))
    
    def _compute_current_rolling_avg(self, team: str, home_or_away: str, horizon: int = 5) -> Dict[str, float]:
        """Calcule la moyenne mobile ACTUALISÉE pour une équipe NBA."""
        if home_or_away == "home":
            team_matches = self.df_historical[self.df_historical["HomeTeam"] == team].copy()
            stats_cols = {
                "HomePTS": "home_pts_avg",
                "Home_FG_Pct": "home_fg_pct_avg",
                "HomeRebounds": "home_rebounds_avg",
                "HomeTurnovers": "home_turnovers_avg",
            }
        else:
            team_matches = self.df_historical[self.df_historical["AwayTeam"] == team].copy()
            stats_cols = {
                "AwayPTS": "away_pts_avg",
                "Away_FG_Pct": "away_fg_pct_avg",
                "AwayRebounds": "away_rebounds_avg",
                "AwayTurnovers": "away_turnovers_avg",
            }
        
        if len(team_matches) < horizon:
            logger.warning("Équipe NBA %s a moins de %d matchs.", team, horizon)
        
        recent_matches = team_matches.tail(horizon)
        
        rolling_avgs = {}
        for raw_col, feature_col in stats_cols.items():
            avg = recent_matches[raw_col].mean() if len(recent_matches) > 0 else 0.0
            rolling_avgs[feature_col] = avg
        
        return rolling_avgs
    
    def predict_match(self, home_team: str, away_team: str, match_id: str = None) -> Optional[Dict]:
        """Prédit l'issue d'un match NBA."""
        try:
            home_avgs = self._compute_current_rolling_avg(home_team, "home")
            away_avgs = self._compute_current_rolling_avg(away_team, "away")
            
            feature_dict = {**home_avgs, **away_avgs}
            
            missing_features = [f for f in self.feature_columns if f not in feature_dict]
            if missing_features:
                logger.error("Features manquantes (NBA) %s vs %s: %s", home_team, away_team, missing_features)
                return None
            
            X = pd.DataFrame([[feature_dict[f] for f in self.feature_columns]], 
                            columns=self.feature_columns)
            
            prediction = self.model.predict(X)[0]  # 0=Away Win, 1=Home Win
            prob_home_win = self.model.predict_proba(X)[0][1]  # P(Home Win)
            
            result = {
                "match_id": match_id or f"{home_team}_{away_team}",
                "home_team": home_team,
                "away_team": away_team,
                "prediction": "Home" if prediction == 1 else "Away",
                "prediction_code": prediction,
                "prob_home_win": round(prob_home_win, 4),
                "prob_away_win": round(1 - prob_home_win, 4),
                "home_avg_stats": home_avgs,
                "away_avg_stats": away_avgs,
            }
            
            logger.info("Prédiction NBA: %s vs %s → %s (H=%.1f%%, A=%.1f%%)",
                       home_team, away_team, result["prediction"],
                       result["prob_home_win"] * 100,
                       result["prob_away_win"] * 100)
            
            return result
        
        except Exception as e:
            logger.error("Erreur prédiction NBA %s vs %s: %s", home_team, away_team, str(e))
            return None


if __name__ == "__main__":
    # Test avec des équipes réelles
    logger.info("=" * 60)
    logger.info("TEST D'INFÉRENCE - FOOTBALL")
    logger.info("=" * 60)
    
    predictor_football = FootballPredictor()
    test_matches = [
        ("Arsenal", "Manchester City"),
        ("Liverpool", "Chelsea"),
        ("Manchester United", "Tottenham"),
    ]
    
    for home, away in test_matches:
        pred = predictor_football.predict_match(home, away)
        if pred:
            print(f"\n{pred['home_team']} vs {pred['away_team']}")
            print(f"  Prédiction: {pred['prediction']}")
            print(f"  P(Home)={pred['prob_home']:.3f}, P(Draw)={pred['prob_draw']:.3f}, P(Away)={pred['prob_away']:.3f}")
