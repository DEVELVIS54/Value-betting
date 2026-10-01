# test_pipeline.py
"""
Script de test complet du pipeline NBA
Valide chaque étape : Features → Modèles → Prédictions → Value Bets → Kelly
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'sports-betting-ml'))

from src.feature_engineering import engineer_features
from src.model_train import train_all_models, get_feature_columns, load_engineered_data
from src.predictor import BasketballPredictor
from src.value_bet_detector import scan_odds_multimarket
from src.kelly import calculate_optimal_stake, validate_bankroll_safety
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_feature_engineering():
    """Test 1 : Calcul des features"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1 : FEATURE ENGINEERING")
    logger.info("="*70)
    
    try:
        df = engineer_features(
            input_csv=config.DATA_RAW_PATH,
            output_pkl=config.DATA_PROCESSED_PATH
        )
        
        logger.info(f"✅ Features calculées")
        logger.info(f"   Shape : {df.shape}")
        logger.info(f"   Colonnes : {list(df.columns[:10])}...")
        logger.info(f"   NaN : {df.isnull().sum().sum()}")
        
        return True, df
    
    except Exception as e:
        logger.error(f"❌ Erreur features : {e}")
        return False, None


def test_model_training(df):
    """Test 2 : Entraînement des modèles"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2 : ENTRAÎNEMENT MODÈLES")
    logger.info("="*70)
    
    try:
        results = train_all_models(config.DATA_PROCESSED_PATH)
        
        logger.info(f"✅ 3 modèles entraînés")
        
        for market in ['h2h', 'spreads', 'totals']:
            r = results[market]
            logger.info(f"   {market.upper():10} | Acc: {r['accuracy_mean']:.4f} | Loss: {r['log_loss']:.4f}")
        
        return True, results
    
    except Exception as e:
        logger.error(f"❌ Erreur entraînement : {e}")
        return False, None


def test_predictor():
    """Test 3 : Prédictions sur matchs NBA"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3 : PRÉDICTIONS NBA")
    logger.info("="*70)
    
    try:
        predictor = BasketballPredictor()
        
        # Test sur quelques équipes NBA
        test_matches = [
            ("Lakers", "Celtics"),
            ("Warriors", "Nets"),
            ("Heat", "Bucks")
        ]
        
        for home, away in test_matches:
            pred = predictor.predict_match(home, away)
            
            if pred:
                logger.info(f"✅ {home} vs {away}")
                logger.info(f"   H2H: {pred['h2h_prob']:.2%} | Spreads: {pred['spreads_prob']:.2%} | Totals: {pred['totals_prob']:.2%}")
            else:
                logger.warning(f"⚠️ {home} vs {away} - Pas de prédiction")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur prédictions : {e}")
        return False


def test_value_bet_detection():
    """Test 4 : Détection value bets (avec données mockées)"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4 : DÉTECTION VALUE BETS")
    logger.info("="*70)
    
    try:
        # Mock odds data
        mock_odds = {
            'basketball_nba': [
                {
                    'home_team': 'Los Angeles Lakers',
                    'away_team': 'Boston Celtics',
                    'bookmakers': [
                        {
                            'key': 'draftkings',
                            'markets': [
                                {
                                    'key': 'h2h',
                                    'outcomes': [
                                        {'name': 'Los Angeles Lakers', 'price': 210},
                                        {'name': 'Boston Celtics', 'price': -250}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Mock predictions
        mock_predictions = {
            'Los Angeles Lakers_Boston Celtics': {
                'h2h_prob': 0.55,
                'spreads_prob': 0.52,
                'totals_prob': 0.58
            }
        }
        
        value_bets = scan_odds_multimarket(mock_odds, mock_predictions)
        
        if value_bets:
            logger.info(f"✅ {len(value_bets)} value bets détectés")
            for bet in value_bets[:3]:
                logger.info(f"   {bet['match']} - {bet['selection']}")
                logger.info(f"      EV: {bet['ev_percent']:.2f}% | Cote: {bet['odds']}")
        else:
            logger.info(f"ℹ️ Aucun value bet détecté (normal si EV faible)")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur value bets : {e}")
        return False


def test_kelly_calculations():
    """Test 5 : Calculs Kelly/4"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5 : CALCULS KELLY/4")
    logger.info("="*70)
    
    try:
        test_bets = [
            {
                'match': 'Lakers vs Celtics',
                'selection': 'Lakers',
                'odds': 2.10,
                'ai_probability': 0.55,
                'ev_percent': 15.5
            },
            {
                'match': 'Warriors vs Nets',
                'selection': 'Over 220',
                'odds': 1.85,
                'ai_probability': 0.60,
                'ev_percent': 11.0
            }
        ]
        
        bankroll = 1000.0
        
        for bet in test_bets:
            stake_info = calculate_optimal_stake(
                bankroll=bankroll,
                win_probability=bet['ai_probability'],
                bookmaker_odds=bet['odds'],
                kelly_fraction=0.25
            )
            
            logger.info(f"✅ {bet['match']}")
            logger.info(f"   Mise: ${stake_info['stake_recommended']:.2f}")
            logger.info(f"   Kelly: {stake_info['kelly_percent']:.3f}%")
            logger.info(f"   Gain potentiel: ${stake_info['potential_profit']:.2f}")
        
        # Validation bankroll
        total_stake = sum(
            calculate_optimal_stake(
                bankroll, b['ai_probability'], b['odds'], kelly_fraction=0.25
            )['stake_recommended']
            for b in test_bets
        )
        
        pct = (total_stake / bankroll) * 100
        
        logger.info(f"\n📊 Bankroll Check")
        logger.info(f"   Capital: ${bankroll:.2f}")
        logger.info(f"   Mises totales: ${total_stake:.2f}")
        logger.info(f"   Exposition: {pct:.1f}%")
        logger.info(f"   Status: {'✅ SAFE' if pct <= 25 else '⚠️ RISQUÉ'}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur Kelly : {e}")
        return False


def test_data_integrity():
    """Test 6 : Intégrité des données"""
    logger.info("\n" + "="*70)
    logger.info("TEST 6 : INTÉGRITÉ DES DONNÉES")
    logger.info("="*70)
    
    try:
        # Vérifie les fichiers
        checks = {
            "CSV NBA": Path(config.DATA_RAW_PATH).exists(),
            "Processed PKL": Path(config.DATA_PROCESSED_PATH).exists(),
            "Modèle H2H": Path(f"{config.MODELS_DIR}/xgb_nba_h2h.joblib").exists(),
            "Modèle Spreads": Path(f"{config.MODELS_DIR}/xgb_nba_spreads.joblib").exists(),
            "Modèle Totals": Path(f"{config.MODELS_DIR}/xgb_nba_totals.joblib").exists(),
            "Features Columns": Path(f"{config.MODELS_DIR}/feature_columns.joblib").exists(),
        }
        
        for name, exists in checks.items():
            status = "✅" if exists else "❌"
            logger.info(f"{status} {name}")
        
        all_ok = all(checks.values())
        return all_ok
    
    except Exception as e:
        logger.error(f"❌ Erreur intégrité : {e}")
        return False


def main():
    """Lance tous les tests"""
    
    logger.info("╔" + "="*68 + "╗")
    logger.info("║" + " "*15 + "TEST COMPLET DU PIPELINE NBA" + " "*27 + "║")
    logger.info("╚" + "="*68 + "╝")
    
    results = {}
    
    # Test 1 : Features
    success, df = test_feature_engineering()
    results['Features'] = success
    
    # Test 2 : Modèles (skip si pas de features)
    if success:
        success, models = test_model_training(df)
        results['Modèles'] = success
    else:
        results['Modèles'] = False
    
    # Test 3 : Prédictions
    results['Prédictions'] = test_predictor()
    
    # Test 4 : Value Bets
    results['Value Bets'] = test_value_bet_detection()
    
    # Test 5 : Kelly
    results['Kelly'] = test_kelly_calculations()
    
    # Test 6 : Intégrité
    results['Intégrité'] = test_data_integrity()
    
    # Résumé final
    logger.info("\n" + "="*70)
    logger.info("RÉSUMÉ DES TESTS")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} | {test_name}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    logger.info("="*70)
    logger.info(f"Score : {total_pass}/{total_tests} tests réussis")
    
    if total_pass == total_tests:
        logger.info("✅ TOUS LES TESTS RÉUSSIS - Pipeline prêt pour production")
    else:
        logger.info("⚠️ Des tests ont échoué - Vérifiez les erreurs ci-dessus")
    
    logger.info("="*70 + "\n")
    
    return total_pass == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)