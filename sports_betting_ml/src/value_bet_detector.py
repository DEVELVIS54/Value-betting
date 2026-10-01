"""
Détecteur de Value Bets Multi-Marchés (H2H + Totals)
Signal pur: EV%, probabilités, cotes. Zéro gestion financière.
Filtre: 1 alerte max par match (meilleur EV retenu)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import numpy as np

from sports_betting_ml import config

logger = logging.getLogger(__name__)


def calculate_ev(probability_ia: float, odds_bookmaker: float) -> float:
    """
    Calcule l'Expected Value (EV) en pourcentage
    EV% = (P × O) - 1, où P=prob IA, O=cote bookmaker
    
    Args:
        probability_ia: Probabilité prédite par l'IA (0-1)
        odds_bookmaker: Cote du bookmaker (décimal)
    
    Returns:
        EV en pourcentage
    """
    if odds_bookmaker <= 0 or probability_ia <= 0:
        return 0.0
    
    ev = (probability_ia * odds_bookmaker) - 1.0
    return ev * 100  # Retourne en %


def detect_value_bets_multimarket(
    odds_data: Dict,
    predictions: Dict,
    ev_threshold: float = 1.0
) -> List[Dict]:
    """
    Scanne H2H + Totals. Détecte EV. Filtre 1 alerte/match (meilleur EV).
    
    Args:
        odds_data: Données brutes The Odds API
        predictions: Prédictions XGBoost pour chaque match
        ev_threshold: Seuil EV minimum (en %)
    
    Returns:
        Liste de value bets (1 par match maximum)
    """
    
    value_bets = []
    bets_by_match = {}  # Groupe par match
    
    games = odds_data.get('games', [])
    logger.info(f"Scan de {len(games)} matchs NBA...")
    
    for game in games:
        match_key = f"{game.get('home_team')}_{game.get('away_team')}"
        
        # Skip si pas de prédiction
        if match_key not in predictions:
            logger.debug(f"⊘ Pas de prédiction pour {match_key}")
            continue
        
        pred = predictions[match_key]
        bookmakers = game.get('bookmakers', [])
        
        if not bookmakers:
            logger.debug(f"⊘ Pas de bookmakers pour {match_key}")
            continue
        
        # === SCAN H2H ===
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') != 'h2h':
                    continue
                
                for outcome in market.get('outcomes', []):
                    odds = outcome.get('price', 0)
                    team_name = outcome.get('name', '')
                    
                    if odds <= 0:
                        continue
                    
                    # Détermine la probabilité selon l'équipe
                    if team_name == game.get('home_team'):
                        prob_ia = pred.get('home_prob_h2h', 0.5)
                    else:
                        prob_ia = pred.get('away_prob_h2h', 0.5)
                    
                    # Calcul EV
                    ev = calculate_ev(prob_ia, odds)
                    
                    if ev >= ev_threshold:
                        bet_h2h = {
                            'timestamp': datetime.now().isoformat(),
                            'match': f"{game['home_team']} vs {game['away_team']}",
                            'home_team': game['home_team'],
                            'away_team': game['away_team'],
                            'market_type': 'H2H',
                            'prediction': f"{team_name} Win",
                            'probability_ia_pct': round(prob_ia * 100, 2),
                            'odds_theorique': round(1.0 / prob_ia, 2) if prob_ia > 0 else 0,
                            'odds_bookmaker': odds,
                            'ev_pct': round(ev, 2),
                            'bookmaker': bookmaker.get('title', 'Unknown')
                        }
                        
                        if match_key not in bets_by_match:
                            bets_by_match[match_key] = []
                        bets_by_match[match_key].append(bet_h2h)
        
        # === SCAN TOTALS ===
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') != 'totals':
                    continue
                
                point_total = market.get('point', 0)
                
                for outcome in market.get('outcomes', []):
                    odds = outcome.get('price', 0)
                    outcome_name = outcome.get('name', '')  # Over ou Under
                    
                    if odds <= 0:
                        continue
                    
                    # Détermine la probabilité selon Over/Under
                    if outcome_name == 'Over':
                        prob_ia = pred.get('total_prob_over', 0.5)
                        prediction_text = f"Over {point_total}"
                    elif outcome_name == 'Under':
                        prob_ia = pred.get('total_prob_under', 0.5)
                        prediction_text = f"Under {point_total}"
                    else:
                        continue
                    
                    # Calcul EV
                    ev = calculate_ev(prob_ia, odds)
                    
                    if ev >= ev_threshold:
                        bet_totals = {
                            'timestamp': datetime.now().isoformat(),
                            'match': f"{game['home_team']} vs {game['away_team']}",
                            'home_team': game['home_team'],
                            'away_team': game['away_team'],
                            'market_type': 'Totals',
                            'prediction': prediction_text,
                            'point': point_total,
                            'probability_ia_pct': round(prob_ia * 100, 2),
                            'odds_theorique': round(1.0 / prob_ia, 2) if prob_ia > 0 else 0,
                            'odds_bookmaker': odds,
                            'ev_pct': round(ev, 2),
                            'bookmaker': bookmaker.get('title', 'Unknown')
                        }
                        
                        if match_key not in bets_by_match:
                            bets_by_match[match_key] = []
                        bets_by_match[match_key].append(bet_totals)
    
    # === FILTRE: 1 ALERTE MAX PAR MATCH ===
    for match_key, bets_on_match in bets_by_match.items():
        if len(bets_on_match) > 1:
            # Multi-alerte: garder celle avec EV maximum
            best_bet = max(bets_on_match, key=lambda x: x['ev_pct'])
            
            dropped = [b for b in bets_on_match if b != best_bet]
            logger.info(
                f"🔄 {match_key}: {len(bets_on_match)} signaux détectés | "
                f"✅ Conservé: {best_bet['market_type']} EV={best_bet['ev_pct']}%"
            )
            
            for d in dropped:
                logger.info(
                    f"   ⊘ Rejeté: {d['market_type']} EV={d['ev_pct']}%"
                )
            
            value_bets.append(best_bet)
        else:
            value_bets.append(bets_on_match[0])
    
    logger.info(f"✅ Après filtrage 1-alerte/match: {len(value_bets)} signaux")
    
    # Résumé par marché
    h2h_count = len([b for b in value_bets if b['market_type'] == 'H2H'])
    totals_count = len([b for b in value_bets if b['market_type'] == 'Totals'])
    logger.info(f"   H2H: {h2h_count} | Totals: {totals_count}")
    
    return value_bets


def export_value_bets_json(value_bets: List[Dict], output_path: str = None) -> str:
    """
    Exporte les signaux en JSON (sans données financières)
    
    Args:
        value_bets: Liste des value bets
        output_path: Chemin du fichier de sortie
    
    Returns:
        Chemin du fichier exporté
    """
    
    if output_path is None:
        output_path = config.ALERTS_OUTPUT_PATH
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Converti types numpy en types standard
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.float64, np.float32, float)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, int)):
            return int(obj)
        return obj
    
    # Structure épurée: zéro mise, zéro kelly
    bets_clean = []
    for bet in value_bets:
        bet_clean = {
            'timestamp': bet['timestamp'],
            'match': bet['match'],
            'market_type': bet['market_type'],
            'prediction': bet['prediction'],
            'probability_ia_pct': bet['probability_ia_pct'],
            'odds_theorique': bet['odds_theorique'],
            'odds_bookmaker': bet['odds_bookmaker'],
            'ev_pct': bet['ev_pct'],
            'bookmaker': bet['bookmaker']
        }
        if 'point' in bet:
            bet_clean['point'] = bet['point']
        bets_clean.append(bet_clean)
    
    # Convertit types numpy
    bets_clean = convert_types(bets_clean)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bets_clean, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📤 Signaux exportés: {output_path} ({len(bets_clean)} signaux)")
    
    return output_path