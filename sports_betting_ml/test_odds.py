import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_ev(ai_probability, bookmaker_odds, min_ev_threshold=0.5):
    """Calcule l'EV d'une cote"""
    
    if ai_probability <= 0 or bookmaker_odds <= 1:
        return None
    
    bookmaker_prob = 1 / bookmaker_odds
    ev_percent = ((ai_probability * bookmaker_odds) - 1) * 100
    
    is_value = ev_percent >= min_ev_threshold
    
    return {
        'ai_probability': ai_probability,
        'bookmaker_odds': bookmaker_odds,
        'bookmaker_probability': round(bookmaker_prob, 4),
        'ev_percent': round(ev_percent, 2),
        'is_value_bet': is_value,
        'status': '✅ VALUE BET' if is_value else '❌ NO VALUE'
    }


def test_odds():
    """Teste plusieurs cotes"""
    
    logger.info("=" * 70)
    logger.info("TEST CALCULATOR EV")
    logger.info("=" * 70)
    
    # Exemples de tests
    test_cases = [
        {'ai_prob': 0.55, 'odds': 2.0, 'name': 'Lakers -3.5 (AI 55% vs Cote 2.0)'},
        {'ai_prob': 0.52, 'odds': 2.1, 'name': 'Celtics +5 (AI 52% vs Cote 2.1)'},
        {'ai_prob': 0.60, 'odds': 1.8, 'name': 'Warriors (AI 60% vs Cote 1.8)'},
        {'ai_prob': 0.50, 'odds': 2.0, 'name': 'Even match (AI 50% vs Cote 2.0)'},
        {'ai_prob': 0.45, 'odds': 2.5, 'name': 'Underdog (AI 45% vs Cote 2.5)'},
        {'ai_prob': 0.58, 'odds': 1.95, 'name': 'Custom 1 (AI 58% vs Cote 1.95)'},
        {'ai_prob': 0.53, 'odds': 2.05, 'name': 'Custom 2 (AI 53% vs Cote 2.05)'},
    ]
    
    for test in test_cases:
        result = calculate_ev(test['ai_prob'], test['odds'])
        
        logger.info(f"\n📊 {test['name']}")
        logger.info(f"   IA Probabilité : {result['ai_probability']:.1%}")
        logger.info(f"   Bookmaker Cote : {result['bookmaker_odds']}")
        logger.info(f"   Bookmaker Prob : {result['bookmaker_probability']:.1%}")
        logger.info(f"   EV : {result['ev_percent']}%")
        logger.info(f"   {result['status']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)
    logger.info("EV > 0.5% = VALUE BET (profitable à long terme)")
    logger.info("EV < 0.5% = SKIP (pas assez d'edge)")
    logger.info("EV < 0% = MAUVAIS PARI (perte mathématique)")
    logger.info("=" * 70)


def test_custom_odds():
    """Teste avec entrée utilisateur"""
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST PERSONNALISÉ")
    logger.info("=" * 70)
    
    try:
        ai_prob = float(input("Entrée ta probabilité IA (0-1, ex: 0.55): "))
        odds = float(input("Entrée la cote du bookmaker (ex: 2.0): "))
        
        if not (0 < ai_prob < 1) or odds <= 1:
            logger.error("❌ Valeurs invalides")
            return
        
        result = calculate_ev(ai_prob, odds)
        
        logger.info(f"\n📊 RÉSULTAT")
        logger.info(f"   IA Probabilité : {result['ai_probability']:.1%}")
        logger.info(f"   Bookmaker Cote : {result['bookmaker_odds']}")
        logger.info(f"   Bookmaker Prob : {result['bookmaker_probability']:.1%}")
        logger.info(f"   EV : {result['ev_percent']}%")
        logger.info(f"   {result['status']}")
        
    except ValueError:
        logger.error("❌ Erreur : entre des nombres valides")


if __name__ == "__main__":
    test_odds()
    test_custom_odds()
