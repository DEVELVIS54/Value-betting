"""
Telegram Bot Notifier pour Value Betting Signals
Envoie les signaux purs (sans données financières) via Telegram
"""

import logging
import requests
from typing import Dict, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Notifie les value bets via Telegram Bot"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialise le notifieur Telegram
        
        Args:
            bot_token: Token du bot Telegram
            chat_id: ID du chat destinataire
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.sent_bets_file = None
        
        logger.info(f"✅ Notifier Telegram initialisé (Chat ID: {chat_id})")
    
    def set_sent_bets_file(self, filepath: str):
        """Configure le fichier de tracking des bets envoyés"""
        self.sent_bets_file = filepath
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def is_bet_already_sent(self, bet_key: str) -> bool:
        """Vérifie si ce signal a déjà été envoyé"""
        if not self.sent_bets_file or not Path(self.sent_bets_file).exists():
            return False
        
        with open(self.sent_bets_file, 'r', encoding='utf-8') as f:
            sent_keys = [line.strip() for line in f.readlines()]
        
        return bet_key in sent_keys
    
    def mark_bet_as_sent(self, bet_key: str):
        """Marque un signal comme envoyé"""
        if not self.sent_bets_file:
            return
        
        with open(self.sent_bets_file, 'a', encoding='utf-8') as f:
            f.write(f"{bet_key}\n")
    
    def format_signal_message(self, bet: Dict) -> str:
        """
        Formate un signal pour Telegram (signal pur - zéro argent)
        Affiche: Match | Marché | Pari | Prob% | Cote Théo | Cote Réelle | EV%
        
        Args:
            bet: Dict contenant le signal value bet
        
        Returns:
            Message formaté pour Telegram
        """
        
        match_text = f"*{bet['match']}*"
        market_text = f"🏀 {bet['market_type']}"
        
        if bet['market_type'] == 'Totals':
            prediction_text = f"📊 {bet['prediction']}"
        else:
            prediction_text = f"🏆 {bet['prediction']}"
        
        prob_text = f"🤖 IA: {bet['probability_ia_pct']:.1f}%"
        odds_theo = f"📐 Théorique: {bet['odds_theorique']:.2f}"
        odds_real = f"💲 Bookmaker: {bet['odds_bookmaker']:.2f}"
        ev_text = f"✅ EV: +{bet['ev_pct']:.2f}%"
        book_text = f"📍 {bet['bookmaker']}"
        timestamp_text = f"⏰ {bet.get('timestamp', datetime.now().isoformat())}"
        
        message = f"""*{bet['match']}*
━━━━━━━━━━━━━━━━━━━━━━━━
🏀 {bet['market_type']}
{prediction_text}
━━━━━━━━━━━━━━━━━━━━━━━━
🤖 IA: {bet['probability_ia_pct']:.1f}%
📐 Théorique: {bet['odds_theorique']:.2f}
💲 Bookmaker: {bet['odds_bookmaker']:.2f}
✅ EV: +{bet['ev_pct']:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━
📍 {bet['bookmaker']}
⏰ {bet.get('timestamp', datetime.now().isoformat())}
"""
        
        return message
    
    def send_message(self, message: str) -> bool:
        """
        Envoie un message via Telegram
        
        Args:
            message: Contenu du message
        
        Returns:
            True si succès, False sinon
        """
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(self.api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Erreur Telegram: {response.status_code} - {response.text}")
                return False
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur connexion Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur envoi Telegram: {e}")
            return False
    
    def send_signal(self, bet: Dict) -> bool:
        """
        Envoie un signal value bet via Telegram (avec déduplication)
        
        Args:
            bet: Dict contenant le signal
        
        Returns:
            True si envoyé, False sinon
        """
        
        # Clé unique pour déduplication
        bet_key = f"{bet['match']}_{bet['market_type']}_{bet['prediction']}_{bet['odds_bookmaker']}"
        
        # Vérifie si déjà envoyé
        if self.is_bet_already_sent(bet_key):
            logger.debug(f"Signal déjà envoyé: {bet_key}")
            return False
        
        # Formate et envoie
        message = self.format_signal_message(bet)
        
        if self.send_message(message):
            self.mark_bet_as_sent(bet_key)
            logger.info(f"✅ Signal envoyé: {bet['match']} - {bet['market_type']}")
            return True
        else:
            logger.warning(f"⚠️ Échec envoi: {bet['match']}")
            return False
    
    def send_signals_batch(self, bets: List[Dict]) -> Dict:
        """
        Envoie un lot de signaux via Telegram
        
        Args:
            bets: Liste de signaux à envoyer
        
        Returns:
            Dict avec stats (sent, failed, duplicates)
        """
        
        stats = {
            'sent': 0,
            'failed': 0,
            'duplicates': 0,
            'total': len(bets)
        }
        
        for bet in bets:
            bet_key = f"{bet['match']}_{bet['market_type']}_{bet['prediction']}_{bet['odds_bookmaker']}"
            
            if self.is_bet_already_sent(bet_key):
                stats['duplicates'] += 1
                logger.debug(f"⊘ Doublon: {bet_key}")
                continue
            
            if self.send_signal(bet):
                stats['sent'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"📊 Résumé Telegram: {stats['sent']} envoyés, {stats['failed']} échoués, {stats['duplicates']} doublons")
        
        return stats


def send_telegram_signal(notifier: TelegramNotifier, bet: Dict) -> bool:
    """
    Fonction helper pour envoyer un signal unique
    
    Args:
        notifier: Instance TelegramNotifier
        bet: Signal value bet
    
    Returns:
        True si succès
    """
    return notifier.send_signal(bet)


def send_telegram_signals_batch(notifier: TelegramNotifier, bets: List[Dict]) -> Dict:
    """
    Fonction helper pour envoyer un lot de signaux
    
    Args:
        notifier: Instance TelegramNotifier
        bets: Liste de signaux
    
    Returns:
        Stats d'envoi
    """
    return notifier.send_signals_batch(bets)