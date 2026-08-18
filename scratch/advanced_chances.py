import random
from itertools import combinations

SEVEN = 7; EIGHT = 8; NINE = 9; TEN = 10; UNDER = 11; OVER = 12; KING = 13; ACE = 14
SUITS = ['Acorns', 'Bells', 'Leaves', 'Hearts']

def generate_deck():
    return [(s, r) for s in SUITS for r in [SEVEN, EIGHT, NINE, TEN, UNDER, OVER, KING, ACE]]

def evaluate_advanced_ulti(cards12):
    # From 12 cards, can we make an Ulti hand?
    # We need the VII of some suit, and at least 4 other cards of that suit.
    for suit in SUITS:
        suit_cards = [c for c in cards12 if c[0] == suit]
        has_seven = any(c[1] == SEVEN for c in suit_cards)
        # If we have the 7 and at least 4 others (total 5+), it's a very playable Ulti
        if has_seven and len(suit_cards) >= 5:
            return True
    return False

def is_safe_betli_card(card, suit_cards_in_hand):
    # A card is safe if it's naturally low (7,8,9)
    if card[1] in [SEVEN, EIGHT, NINE]: return True
    # Or if it's a King and we DON'T have the Ace (meaning opponents have it, we can lead the King to be killed by the Ace)
    if card[1] == KING and not any(c[1] == ACE for c in suit_cards_in_hand): return True
    # Or if it's an Over and we DON'T have King or Ace
    if card[1] == OVER and not any(c[1] in [KING, ACE] for c in suit_cards_in_hand): return True
    # Or if it's an Under and we DON'T have Over, King, or Ace
    if card[1] == UNDER and not any(c[1] in [OVER, KING, ACE] for c in suit_cards_in_hand): return True
    return False

def evaluate_advanced_betli(cards12):
    # Try to find a 10-card subset that is entirely "safe" Betli cards
    safe_cards = []
    for c in cards12:
        suit_cards = [sc for sc in cards12 if sc[0] == c[0]]
        if is_safe_betli_card(c, suit_cards):
            safe_cards.append(c)
            
    # If we can find at least 10 safe cards from our 12, Betli is extremely playable
    if len(safe_cards) >= 10:
        return True
    return False

def evaluate_advanced_durchmars(cards12):
    # Durchmars needs 10 winning tricks.
    # High cards (Ace, King) are winners. Long suits become winners once high cards are gone.
    # Heuristic: We need at least 5 top-tier cards (Aces/Kings) across the 12 cards, 
    # OR a massive long suit (6+ cards) with the Ace.
    aces = [c for c in cards12 if c[1] == ACE]
    kings = [c for c in cards12 if c[1] == KING]
    
    if len(aces) + len(kings) >= 5:
        return True
        
    for suit in SUITS:
        suit_cards = [c for c in cards12 if c[0] == suit]
        if len(suit_cards) >= 6 and any(c[1] == ACE for c in suit_cards):
            return True
            
    return False

def run_advanced_simulation(num_games=50000):
    ulti_count = 0
    betli_count = 0
    durchmars_count = 0
    any_playable_count = 0
    
    deck = generate_deck()
    
    for _ in range(num_games):
        random.shuffle(deck)
        
        p1 = deck[0:10]
        p2 = deck[10:20]
        p3 = deck[20:30]
        talon = deck[30:32]
        
        game_has_ulti = False
        game_has_betli = False
        game_has_durchmars = False
        
        for p in [p1, p2, p3]:
            # The player gets the talon
            cards12 = p + talon
            
            if evaluate_advanced_ulti(cards12): game_has_ulti = True
            if evaluate_advanced_betli(cards12): game_has_betli = True
            if evaluate_advanced_durchmars(cards12): game_has_durchmars = True
            
        if game_has_ulti: ulti_count += 1
        if game_has_betli: betli_count += 1
        if game_has_durchmars: durchmars_count += 1
        
        if game_has_ulti or game_has_betli or game_has_durchmars:
            any_playable_count += 1
            
    print(f"--- ADVANCED Simulation of {num_games:,} Games ---")
    print(f"Games with a playable Ulti:      {ulti_count / num_games * 100:.2f}%")
    print(f"Games with a playable Betli:     {betli_count / num_games * 100:.2f}%")
    print(f"Games with a playable Durchmars: {durchmars_count / num_games * 100:.2f}%")
    print(f"Games with ANY playable hand:    {any_playable_count / num_games * 100:.2f}%")
    print(f"Games where ALL MUST PASS:       {(1 - any_playable_count / num_games) * 100:.2f}%")

if __name__ == "__main__":
    run_advanced_simulation()
