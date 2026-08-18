import random

# Ranks
SEVEN = 7; EIGHT = 8; NINE = 9; TEN = 10; UNDER = 11; OVER = 12; KING = 13; ACE = 14
SUITS = ['Acorns', 'Bells', 'Leaves', 'Hearts']

# Generate deck
deck = []
for suit in SUITS:
    for rank in [SEVEN, EIGHT, NINE, TEN, UNDER, OVER, KING, ACE]:
        deck.append((suit, rank))

def is_ulti_playable(hand):
    # Has VII and at least 4 other cards of the same suit
    for suit in SUITS:
        suit_cards = [c for c in hand if c[0] == suit]
        has_seven = any(c[1] == SEVEN for c in suit_cards)
        if has_seven and len(suit_cards) >= 5:
            return True
    return False

def is_betli_playable(hand):
    # No Aces, No Kings, at most 1 Over/Under
    aces_kings = [c for c in hand if c[1] in [ACE, KING]]
    overs_unders = [c for c in hand if c[1] in [OVER, UNDER]]
    if len(aces_kings) == 0 and len(overs_unders) <= 1:
        return True
    return False

def is_durchmars_playable(hand):
    # Very strong high cards: At least 3 Aces and 2 Kings, OR no cards below 10
    aces = [c for c in hand if c[1] == ACE]
    kings = [c for c in hand if c[1] == KING]
    low_cards = [c for c in hand if c[1] in [SEVEN, EIGHT, NINE]]
    
    if len(aces) >= 3 and len(kings) >= 2:
        return True
    if len(low_cards) == 0:
        return True
    return False

def run_simulation(num_games=100000):
    ulti_count = 0
    betli_count = 0
    durchmars_count = 0
    any_playable_count = 0
    
    for _ in range(num_games):
        random.shuffle(deck)
        # We check all 3 players
        p1 = deck[0:10]
        p2 = deck[10:20]
        p3 = deck[20:30]
        
        game_has_ulti = False
        game_has_betli = False
        game_has_durchmars = False
        
        for p in [p1, p2, p3]:
            if is_ulti_playable(p): game_has_ulti = True
            if is_betli_playable(p): game_has_betli = True
            if is_durchmars_playable(p): game_has_durchmars = True
            
        if game_has_ulti: ulti_count += 1
        if game_has_betli: betli_count += 1
        if game_has_durchmars: durchmars_count += 1
        
        if game_has_ulti or game_has_betli or game_has_durchmars:
            any_playable_count += 1
            
    print(f"--- Simulation of {num_games:,} Games ---")
    print(f"Games with a playable Ulti hand:      {ulti_count / num_games * 100:.2f}%")
    print(f"Games with a playable Betli hand:     {betli_count / num_games * 100:.2f}%")
    print(f"Games with a playable Durchmars hand: {durchmars_count / num_games * 100:.2f}%")
    print(f"Games with ANY playable hand:         {any_playable_count / num_games * 100:.2f}%")
    print(f"Games where ALL MUST PASS:            {(1 - any_playable_count / num_games) * 100:.2f}%")

if __name__ == "__main__":
    run_simulation()
