import re

file_path = r'C:\ulti_ai\engine\environments\ulti.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacement = '''            deck.shuffle()
            forced_bid_id = options.get("forced_bid_id") if options else None
            if forced_bid_id is not None:
                import random
                from engine.core import Rank, Suit
                target_suit = Suit.HEARTS if forced_bid_id in [1, 3, 5, 7, 9] else Suit.ACORNS
                cards = deck.cards
                p0_desired = []
                if forced_bid_id in [4, 5]: # Betli
                    safe = [c for c in cards if c.rank not in [Rank.ACE, Rank.KING, Rank.OVER]]
                    if len(safe) >= 12:
                        p0_desired = random.sample(safe, 12)
                elif forced_bid_id in [8, 9]: # Durchmars
                    aces = [c for c in cards if c.rank == Rank.ACE]
                    kings = [c for c in cards if c.rank == Rank.KING]
                    others = [c for c in cards if c not in aces and c not in kings]
                    p0_desired = aces + kings + random.sample(others, 12 - len(aces) - len(kings))
                elif forced_bid_id in [6, 7]: # Ulti
                    trumps = [c for c in cards if c.suit == target_suit]
                    vii = [c for c in trumps if c.rank == Rank.SEVEN][0]
                    high_trumps = [c for c in trumps if c.rank in [Rank.ACE, Rank.KING, Rank.OVER, Rank.TEN]]
                    others = [c for c in cards if c not in high_trumps and c != vii]
                    p0_desired = [vii] + high_trumps[:3] + random.sample(others, 8)
                elif forced_bid_id in [2, 3]: # 40-100
                    trumps = [c for c in cards if c.suit == target_suit]
                    k = [c for c in trumps if c.rank == Rank.KING][0]
                    o = [c for c in trumps if c.rank == Rank.OVER][0]
                    aces = [c for c in cards if c.rank == Rank.ACE and c.suit != target_suit]
                    others = [c for c in cards if c not in [k, o] and c not in aces]
                    p0_desired = [k, o] + aces + random.sample(others, 12 - 2 - len(aces))
                    
                if p0_desired:
                    remaining = [c for c in cards if c not in p0_desired]
                    deck.cards = p0_desired + remaining
            
            self.hands = [
'''

content = content.replace('            deck.shuffle()\n            self.hands = [', replacement)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced deck.shuffle successfully.")
