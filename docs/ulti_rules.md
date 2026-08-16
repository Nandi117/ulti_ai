# Ulti - Official Game Rules Summary

Ulti is a complex traditional Hungarian trick-taking card game for 3 players, played with a 32-card German-suited deck. It is highly strategic and involves an auction/bidding phase followed by the trick-taking phase.

## 1. Core Concepts
- **Players**: 3 players.
- **Deck**: 32 cards (German/Hungarian suited: Acorns, Leaves, Hearts, Bells; ranks 7, 8, 9, 10, Under, Over, King, Ace).
- **Direction**: Dealing, bidding, and play proceed counter-clockwise.
- **Objective**: The declarer (winner of the auction) must fulfill their bid. Bids can range from taking specific tricks, winning the last trick with the 7 of trumps ("Ulti"), or avoiding all tricks.

## 2. Setup & Dealing
- **Dealing**: The dealer gives 10 cards each to two players and 12 cards to the player to their right (the starter).
- **Talon**: The player with 12 cards sets aside two cards face down to form the *talon*.

## 3. The Auction (Bidding)
- Players bid to become the declarer. 
- If a player decides to bid, they pick up the talon, declare their contract (the game type), and then discard two cards face down to reform the talon.
- Bids have varying point values. Common game types:
  - **Party (Pass)**: A basic game, trumps are chosen, declarer must take more points than defenders combined.
  - **Ulti**: Winning the final trick specifically with the 7 of trumps.
  - **Betli**: A "misere" contract where the declarer must take *zero* tricks.
  - **Durchmarsch**: The declarer must take *all* the tricks.
- Defenders can say **Contra** (double the stakes) if they think the declarer will fail. The declarer can reply with **Rekontra** (redouble).

## 4. Gameplay (Trick-taking)
- The declarer leads the first trick.
- Players *must* follow suit if possible. If they cannot follow suit, they must play a trump card (if the game has trumps). 
- If a player can neither follow suit nor play a trump, they can discard any card.
- A trick is won by the highest trump, or if no trumps are played, the highest card of the led suit.
- The winner of the trick leads the next trick.

## 5. Scoring
- Points are awarded based on the contract, multipliers (Contra/Rekontra), and whether the declarer succeeded or failed.

## Implementation Notes for RL Engine
- **State Space**: Must represent player hands (3x10), the talon (2), current trump suit, bid history, trick history, and score.
- **Action Space**: 
  - Bidding phase: Declare a contract or pass. If declaring, select 2 cards to discard.
  - Play phase: Select a valid card from hand to play. Action masking is strictly required to prevent illegal moves (must follow suit).
- **Rewards**: Complex and delayed. Intermediate rewards may be needed, but ultimate reward is driven by the game score at the end of the hand.
