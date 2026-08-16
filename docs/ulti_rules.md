# Rablóulti - Official Game Rules & Full Bidding Table

Rablóulti (Robber Ulti) is the most popular and complex variant of the traditional Hungarian trick-taking card game. It involves a massive bidding phase where players can declare complex combinations of contracts.

## 1. Core Concepts
- **Players**: 3 players.
- **Deck**: 32-card German-suited deck.
- **Direction**: Counter-clockwise.

## 2. The Auction (Bidding)
In Rablóulti, the bidding is highly combinatorial. A player can combine multiple basic contracts into a single bid. If a player bids, they can pick up the talon (2 cards) and discard 2 cards.

### Basic Contracts:
1. **Passz (Party)**: Simply taking more points than the defenders.
2. **Ulti**: Winning the last (10th) trick with the 7 of trumps.
3. **Betli**: A misere game (taking NO tricks).
4. **Durchmars**: Taking ALL the tricks.
5. **20-100 / 40-100**: Declaring that the player will reach 100 points via card values and "bélák" (King-Upper pairs of the same suit: 40 points in trump, 20 in other suits).
6. **Négy ász**: Taking all four Aces.

### Color Multipliers:
- **Piros (Red/Hearts)**: If the trump suit is chosen as Hearts, the base value of the contract is doubled!

### Combinations & Values (Extracted from Wikipedia):
The combinations scale immensely in points. Here is the structure of the most notable bids and their base points (which can double if the bid fails):
- **Passz**: 1 point (Piros passz: 2 points)
- **40-100**: 4 points (Piros 40-100: 8 points)
- **Ulti**: 4+1 points (Piros ulti: 8+2 points)
- **Betli**: 5 points (Piros betli: 10 points)
- **Durchmars**: 6 points (Piros durchmars: 12 points)
- **40-100 ulti**: 8 points (Piros 40-100 ulti: 16 points)
- **20-100**: 8 points (Piros 20-100: 16 points)
- **Ulti durchmars**: 10 points (Piros ulti durchmars: 20 points)
- **20-100 ulti**: 12 points (Piros 20-100 ulti: 24 points)
- **Terített (Open/Spread)**: Increases the value massively (e.g., Terített durchmars is 12 points, Piros terített durchmars is 24 points).
- **Maximum Bid**: Piros 20-100 ulti terített durchmars (48 points).

## 3. Implementation Planning for RL Agent
Because of Rablóulti's combinatorial explosion of bids, our Action Space for the bidding phase must be carefully designed. 
- **Bidding Action Space**: Instead of a flat discrete list of 50+ combinations, we may need a MultiDiscrete action space (e.g., `[Trump Suit, Base Game, Ulti Flag, 100 Flag, 4 Aces Flag, Spread Flag]`).
- **Action Masking**: The mask must strictly enforce the hierarchy of bids. A player can only bid if the total point value of their bid is strictly greater than the previous bid!
