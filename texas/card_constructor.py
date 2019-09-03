
class DeckConstructor:
    def __init__(self):
        self.rankdict = None
        pass
    
    def get_deck(self):
        pass

class CartesianConstructor(DeckConstructor):
    def __init__(self,rank,suit):
        self.rank = rank
        self.suit = suit
        self.rankdict = {}
        self.deck = [rank + suit for rank in self.rank for suit in self.suit]
        self.deck = set(self.deck)
        for i,one_rank in enumerate(self.rank[::-1]):
            self.rankdict[one_rank] = i 
    
    def get_deck(self):
        return self.deck
    