import os
import sys

class CardAbstraction():
    def __init__(self,deck=None):
        self.deck = deck
    
    def abstract(self,cards):
        pass

# NoAbstraction 实际并没有进行卡牌 abstraction
class NoAbstraction(CardAbstraction):
    def __init__(self,deck=None,gapline="|"):
        super().__init__(deck)
        self.gapline = gapline
        self.deck = deck
        
    def abstract(self,cards):
        return self.gapline.join(sorted(cards))
    
class TwoCardAbstraction(CardAbstraction):
    def __init__(self,deck=None):
        super().__init__(deck)
        
    def get_higher_rank(self,rank1,rank2):
        assert(self.deck is not None)
        if self.deck.rankdict[rank1] > self.deck.rankdict[rank2]:
            return rank1
        else:
            return rank2
        
    def abstract(self,cards):
        ''' Takes a hand (array of size two) and compresses the hand into simpler representation
            Also puts higher card in front

            i.e. Th 9h becomes T9s as both cards share the same suit
                Th 9s becomes T9o as both cards do not share the same suit (off-suit)
                Th Ts becomes TT (pair of tens)
        '''
        hand = cards
        rank1 = hand[0][0]
        suit1 = hand[0][1]
        rank2 = hand[1][0]
        suit2 = hand[1][1]

        # pair
        if rank1 == rank2:
            return rank1 + rank2

        hand = self.get_higher_rank(rank1, rank2)
        hand += rank2 if hand==rank1 else rank1
        hand += 's' if suit1==suit2 else 'o'
        return hand
        
class MultiCardPlainAbstraction(CardAbstraction):
    def __init__(self,deck=None):
        super().__init__(deck)
        
    def abstract(self,cards):
        """
        number of hands should be at least two
        """
        hand_abstract = ""
        for one_card in cards:
            hand_abstract += one_card
        return hand_abstract
