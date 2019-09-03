import itertools

class CardCompairer():
    def __init__(self,deck):
        self.deck = deck
        pass
    
    def larger(self,hand1,hand2):
        """
        land1,hand2: list like objects contains cards, e.g. ['Ts', '8d', '9h', '7d']
        return True if hand1 > hand 2  
        return False if hand1 < hand 2 
        return None if hand1 == hand2
        """
        pass
    
class TwoCardTexasCompairer(CardCompairer):
    def get_higher_rank(self,rank1,rank2):
        assert(self.deck is not None)
        if self.deck.rankdict[rank1] > self.deck.rankdict[rank2]:
            return rank1
        else:
            return rank2
        
    
    def simplify_hand(self, hand):
        ''' Takes a hand (array of size two) and compresses the hand into simpler representation
            Also puts higher card in front

            i.e. Th 9h becomes T9s as both cards share the same suit
                Th 9s becomes T9o as both cards do not share the same suit (off-suit)
                Th Ts becomes TT (pair of tens)
        '''
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

    def larger(self,hand1,hand2):
        hand1 = self.simplify_hand(hand1)
        hand2 = self.simplify_hand(hand2)
        retval = self.get_winner(hand1,hand2)
        if retval == 1:
            return True
        elif retval == 2:
            return False
        else:
            return None
        
    def get_winner(self, hand1, hand2):
        ''' Gets the winner between the two hands
            Pair > Suited > Off-suited
            If two hands are in the same category, then the higher card of each hand
            breaks the tie, followed by the second card

            returns 1 if the first hand wins
            returns 2 if the second hand wins
            returns 0 if the hands are tied
        '''

        is_hand1_pair = hand1[0]==hand1[1]
        is_hand2_pair = hand2[0]==hand2[1]

        # both pair
        if is_hand1_pair and is_hand2_pair:
            if hand1[0] == hand2[0]:
                return 0
            if hand1[0] == self.get_higher_rank(hand1[0],hand2[0]):
                return 1
            else:
                return 2
        elif is_hand1_pair:
            return 1
        elif is_hand2_pair:
            return 2

        is_hand1_suited = hand1[2]=='s'
        is_hand2_suited = hand2[2]=='s'

        # both suited
        if is_hand1_suited and is_hand2_suited:
            if hand1[0] == hand2[0]:
                if hand1[1] == hand2[1]:
                    return 0
                if hand1[1] == self.get_higher_rank(hand1[1], hand2[1]):
                    return 1
                return 2
            if hand1[0] == self.get_higher_rank(hand1[0],hand2[0]):
                return 1
            else:
                return 2
        elif is_hand1_suited:
            return 1
        elif is_hand2_suited:
            return 2

        # both unsuited
        if hand1[0] == hand2[0]:
            if hand1[1] == hand2[1]:
                return 0
            if hand1[1] == self.get_higher_rank(hand1[1], hand2[1]):
                return 1
            return 2
        if hand1[0] == self.get_higher_rank(hand1[0],hand2[0]):
            return 1
        else:
            return 2
        
        
    
class FiveCardDicTexasCompairer(CardCompairer):
    def __init__(self,deck,dicfile):
        super().__init__(deck)
        self.dicfile = dicfile
        readed_dic = {}
        with open(self.dicfile) as fhdl:
            for line in fhdl:
                cards,rank = line.strip().split(',')
                rank = int(rank)
                cards = tuple(cards.split('-'))
                readed_dic[cards] = rank
            self.card5dic = readed_dic
    
    def rank(self,hand):
        # rank越小牌越大
        assert(len(hand) == 7)
        minrank = None
        for each_fivecard in itertools.combinations(hand,5):
            onerank = self.card5dic[each_fivecard]
            minrank = onerank if minrank is None else min(onerank,minrank)
        return minrank
        
    def larger(self,hand1,hand2,publiccard):
        # 德州扑克每个人 两张牌，public card有5张
        assert(len(hand1) == 2 and len(hand2) == 2 and len(publiccard) == 5)
        hand1 = hand1 + publiccard
        hand2 = hand2 + publiccard
        
        hand1 = sorted(hand1)
        hand2 = sorted(hand2)
        
        # 德州扑克 更大的一方是 两个人 所有手牌+public牌中最大的那个5张牌 的拥有者。
        rank1 = self.rank(hand1)
        rank2 = self.rank(hand2)
        if rank1 < rank2:
            return True
        elif rank1 > rank2:
            return False
        else:
            return None
        