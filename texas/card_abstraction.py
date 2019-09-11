import os
import sys
from tqdm import tqdm
import itertools
from scipy.special import comb, perm
import numpy as np

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

class AbstractionGenerator():
    def __init__(self,deck=None,dicfile=None):
        self.deck = list(sorted(deck.get_deck()))
        self.dicfile = dicfile
        readed_dic = {}
        with open(self.dicfile) as fhdl:
            for line in fhdl:
                cards,rank = line.strip().split(',')
                rank = int(rank)
                cards = tuple(cards.split('-'))
                readed_dic[cards] = rank
            self.card2rank = readed_dic
            
        self.buckets_len = comb(len(self.deck) -  5,2)
        
        card2ind = {}
        ind2card = {}
        for ind,one_card in enumerate(self.deck):
            card2ind[one_card] = ind
            ind2card[ind] = one_card 
        self.card2ind = card2ind
        self.ind2card = ind2card

    def generate_histrogram(self,public_card_number,bins=20):
        # 公共牌其实只有 0，3，4，5 这几种可能（在德州扑克中）,所以
        # TODO 实现 公共牌有 3,4的情况
        dis = 1.0 / bins
        assert(public_card_number in [0])
        histdic = {}
        
        dic_lens = len(self.publicdic)
        keys = list(self.publicdic.keys())
        for i in tqdm(range(dic_lens)):
            one_key = keys[i]
            val = self.publicdic[one_key]
            for  (private_hand,percent) in val:
                if private_hand not in histdic:
                    histdic[private_hand] = np.zeros(bins)
                
                one_ind = int(percent // dis)
                one_ind = min(one_ind,bins - 1)
                histdic[private_hand][one_ind] += 1
        
        for key in histdic:
            histdic[key] = histdic[key] / np.sum(histdic[key])
        self.histdic = histdic
        
        
    def generate_dict_allpublic(self):
        publicdic = {}
        combs = list(itertools.combinations(self.deck,5))
        for one_public in tqdm(combs):
            publicdic[one_public] = self.process_one_public(one_public)
        self.publicdic = publicdic
        
    def process_one_public(self,public):
        cards = self.deck
        card2ind = self.card2ind
        ind2card = self.ind2card
        
        cards_except_public = set(cards[:]) - set(public)
        cards_except_public = list(cards_except_public)

        private_buckets = []
        for one_private in itertools.combinations(cards_except_public,2):
            fullboard = list(one_private) + list(public)
            fullboard = tuple(sorted(fullboard))
            rank = self.card2rank[fullboard]
            private_buckets.append([sorted(one_private),rank])


        bucket_ele_number = comb(len(cards_except_public),2)
        private_buckets = sorted(private_buckets,key=lambda x:x[1])[::-1]

        ranknum = {}
        beats = {}
        dp_mat = np.zeros([len(cards),int(self.buckets_len) + 1],dtype = np.int)
        #dp_mat -= dp_mat

        last_diff_index = -1
        before_type_last_index = []

        rev_last_diff_ind = len(private_buckets) - 1
        same_type_last_index = []
        for ind,(private_hand,rank) in enumerate(private_buckets):
            rev_ind = len(private_buckets) - 1 - ind
            if rev_ind < len(private_buckets) - 1 and private_buckets[rev_ind][1] != private_buckets[rev_ind + 1][1]:
                rev_last_diff_ind = rev_ind
            same_type_last_index.append(rev_last_diff_ind)

            if ind > 0 and rank != private_buckets[ind - 1][1]: # 如果当前元素的rank 和前一个不同
                last_diff_index = ind - 1
            before_type_last_index.append(last_diff_index)

            ranknum.setdefault(rank,0)
            ranknum[rank] += 1
            beat_number = ind + 1 - (float(ranknum[rank]) )# TODO add this / 2)

            beats.setdefault(rank,0)
            beats[rank] = max(beats[rank],beat_number)

            for one_private in private_hand:
                card_ind = card2ind[one_private]
                dp_mat[card_ind,ind] = 1
            if ind > 0:
                dp_mat[:,ind] += dp_mat[:,ind - 1]

        before_type_last_index.append(0)
        same_type_last_index = same_type_last_index[::-1]

        results = []
        #print("index\tbeats[rank]\tprivate_hand\tnum_beat")
        for ind,(private_hand,rank) in enumerate(private_buckets):
            #percent = float(ind) / bucket_ele_number # 不能这么除
            # TODO subnum的计算需要在同类型手牌最后一个, beats没有必要, 但是需要加上同类型的去掉手牌中出现牌的牌中的一半
            # same_type_last_index = 同类型手牌最后一个index
            # before_type_last_index = 前面不同类型手牌最后一个index
            # 总beats数量 = beats[rank] - 两个dp_mat[一张手牌,before_type_last_index] ##(这是前面不同rank的有效牌数量)
            #               + { ranknum[rank] - 两个 (dp_mat[一张手牌,same_type_last_index] - 
            #                                   dp_mat[一张手牌,before_type_last_index]  + 3)
            #                 } / 2

            ## old solution
            #sub_num = (sum([dp_mat[card2ind[i],ind - 1] for i in private_hand])) if ind else 0
            #print("{}\t{}\t\t{}\t{}".format(ind,beats[rank],private_hand,sub_num))
            #percent = (float(beats[rank]) - sub_num ) / (bucket_ele_number - (len(cards) - 5) * 2 + 1)

            last_diff_index = before_type_last_index[ind]
            rev_last_diff_ind = same_type_last_index[ind]

            #sub_num = (sum([dp_mat[card2ind[i],last_diff_index] for i in private_hand]))
            #percent = float(last_diff_index + 1 - sub_num) / (bucket_ele_number - (len(cards) - 5) * 2 + 1)

            beat_number = beats[rank] - sum([dp_mat[card2ind[i],last_diff_index] for i in private_hand]) \
                        +  (ranknum[rank] -
                             sum([(dp_mat[card2ind[i],rev_last_diff_ind] - dp_mat[card2ind[i],last_diff_index]) \
                               for i in private_hand]) + 2
                        ) / 2
            percent = float(beat_number) / (bucket_ele_number - (len(cards) - 5) * 2 + 1)
            #print("{}\t{}\t\t{}\t{}".format(ind,beats[rank],private_hand,beat_number))

            private_hand = tuple(private_hand)
            #private2ranks.setdefault(private_hand,[])
            #private2ranks[private_hand].append(percent)
            results.append((private_hand,percent))

        return results
