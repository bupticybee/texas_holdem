import random
import numpy as np

class CardSampler():
    def __init__(self,deck):
        self.deck = deck
    
class RandomCardSampler(CardSampler):
    def __init__(self,deck,random_seed=None):
        super().__init__(deck)
        if random_seed is not None:
            random.seed(random_seed)
            
    def sample_card(self,number):
        deck = self.deck.get_deck()
        assert(number <= len(deck))
        sampled_cards = random.sample(deck,number)
        return sampled_cards