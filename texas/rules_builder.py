import os
import sys
from utils.utils import Struct

class RulesBuilder():
    def __init__(self,rule):
        self.rule = rule
        self.parse_rules()
    
    def parse_rules(self):
        pass
    
    def get_beginning_ante(self):
        pass
    
        
class TwoCardRulesBuilder(RulesBuilder):
    """
    two card texas holdem game tree builder
    """
    def parse_rules(self):
        for k,v in self.rule.items():
            setattr(self,k,v)
        rule = Struct(**self.rule)
        self.rule = rule
    
    def get_beginning_chip(self):
        antes = [self.rule.amounts['ante'] for i in range(self.players)]
        return antes
    
    
class FiveCardRulesBuilder(RulesBuilder):
    """
    two card texas holdem game tree builder
    """
    def parse_rules(self):
        for k,v in self.rule.items():
            setattr(self,k,v)
        rule = Struct(**self.rule)
        self.rule = rule
    
    def get_beginning_chip(self):
        assert(self.players >= 2)
        # 中间不下注的人的 bet 都是0
        midbets = []
        for i in range(self.players - 2):
            midbets.append(0)
        chips = [self.rule.amounts['small_blind']] + midbets + [self.rule.amounts['big_blind']]
        return chips 
    