import os
import sys
import numpy as np
from copy import deepcopy
import random
import itertools

class PublicChanceCFR():
    def __init__(self,deck,card_sampler,compairer,rule,tree_builder,card_abstraction,strategy_profile,iterations=100,print_interval=10,debug=False):
        # 牌堆
        self.deck = deck
        # 抽牌器
        self.card_sampler = card_sampler
        # 手牌大小比较器
        self.compairer = compairer
        # 规则
        self.rule = rule
        self.players = self.rule.players
        # 游戏树
        self.tree_builder = tree_builder
        self.treeroot = tree_builder.root
        
        # 可配置参数
        # 迭代轮数
        self.iterations = iterations
        # 打印间隔
        self.print_interval = print_interval
        # 卡牌 abstraction
        self.card_abstraction = card_abstraction
        # 训练出来的模型
        self.strategy_profile = strategy_profile
        
        # 是否调试模式
        self.debug = debug
        
    def train(self):
        util = np.zeros(self.players)
        
        #sampled_cards = [i.split(',') for i in open('/mnt/cards.txt').read().strip().split("\n")]
        # 训练iterations 轮
        for i in range(self.iterations):
            
            if i % self.print_interval == 0 and i != 0:
                print("Expected value after %i iterations: %s" % (i, util / i))
            # 抽取 每个人手牌 * 游戏人数  的扑克牌
            publiccards = self.card_sampler.sample_card(sum(self.rule.rounds_cards))
            self.publiccards = publiccards
            self.deck_remain = self.deck.get_deck() - set(publiccards)
            self.hc_pairs = list(itertools.combinations(self.deck_remain,2))
            
            if self.debug:
                print(publiccards)
            
            reach_probs = {}
            for one_players in range(self.players):
                state_dic = {}
                for one_hc in self.hc_pairs:
                    state_dic[one_hc] = 1 / len(self.hc_pairs)
                reach_probs[one_players] = state_dic
                        
            node = self.treeroot
            self.cfr(node,reach_probs)
        return util
    
    def get_infoset_str(self,private,public,node):
        return "b{}_{}".format(self.card_abstraction.abstract((private,public)),",".join(node.bet_history))
    
    def cfr(self,node,reach_probs):
        player = node.player
        strategies = {}
        if node.terminal == True:
            # 当一方弃牌（fold）的时候，node的状态会达到terminal,返回预先计算好的在这种情况下的各方payoff
            payoffs = {}
            for one_player in players:
                player_payoffs = {}
                for one_hc in self.hc_pairs:
                    player_payoffs[one_hc] = reach_probs[one_player][one_hc] * node.payoff[one_player]
                payoffs[one_player] = player_payoffs
            return payoffs
        elif node.showdown == True:
            # 当双方摊派的情况，单方call或者双方check的时候，会到达 showdown状态
            payoffs = {}
            assert(player in [0,1])
            for one_player in range(self.rule.players):
                hc_win = {}
                payoffs[one_player] = {} 
                for hc_i in self.hc_pairs:
                    for jc_j in self.hc_pairs:
                        winner = self.compairer.larger(hc_i,hc_j,self.publiccards)
                        if winner == True:
                            hc_win[hc_i] += reach_probs[1 - one_player][hc_i]
                        elif winner == False:
                            pass
                        else:
                            hc_win[hc_i] += (reach_probs[1 - one_player])[hc_i] / 2
                payoffs[one_player] = hc_win
            for one_player in range(self.rule.players):
                # TODO 考虑tie的情况
                for one_hc in self.hc_pairs:
                    prob_win = payoffs[one_player][one_hc]
                    payoffs[one_player][one_hc] = reach_probs[one_player][one_hc] * (prob_win * node.payoffs[one_player][one_player] + (1 - prob_win) * node.payoffs[1 - one_player][one_player])
            return payoffs
        else:
            possible_actions = list(node.children.keys())
            payoffs = {}
            
            # 这里需要做的事情（1）计算各个infoset的payoff（2）更新策略strategy profile
            # strategies <hc:action:prob>
            strategies = {}
            for each_hc in self.hc_pairs:
                each_infoset = self.get_infoset_str(each_hc,self.publiccards[:node.betting_round - 1],node)
                # TODO try not to use random uniform initization
                # infoset_strategy like <key:value> -> <action,prob>
                infoset_strategy = self.strategy_profile.get_strategy(each_infoset,possible_actions,reach_probs[player][each_hc])
                strategies[one_hc] = infoset_strategy
                
            # payoffs like <player:hc:value>
            payoffs = {}
            for one_player in range(self.rule.players):
                for one_hc in self.hc_pairs:
                    payoffs[one_player][one_hc] = 0
                
            for one_action in possible_actions:
                reach_prob_next = copy.deepcopy(reach_probs)
                for one_hc in self.hc_pairs:
                    reach_prob_next[1 - player][one_hc] *= strategies[one_hc][one_action]
                # child_payoffs like <player:hc:payoff>
                child_payoffs = self.cfr(root.children[one_action],reach_prob_next)
                # TODO 将child_payoffs 整合进payoffs,然后返回
                for one_hc in self.hc_pairs:
                    payoffs[1 - player][one_hc] += child_payoffs[1 - player][one_hc]
                    payoffs[player][one_hc] += strategies[one_hc][one_action] * child_payoffs[player][one_hc]
            
            return payoffs

        
class StrategyNode(object):
    """
    在一个infoset中的策略在这个结构中存储和操作
    """
    def __init__(self,actions):
        self.actions = actions
        
        self.regret_sum_ = dict()
        self.strategy_ = dict()
        self.strategy_sum_ = dict()

        for action in actions:
            self.regret_sum_[action] = 0.0
            self.strategy_[action] = 0.0
            self.strategy_sum_[action] = 0.0

        
    def get_strategy(self,reach_prob):
        normalizing_sum = 0

        for action in self.actions:
            self.strategy_[action] = self.regret_sum_[action] if self.regret_sum_[action] > 0 else 0
            normalizing_sum += self.strategy_[action]

        for action in self.actions:
            if normalizing_sum > 0:
                self.strategy_[action] /= normalizing_sum
            else:
                self.strategy_[action] = 1.0 / len(self.actions)
            # 一个info set的 strategy sum是由各轮迭代的 strategy的加权sum组成的，权值是这个infoset某一轮迭代中
            #       的reach prob
            # 重要： 这个reach prob的权制指这个info set的所有者 根据自己的选择到达这个infoset的概率，假设对手总是
            #       选择到达这个infoset的动作
            self.strategy_sum_[action] += reach_prob * self.strategy_[action]

        return self.strategy_
    
    def get_average_strategy(self):
        average_strategy = dict
        normalizing_sum = 0

        for action in self.actions:
            normalizing_sum += self.strategy_sum_[action]

        for action in self.actions:
            if normalizing_sum > 0:
                average_strategy[action] = self.strategy_sum_[action] / normalizing_sum
            else:
                average_strategy[action] = 1.0 / len(self.actions);

        return average_strategy
        
                
class StrategyProfile(object):
    """
    Strategy Profile 用来记录在一局游戏中的策略，相当于训练出来的模型
    控制所有infoset中的策略
    这些infoset存储在名为strategy的字典中，key为这些infoset的名字，val为一个个StrategyNode的策略记录
    """
    def __init__(self):
        self.strategy = {}
        
    def get_strategy(self,infoset,possible_actions,reach_prob):
        if infoset not in self.strategy:
            self.strategy[infoset] = StrategyNode(possible_actions)
        return self.strategy[infoset].get_strategy(reach_prob)
    
    def update_regret(self,infoset,action,weighted_regret):
        self.strategy[infoset].regret_sum_[action] += weighted_regret
    