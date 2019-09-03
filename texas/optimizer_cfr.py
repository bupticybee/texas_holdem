import os
import sys
import numpy as np
from copy import deepcopy
import random

class CFR():
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
            cards = self.card_sampler.sample_card(self.rule.deal_card_number * self.players)
            #cards = sampled_cards[i]
            if self.debug:
                print(cards)
            
            players_cards = []
            for i in range(self.players):
                # 每个人的手牌
                one_hand,cards = cards[:self.rule.deal_card_number],cards[self.rule.deal_card_number:]
                players_cards.append(one_hand)

            reach_probs = np.ones(self.players)

            # cache 住infoset 和winner
            self.infosets = [self.card_abstraction.abstract(i) for i in players_cards]
            self.winner = self.compairer.larger(players_cards[0],players_cards[1])
                        
            node = self.treeroot
            util += self.cfr(players_cards,node,reach_probs)
        return util / i
                
    def cfr(self,cards,node,reach_probs):
        player = node.player
        # 一个infoset是有私有信息（自己的手牌） 和公共信息（双方的下注，这里记录在node中）构成的
        holdcard = cards[player]
        infoset = self.infosets[player] + node.serialize()#self.card_abstraction.abstract(holdcard) + node.serialize()
        
        if node.terminal == True:
            if self.debug:
                print(cards,node.serialize(),node.payoff)
            # 当一方弃牌（fold）的时候，node的状态会达到terminal,返回预先计算好的在这种情况下的各方payoff
            return node.payoff
        elif node.showdown == True:
            # 当双方摊派的情况，单方call或者双方check的时候，会到达 showdown状态
            # 只考虑了两个玩家的情况
            # TODO cache住这个winner，在一次cfr迭代中，会有多个地方对同样的卡牌进行winner判定
            winner = self.winner#self.compairer.larger(cards[0],cards[1])
            if winner is None:
                if self.debug:
                    print(cards,node.serialize(),node.payoffs['tie'])
                return node.payoffs['tie']
            elif winner == True:
                if self.debug:
                    print(cards,node.serialize(),node.payoffs[0])
                # player 0的牌比player 1的牌大,返回预先计算好的在这种情况下的各方payoff
                return node.payoffs[0]
            elif winner == False:
                if self.debug:
                    print(cards,node.serialize(),node.payoffs[1])
                # player 0的牌比player 1的牌小
                return node.payoffs[1]
            
        # 当没有人fold，并且也还没到摊派的时候,那么这个时候 这个node是一定有child（下一步）的,因为游戏还没结束
        possible_actions = list(node.children.keys())
        # 自己的的reach prob
        reach_prob = reach_probs[player]
        # 对手的reach prob
        reach_prob_oppo = reach_probs[(player + 1) % self.players]
        
        strategy_dic = self.strategy_profile.get_strategy(infoset,possible_actions,reach_prob)
        
        # 预期收益,字典，key为动作，value为预期这个动作产生的收益
        util = {}
        
        # 节点的预期收益（对于id为player的玩家）
        node_util = [0 for i in range(self.players)]
        for action in possible_actions:
            next_prob = [i for i in reach_probs]#deepcopy(reach_probs)
            next_prob[player] *= strategy_dic[action]
            
            util[action] = self.cfr(cards,node.children[action],next_prob)
            # node utils 对各个子节点的预期收益进行加权平均，权值为strategy profile中的策略
            for i in range(self.players):
                node_util[i] += strategy_dic[action] * util[action][i]
            
        if self.debug:
            print(cards,node.serialize(),node_util)
            print(infoset)
            
        for action in possible_actions:
            # 所谓regret就是在当前节点选择了某一个action后的收益期望 减去 在当前节点的收益期望 
            regret = util[action][player] - node_util[player]
            if self.debug:
                print("\t action: {} strategy:{} util:{} regret {} reachprob {}".format(action,strategy_dic[action],util[action][player],regret,str(reach_probs)))
            
            weighted_regret = reach_prob_oppo * regret
            self.strategy_profile.update_regret(infoset,action,weighted_regret)
            
        return node_util
        
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
    