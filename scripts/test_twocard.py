import sys
import os
sys.path.append('../')
import yaml
from utils.utils import ModelLoader,DepLoader
with open('../config/rule_two_card.yaml') as fhdl:
    cfg = yaml.load(fhdl)
    
env = DepLoader(cfg)

env.trainer.train()