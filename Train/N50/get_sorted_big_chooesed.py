import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from hdqn_mdp import *
from my_env import *
from load import *
from hdqn import *
from my_env import *
#已知degree非零图选中的rna（chooesed）
#求整个大图选中rna的id
#需要知道degree非零图与整个大图对应关系
from load import *
import load
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch

agent = hDQN(batch_size1=128)
print("初始化完毕！")
agent.meta_controller = torch.load("meta-all_"+str(100)+".pt",map_location=torch.device("cuda:4"))
agent.controller = torch.load("controller-"+str(100)+".pt",map_location=torch.device("cuda:4"))
state = -torch.ones((1,3164)).to("cuda:4")
goal_p =agent.meta_controller.forward(state)
action_p = []
for goal in range(agent.num_goal):
    sub_state_t = -torch.ones((1,agent.num_action,1)).to("cuda:4")

    action_goal_p = agent.controller.forward(sub_state_t, agent.adjs[goal])
    action_p.append(action_goal_p)

sub_station = torch.load("../../../Sample/N50/degree0_2_subgraph_50_6_.pt")
res = {}
for i in range(agent.num_goal):
    for j in range(agent.num_action):
        bignode = sub_station[i][j]
        if bignode not in res.keys():
            count = 0
            res[bignode] = [goal_p[0, i] * action_p[i][0, j, 0], bignode, count]
        else:
            count = res[bignode][2] + 1
            res[bignode][0] = (res[bignode][0] + goal_p[0, i] * action_p[i][0, j, 0])
            res[bignode][2] = count
for key in res.keys():
    res[key][0] = res[key][0] / res[key][2]
torch.save(res, "N50_old_6_allscores.pt")
res_list = [value for key, value in res.items()]
res_sort = sorted(res_list, key=lambda x: x[0], reverse=True)
print(res_sort[0])
big_sort_chooesed = [node for _, node, _ in res_sort]
print(len(big_sort_chooesed))
degree_nodes = torch.load("../../../Sample/N50/graphid_2_degree0_50_6_add_ep.pt")
res_last = []
for node in big_sort_chooesed:
    res_last.append(int(degree_nodes[node, 0]))
torch.save(res_last, "N50_6_100.pt")



