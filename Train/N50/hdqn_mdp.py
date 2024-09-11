import numpy as np
import random
from collections import namedtuple

from dgl import DGLError
from torch.optim.lr_scheduler import StepLR
from my_env import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from GAT import *
from replay_buffer import *
# from controller_replay_buffer import *
import random
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from controller_replay_buffer import *
#选择子网的外部网络Q（输入是大网状态（3164），输出是N子网数量（156））
class MetaController(nn.Module):
    def __init__(self, in_features=3164, out_features=596):
        """
        Initialize a Meta-Controller of Hierarchical DQN network for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(in_features, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, out_features)

    def forward(self, x):
        x = F.relu(F.dropout(self.fc1(x),0.1, training=self.training))
        x = F.relu(F.dropout(self.fc2(x),0.1, training=self.training))
        x = F.relu(F.dropout(self.fc3(x),0.1, training=self.training))

        return F.softmax(x,dim=1)
#子网中选择动作a（输入是子网状态N，输出是选择子网的节点a）


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class hDQN():
    """
    The Hierarchical-DQN Agent
    Parameters
    ----------
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        num_goal: int
            The number of goal that agent can choose from
        num_action: int
            The number of action that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
    """
    def __init__(self,
                 num_goal=596,
                 num_action=50,
                 batch_size1=128,
                 batch_size2=32,
                 ):
        ###############
        # BUILD MODEL #
        ###############
        self.num_goal = num_goal
        self.num_action = num_action
        self.node_allnum = 3164
        self.batch_size1 = batch_size1
        self.batch_size2 = batch_size2


        # 构建 meta-controller 和 controller

        self.meta_controller = MetaController().to("cuda:3")
        self.target_meta_controller = MetaController().to("cuda:3")
        self.controller = GAT(n_feat=1,n_hid=32,n_class=1,dropout=0.1,alpha=1,n_heads=3).to("cuda:3")
        self.target_controller = GAT(n_feat=1,n_hid=32,n_class=1,dropout=0.1,alpha=1,n_heads=3).to("cuda:3")

        # self.controller.load_state_dict(torch.load("sub_1_9999controllerpt.pt"))
        # self.target_controller.load_state_dict(torch.load("sub_1_9999target_controllerpt.pt"))
        # 定义优化器
        self.meta_optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=0.0001)
        self.meta_scheduler = StepLR(self.meta_optimizer, step_size=5000,gamma=1)
        self.ctrl_optimizer = torch.optim.Adam(self.controller.parameters(), lr=0.0001)
        self.controller_scheduler = StepLR(self.ctrl_optimizer, step_size=5000,gamma=1)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayBuffer(100000, self.node_allnum)
        self.ctrl_replay_memory = ControllerReplayBuffer(100000, self.num_action)
        self.meta_learn_step_counter = 0
        self.controller_learn_step_counter = 0
        self.maxiter = 1000
        self.meta_loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.controller_loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.gamma = 0.99

        #读取非零图
        # self.degree0_graph = load_degree0_graph()
        # self.degree0_adj = torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(self.degree0_graph).to_networkx().to_undirected()).todense(),dtype=float)).to("cuda:3")
        # self.degree0_graph= self.degree0_graph.to("cuda:3")

        # self.rna_num = {}
        # self.num_rna = {}
        # curnode = 0
        # for name in self.degree0_graph.ntypes:
        #     tmp = {}
        #     for i in range(self.degree0_graph.number_of_nodes(name)):
        #         tmp[i] = curnode
        #         self.num_rna[curnode] = (name, i)
        #         curnode += 1
        #     self.rna_num[name] = tmp
        # 需要记录第几个子图的第几个节点，表示大图中的哪个节点
        self.station_sub_action = torch.load("degree0_2_subgraph_50.pt")
        #读取所有子图
        # self.subs = []
        self.adjs = []
        for i in range(num_goal):
            g = load_subk_GAT_graph(i)
            adj = torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(g).to_networkx().to_undirected()).todense(),dtype=float)).to("cuda:3")
            # g = g.to("cuda:3")
            # self.subs.append(g)
            self.adjs.append(adj)
        self.adjs = torch.stack(self.adjs).to("cuda:3")

    #选择
    def select_goal(self, state, epilson,subgraph_control,test=False):

        if test:
            # print(test)
            # goal_index = [i for i, s in enumerate(subgraph_control) if s == 1]
            goal_index = torch.nonzero(subgraph_control==1)[:,0].tolist()
            goals_value = self.meta_controller.forward(state)
            # print(actions_value.shape)
            goals_value[0, goal_index] = float("-inf")
            goal = torch.argmax(goals_value).item()
            return goal
        # print("***")
        sample = random.random()
        goal_index = torch.nonzero(subgraph_control==1)[:,0].tolist()
        if sample < epilson:
            goals_value = self.meta_controller.forward(state)
            # print(actions_value.shape)
            goals_value[0,goal_index] = float("-inf")
            goal = torch.argmax(goals_value).item()

            return goal
        else:
            # goal = int(torch.IntTensor([random.randrange(self.num_goal)]))
            chices = torch.nonzero(subgraph_control==0)[:,0].tolist()

            return random.choice(chices)


    def select_action(self, state,goal, epilson,test=False):
        if test:
            # action_index = [i for i, s in enumerate(state[0, :, 0]) if s == 1]
            action_index = torch.nonzero(state==1)[:,1].tolist()
            actions_value = self.controller.forward(state, self.adjs[goal])

            actions_value[0, action_index, 0] = float("-inf")
            action = torch.argmax(actions_value).item()
            return action
        # print("***")
        sample = random.random()
        if sample < epilson:
            action_index = torch.nonzero(state==1)[:,1].tolist()
            actions_value = self.controller.forward(state, self.adjs[goal])
            # print(actions_value.shape)
            actions_value[0, action_index, 0] = float("-inf")
            action = torch.argmax(actions_value).item()
            return action
        else:
            # 从未被选择的action中选择action
            chices = torch.nonzero(state!=1)[:,1].tolist()

            return random.choice(chices)



    def update_meta_controller(self):
        if self.meta_learn_step_counter % self.maxiter == 0:
            self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        self.meta_learn_step_counter += 1

        # sample batch transitions
        #mini batch更新eval网络参数
        #每一步都更新eval参数，一定时间赋值给target
        b_s,b_a,b_r,b_s_,b_done = self.meta_replay_memory.sample(self.batch_size1)
        # print(b_s.shape,b_a.shape,b_r.shape,b_s_.shape,b_done)


        q_eval = self.meta_controller.forward(b_s).gather(1, b_a.long().unsqueeze(1))

        q_next = self.target_meta_controller.forward(b_s_).detach().max(dim=1)[0]


        for batch in range(self.batch_size1):# 注意！这里要判断一下是否是done，如果结束的话q-target只有reward部分，不加next状态的q值

            if b_done[batch,0]==1:
                q_next[batch] = torch.tensor(0).to("cuda:3")
        q_target=0.99 * q_next + torch.tensor(b_r).clone().detach().to("cuda:3") #=当初获得的奖励+下一步的价值（max（1）返回的是【最大值，索引】，所以取出下标0的最大值

        loss=self.meta_loss_func(q_eval, q_target)
        loss.backward()
        self.meta_optimizer.step()
        self.meta_scheduler.step()
        self.meta_optimizer.zero_grad()

    #只能batch=1更新
    def update_controller(self):
        if self.controller_learn_step_counter % self.maxiter == 0:
            self.target_controller.load_state_dict(self.controller.state_dict())
        self.controller_learn_step_counter += 1

        # sample batch transitions
        #mini batch更新eval网络参数
        #每一步都更新eval参数，一定时间赋值给target
        # b_s,b_a,b_g,b_r,b_s_,done = self.ctrl_replay_memory.sample(self.batch_size)
        b_s,b_a,b_g,b_r,b_s_,done = self.ctrl_replay_memory.sample(self.batch_size2)
        # print(b_s)
        #
        # print(self.controller.forward(b_s,self.adj))

        q_eval = self.controller.forward(b_s,self.adjs[b_g.long()]).squeeze(2).gather(1, b_a.long().unsqueeze(1)).squeeze(1)
        # print(q_eval,"***",q_eval.shape)
        q_next=self.target_controller.forward(b_s_,self.adjs[b_g.long()]).squeeze(2).detach().max(dim=1)[0]
        # print(q_next,"******",q_next.shape)

        # for one_done,batch in zip(done,range(self.batch_size)):# 注意！这里要判断一下是否是done，如果结束的话q-target只有reward部分，不加next状态的q值
        for one_done,batch in zip(done,range(1)):# 注意！这里要判断一下是否是done，如果结束的话q-target只有reward部分，不加next状态的q值
            if one_done:
                q_next[batch] = torch.tensor(0).to("cuda:3")

        q_target=self.gamma* q_next + b_r.to("cuda:3") #=当初获得的奖励+下一步的价值（max（1）返回的是【最大值，索引】，所以取出下标0的最大值
        # print(q_eval,q_target)
        loss=self.controller_loss_func(q_eval, q_target)
        # print("loss",loss)
        loss.backward()
        self.ctrl_optimizer.step()
        self.controller_scheduler.step()
        self.ctrl_optimizer.zero_grad()

