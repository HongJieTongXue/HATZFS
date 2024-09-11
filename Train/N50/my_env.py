import random
import torch
from load import *
import networkx as nx
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class StochasticMDPEnv:

    def __init__(self):
        # 度不为0图节点数
        self.degree0_num = 3164
        # 子图数量
        self.sub_num = 596
        # 子图节点数
        self.sub_node_num = 50

        #表示有哪些节点被选中1，哪些没有-1
        self.current_state = -torch.ones((1,self.degree0_num)).to("cuda:3")
        self.sub_node_state = -torch.ones((self.sub_num, self.sub_node_num)).to("cuda:3")

        #需要记录第几个子图的第几个节点，表示大图中的哪个节点
        #***********************************************
        #这个转为tensor
        #***********************************************
        # self.station_sub_action = torch.tensor(torch.load("../../../Sample/degree0_2_subgraph_300_2_.pt")).to("cuda:3")
        self.station_sub_action = torch.load("degree0_2_subgraph_50.pt")
        self.sub_indexs = [list(j.values()) for i, j in self.station_sub_action.items()]
        self.sub_indexs = torch.tensor(self.sub_indexs).to("cuda:3")

        #***********************************************
        #这个转为tensor
        #***********************************************
        #记录大图的一个节点包含了哪些子图节点
        self.degree0_node_subnode = {}
        for i in range(self.sub_num):
            for j in range(self.sub_node_num):
                degree0_node = self.station_sub_action[i][j]
                if degree0_node not in self.degree0_node_subnode.keys():
                    self.degree0_node_subnode[degree0_node] = [(i,j)]
                else:
                    self.degree0_node_subnode[degree0_node].append((i,j))


        #需要存放所有子图的邻接矩阵
        # self.adjs = []
        #记录子图是否可控
        self.subgraph_control = [0]*self.sub_num
        #加载子图所有节点邻居层
        self.all_neibor = torch.load("all_neibor_N50.pt")

        self.adjs = []
        for i in range(self.sub_num):
            g = load_subk_GAT_graph(i)
            adj = torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(g).to_networkx().to_undirected()).todense(),dtype=float)).to("cuda:3")

            self.adjs.append(adj)
        self.adjs = torch.stack(self.adjs).to("cuda:3")

        degree0_graph = load_degree0_graph()
        self.adj_big_graph = torch.from_numpy(
            np.array(nx.adjacency_matrix(dgl.to_homogeneous(degree0_graph).to_networkx().to_undirected()).todense(),
                     dtype=float)).to("cuda:3")



    def reset(self):
        # 整个大图被选中情况
        self.current_state = -torch.ones((1,self.degree0_num)).to("cuda:3")
        # self.subgraph_control = [0]*self.sub_num
        self.subgraph_control = torch.zeros(self.sub_num).to("cuda:3")
        # 不仅需要返回整个大图被选中情况，还有各个子图被选中情况
        self.sub_node_state = -torch.ones((self.sub_num,self.sub_node_num)).to("cuda:3")

        return self.current_state

    def get_substate(self,k):
        return self.sub_node_state[k,:]

    def meta_isdone(self):
        if  torch.sum(self.current_state)== self.degree0_num:
            return 1
        else:
            return 0

    def controller_isdone(self,goal):
        if torch.sum(self.sub_node_state[goal,:]) == self.sub_node_num:
            return True
        else:
            return False

    def step(self, sub,action):
        # print(sub,action)
        # 首先更新节点状态
        if self.sub_node_state[sub, action] != 1:
            self.current_state[0,self.station_sub_action[sub][action]] = 1
            self.sub_node_state[sub, action] = 1
            all_r = 1
            # print(torch.sum(self.current_state))
            # r, isdone = self.ZFS5(sub)
            # # print(torch.sum(self.current_state))
            # if torch.sum(self.sub_node_state[sub, :]) == self.sub_node_num:
            #     self.subgraph_control[sub] = 1
            #     isdone = 1
            # all_r += r
            
            for su,a in self.degree0_node_subnode[self.station_sub_action[sub][action]]:
                self.sub_node_state[su, a] = 1
                self.current_state[0,self.station_sub_action[su][a]] = 1
                # all_r += 1
                # 根据ZFS更新节点状态
                # action是子图中的第action个动作，所以是需要记录第几个子图
                # 输入是sub，action
                r, isdone = self.ZFS5(su)
                if torch.sum(self.sub_node_state[su,:]) ==self.sub_node_num:
                    self.subgraph_control[su] = 1
                    if su == sub:
                        isdone = 1
                all_r+=r
        else:

            all_r = 0
            isdone = 0

        # return self.sub_node_state[sub,:],all_r,isdone,indexss
        return self.sub_node_state[sub,:],all_r,isdone

    def get_intersect(self,set1,set2):
        # try:
        if len(set1)==0 or len(set2)==0:
            return torch.tensor([])
        # intersection = set1[(set1.view(1, -1) == set2.view(-1, 1)).any(dim=0)]
        a_cat_b, counts = torch.cat([set1, set2]).unique(return_counts=True)
        intersection = a_cat_b[torch.where(counts.gt(1))]
        # except:
        #     print(set1.shape)
        #     print(set2.shape,"***")
        return intersection

    #邻居层可以被扩散，但不能扩散别的节点
    def ZFS5(self,sub):


        actions = torch.nonzero(self.sub_node_state[sub, :] == 1)
        # 黑色节点
        blacks = actions[:, 0].tolist()

        len1 = len(blacks)
        len2 = -1
        reward = 0

        reward1 = torch.sum(self.current_state)
        while len1!= len2:
            len1 = len2
            # print(len1,len2)
            # 获取黑色节点在原图的index，
            black_indexs = self.sub_indexs[sub, blacks].tolist()
            # 找这些黑色节点的邻居节点
            blacks_neibor = torch.nonzero(self.adj_big_graph[black_indexs, :])[:, 1]

            blacks_neibor_count = torch.count_nonzero(self.adj_big_graph[black_indexs, :], dim=1).tolist()
            # 找原图的白色节点
            # 找黑色节点的白色邻居数量
            blacks_split_neibor = torch.split(blacks_neibor, blacks_neibor_count)

            # 取交集，判断有没有只有一个白色邻居的黑色节点
            black_one_white_neibor = []
            big_whites = torch.nonzero(self.current_state == -1)[:, 1]
            for black_index, white_nodes in enumerate(blacks_split_neibor):
                inc = self.get_intersect(white_nodes, big_whites)
                if len(inc) == 1:
                    black_one_white_neibor.append([black_indexs[black_index], int(inc)])
            # print(len(black_one_white_neibor),end=" ")
            #如果有黑色节点只有一个白色邻居的话，就需要把那些白色邻居都变为黑色
            if len(black_one_white_neibor) > 0:
                for black_big_index, white_big_index in black_one_white_neibor:
                    self.current_state[0, white_big_index] = 1
                    for s, a in self.degree0_node_subnode[white_big_index]:
                        self.sub_node_state[s, a] = 1
                        if torch.sum(self.sub_node_state[s, :]) == self.sub_node_num:
                            self.subgraph_control[s] = 1


            else:
                return 0, 0

            actions = torch.nonzero(self.sub_node_state[sub, :] == 1)
            # 黑色节点
            blacks = actions[:, 0].tolist()

            len2 = len(blacks)


        is_done = 0
        reward2 = torch.sum(self.current_state)
        if len2 == self.sub_node_num:
            is_done = 1
        reward = int(reward2 - reward1)
        return reward, is_done


