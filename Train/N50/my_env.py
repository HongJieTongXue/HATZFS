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
        self.sub_num = 335
        # 子图节点数
        self.sub_node_num = 50

        #表示有哪些节点被选中1，哪些没有-1
        self.current_state = -torch.ones((1,self.degree0_num)).to("cuda:4")
        self.sub_node_state = -torch.ones((self.sub_num, self.sub_node_num)).to("cuda:4")
        #需要记录第几个子图的第几个节点，表示大图中的哪个节点
        self.station_sub_action = torch.load("../../../Sample/N50/degree0_2_subgraph_50_6_.pt")
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
        self.adjs = []
        #记录子图是否可控
        self.subgraph_control = [0]*self.sub_num
        #加载子图所有节点邻居层
        self.all_neibor = torch.load("all_neibor_N50-6.pt")
        # self.all_neibor = torch.load("all_neibor_N100.pt")


        for k in range(self.sub_num):
            # print(k)
            g = load_subk_hetero_graph(k)
            adjacency_matrix = torch.from_numpy(
                np.array(nx.adjacency_matrix(dgl.to_homogeneous(g).to_networkx().to_undirected()).todense(),
                         dtype=float))
            self.adjs.append(adjacency_matrix)


    def reset(self):
        # 整个大图被选中情况
        self.current_state = -torch.ones((1,self.degree0_num)).to("cuda:4")
        self.subgraph_control = [0]*self.sub_num
        # 不仅需要返回整个大图被选中情况，还有各个子图被选中情况
        self.sub_node_state = -torch.ones((self.sub_num,self.sub_node_num)).to("cuda:4")

        return self.current_state

    def get_substate(self,k):
        return self.sub_node_state[k,:]
    def meta_isdone(self):
        # print(torch.sum(self.current_state))
        if  torch.sum(self.current_state)== self.degree0_num:
            return 1
        else:
            return 0
    def controller_isdone(self,goal):
        # print(torch.sum(self.sub_node_state[goal,:]))
        if torch.sum(self.sub_node_state[goal,:]) == self.sub_node_num:
            return True
        else:
            return False

    def step(self, sub,action):
        # 首先更新节点状态
        if self.sub_node_state[sub, action] != 1:
            self.current_state[0,self.station_sub_action[sub][action]] = 1
            self.sub_node_state[sub, action] = 1
            all_r = 0
            # indexss = []
            # 不仅需要改变该子图对应节点变为1，还要把所有其他子图中该RNA变为1
            for sub,a in self.degree0_node_subnode[self.station_sub_action[sub][action]]:
                self.sub_node_state[sub, a] = 1
                self.current_state[0,self.station_sub_action[sub][a]] = 1
                all_r += 1
                # 根据ZFS更新节点状态
                # action是子图中的第action个动作，所以是需要记录第几个子图
                # 输入是sub，action
                r, isdone = self.ZFS5(sub)
                # indexss.extend(indexs)
                if torch.sum(self.sub_node_state[sub,:]) ==self.sub_node_num:
                    self.subgraph_control[sub] = 1
                    isdone = 1
                all_r+=r
        else:
            reward = 0
            all_r = 0
            isdone = 0
            indexss = []


        # return self.sub_node_state[sub,:],all_r,isdone,indexss
        return self.sub_node_state[sub,:],all_r,isdone
    #邻居层可以被扩散，但不能扩散别的节点
    def ZFS5(self,sub):
        actions = [i for i,s in enumerate(self.sub_node_state[sub,:]) if s==1]
        len1 = len(actions)
        len2 = -1
        reward = 0
        reward1 = torch.sum(self.current_state)
        while len1 != len2:
            len1 = len(actions)
            # print("ZFS LEN1:",len1,end=" ")
            for a in actions:
                a = int(a)
                #判断该节点a是否可以传播
                if torch.sum(self.current_state[0,self.all_neibor[sub][a]]) == (len(self.all_neibor[sub][a])-2):
                    #可以传播
                    #判断该节点是否在子图内，子图内可传播，子图外不可
                    index = torch.nonzero(self.current_state[0, self.all_neibor[sub][a]] == -1)
                    node = self.all_neibor[sub][a][index]
                    node = int(node)

                    reward += 1
                    #如果在子图内
                    if node in list(self.station_sub_action[0].values()):
                        for sub_new, a_new in self.degree0_node_subnode[node]:
                            if sub_new == sub:
                                actions.append(a_new)
                                break
                    # print(node,self.current_state.shape,end = " ")
                    # self.sub_node_state[sub, index] = 1
                    self.current_state[0, node] = 1
                    # 还需要把其他包含node的子图变为选中状态
                    for sub_new, a_new in self.degree0_node_subnode[node]:
                        self.sub_node_state[sub_new, a_new] = 1
                else:
                    continue
            len2 = len(actions)
            # print("ZFS LEN2:",len2)
        is_done = 0
        reward2 = torch.sum(self.current_state)
        if len2 == self.sub_node_num:
            is_done = 1
        reward = reward2-reward1
        return reward,is_done
    def ZFS4(self,sub):
        actions = [i for i,s in enumerate(self.sub_node_state[sub,:]) if s==1]
        len1 = len(actions)
        len2 = -1
        reward = 0
        while len1 != len2:
            len1 = len(actions)
            # print("ZFS LEN1:",len1,end=" ")
            for a in actions:
                a = int(a)
                #判断该节点a是否可以传播
                if torch.sum(self.current_state[0,self.all_neibor[sub][a]]) == (len(self.all_neibor[sub][a])-2):
                    #可以传播
                    #判断该节点是否在子图内，子图内可传播，子图外不可
                    index = torch.nonzero(self.current_state[0, self.all_neibor[sub][a]] == -1)
                    node = self.all_neibor[sub][a][index]
                    node = int(node)

                    reward += 1
                    #如果在子图内
                    if node in list(self.station_sub_action[0].values()):
                        for sub_new, a_new in self.degree0_node_subnode[node]:
                            if sub_new == sub:
                                actions.append(a_new)

                                #可被扩散节点如果在子图内就可以被扩散，邻居层则不行
                                self.current_state[0, node] = 1
                                # 还需要把其他包含node的子图变为选中状态
                                for sub_new, a_new in self.degree0_node_subnode[node]:
                                    self.sub_node_state[sub_new, a_new] = 1
                                    if torch.sum(self.sub_node_state[sub, :]) == self.sub_node_num:
                                        self.subgraph_control[sub] = 1
                                break
                    # print(node,self.current_state.shape,end = " ")
                    # self.sub_node_state[sub, index] = 1

                else:
                    continue
            len2 = len(actions)
            # print("ZFS LEN2:",len2)
        is_done = 0
        if len2 == self.sub_node_num:
            is_done = 1

        return reward,is_done

    #邻居层可以被扩散，但不能扩散别的节点
    def ZFS3(self,sub):
        actions = [i for i,s in enumerate(self.sub_node_state[sub,:]) if s==1]
        len1 = len(actions)
        len2 = -1
        reward = 0
        while len1 != len2:
            len1 = len(actions)
            # print("ZFS LEN1:",len1,end=" ")
            for a in actions:
                a = int(a)
                #判断该节点a是否可以传播
                if torch.sum(self.current_state[0,self.all_neibor[sub][a]]) == (len(self.all_neibor[sub][a])-2):
                    #可以传播
                    #判断该节点是否在子图内，子图内可传播，子图外不可
                    index = torch.nonzero(self.current_state[0, self.all_neibor[sub][a]] == -1)
                    node = self.all_neibor[sub][a][index]
                    node = int(node)

                    reward += 1
                    #如果在子图内
                    if node in list(self.station_sub_action[0].values()):
                        for sub_new, a_new in self.degree0_node_subnode[node]:
                            if sub_new == sub:
                                actions.append(a_new)
                                break
                    # print(node,self.current_state.shape,end = " ")
                    # self.sub_node_state[sub, index] = 1
                    self.current_state[0, node] = 1
                    # 还需要把其他包含node的子图变为选中状态
                    for sub_new, a_new in self.degree0_node_subnode[node]:
                        self.sub_node_state[sub_new, a_new] = 1
                else:
                    continue
            len2 = len(actions)
            # print("ZFS LEN2:",len2)
        is_done = 0
        if len2 == self.sub_node_num:
            is_done = 1

        return reward,is_done

    # 需要判断能扩散几个节点(考虑邻居层)
    def ZFS2(self,sub):
        actions = [i for i,s in enumerate(self.sub_node_state[sub,:]) if s==1]
        len1 = len(actions)
        len2 = -1
        reward = 0
        while len1 != len2:
            len1 = len(actions)
            # print("ZFS LEN1:",len1,end=" ")
            for a in actions:
                count = 0
                index = -1
                a = int(a)
                #判断该节点a是否可以传播
                if torch.sum(self.current_state[0,self.all_neibor[sub][a]]) == (len(self.all_neibor[sub][a])-2):
                    #可以传播
                    #判断该节点是否在子图内，子图内可传播，子图外不可
                    index = torch.nonzero(self.current_state[0, self.all_neibor[sub][a]] == -1)
                    node = self.all_neibor[sub][a][index]
                    #在子图内
                    if node in list(self.station_sub_action[0].values()):
                        reward += 1
                        actions.append(index)
                        self.sub_node_state[sub, index] = 1
                        self.current_state[0, node] = 1
                        node = int(node)
                        #还需要把其他包含node的子图变为选中状态
                        for sub_new, a_new in self.degree0_node_subnode[node]:
                            self.sub_node_state[sub_new, a_new] = 1
                        print("************")
                    else:
                        print("111")
                        continue
                else:
                    continue
            len2 = len(actions)
            # print("ZFS LEN2:",len2)
        is_done = 0
        if len2 == self.sub_node_num:
            is_done = 1

        return reward,is_done

    # 需要判断能扩散几个节点
    def ZFS(self,sub):
        actions = [i for i,s in enumerate(self.sub_node_state[sub,:]) if s==1]
        len1 = len(actions)
        len2 = -1
        reward = 0
        indexs = []
        while len1 != len2:
            len1 = len(actions)
            # print("ZFS LEN1:",len1,end=" ")
            for a in actions:
                count = 0
                index = -1
                for i in range(len(self.adjs[sub])):
                    if self.adjs[sub][a, i] and i not in actions:
                        count += 1
                        index = i
                        if count >= 2:
                            break
                if count == 1:
                    reward += 1
                    actions.append(index)
                    indexs.append(index)
                    self.sub_node_state[sub,index] = 1
                    self.current_state[0,self.station_sub_action[sub][index]] = 1
            len2 = len(actions)
            # print("ZFS LEN2:",len2)
        is_done = 0
        if len2 == self.sub_node_num:
            is_done = 1

        return reward,is_done,indexs
