import random
import copy
from load import *
import load
import networkx as nx
import matplotlib.pyplot as plt

g=load.load_random_hetero_graph('../data/PAAD_network', Specified_quantity={'lncrna': 1500, 'mirna': 168, 'mrna': 2519})
adjacency_matrix=torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(g).to_networkx().to_undirected()).todense(),dtype=float))
gragh = nx.Graph()
for i in range(len(adjacency_matrix)):
    gragh.add_node(i)
    gragh.nodes[i]["data"] = i

for i in range(len(adjacency_matrix)):
    for j in range(len(adjacency_matrix)):
        if adjacency_matrix[i,j]:
            gragh.add_edge(i,j)

#去掉真实网络中去点度为0的孤立节点
new_nodes = [_ for _, v in nx.degree(gragh) if v != 0]

# plt.subplot(211)
graph_new = gragh.subgraph(new_nodes)


# 写一个函数，输入为list[node,...,],输出为按照度排序的list，相同degree要随机打乱，每次生成结果不同
def get_degree_sorted_list(nodes):
    # 按度排序好
    nodes = sorted(nodes, key=lambda x: nodes_degree_dict[x])
    degree_sorted_nodes = [(node, nodes_degree_dict[node]) for node in nodes]
    # 相同度要随机打乱
    dict_degree = {}
    # 获取各个degree的list集合
    for node, degree in degree_sorted_nodes:
        if degree not in dict_degree.keys():
            dict_degree[degree] = [node]
        else:
            dict_degree[degree].append(node)
    # 将各个list随机打乱，然后按度拼接在一起
    for degree, node_list in dict_degree.items():
        random.shuffle(dict_degree[degree])
    # dict_degree_key = sorted(dict_degree.keys(),reverse=True)
    dict_degree_key = sorted(dict_degree.keys())
    sorted_nodes = []
    for key in dict_degree_key:
        sorted_nodes.extend(dict_degree[key])
    return sorted_nodes


N = 50
idexs = [2,3,4,5,6]
for idex in idexs:
    print(idex)
    CS = []
    CS_len = []
    init_graph = []
    for c in nx.connected_components(graph_new):
        # 得到不连通的子集
        nodeSet = list(graph_new.subgraph(c).nodes())
        CS.append(c)
        CS_len.append(len(nodeSet))
        if len(nodeSet) <N:
            init_graph.append(nodeSet)
    # print(CS_len)
    graph_new = graph_new.subgraph(CS[0])

    #获取新图的节点
    new_sort_nodes = list(graph_new.nodes())
    node_dict = {}
    node_dict_reverse = {}
    #记录新节点在邻接矩阵中的index下标
    for i,n in enumerate(new_sort_nodes):
        node_dict[n] = i
        node_dict_reverse[i] = n
    new_A=np.array(nx.adjacency_matrix(graph_new).todense())

    #首先需要获取所有节点的邻居集合
    node_neigh = {}
    for i in range(len(new_A)):
        for j in range(len(new_A)):
            if new_A[i,j]:
                if node_dict_reverse[i] not in node_neigh.keys():
                    node_neigh[node_dict_reverse[i]] = [node_dict_reverse[j]]
                else:
                    node_neigh[node_dict_reverse[i]].append(node_dict_reverse[j])
    # print(node_neigh)


    # 需要对把新节点按度排序
    degree_list = [(node, v) for node, v in nx.degree(graph_new)]
    nodes_degree_dict = {}
    for node, degree in degree_list:
        nodes_degree_dict[node] = degree
    # print(degree_list)
    # degree_sorted_nodes = sorted(degree_list,key=lambda x: x[1],reverse=True)
    degree_sorted_nodes = sorted(degree_list, key=lambda x: x[1])
    # print(degree_sorted_nodes)

    nodes = [node for node,_ in degree_list]
    sorted_nodes = get_degree_sorted_list(nodes)
    # 用K阶邻居采样
    # N表示每个子图大小
    # 按照度由大到小的顺序进行采样，直到满足终止条件
    # 按照度由小到大的顺序进行采样，直到满足终止条件
    # 采样终止条件是所有节点都被采样到
    nodes_if_choosed = [0] * len(sorted_nodes)
    all_subgraph = []
    cursor = 0
    while sum(nodes_if_choosed) != len(sorted_nodes):
        sub_graph = []
        # 从new_sort_nodes，随机产生一个节点，生成K阶邻居
        # node = sorted_nodes[cursor]
        node = degree_sorted_nodes[cursor][0]
        cursor += 1
        sub_graph.append(node)
        now_nodes = [node]
        neigh_nodes_k = []
        for n in now_nodes:
            neigh_nodes_k.extend(node_neigh[n])
        nodes_if_choosed[node_dict[node]] = 1
        # 开始添加node的K阶邻居，直到数量达到N
        while len(set(sub_graph)) != N:
            # 先判断现有节点+新K-阶邻居节点够不够达到N，不够直接全加，否则按度分布加
            nodes_can_add = [node for node in neigh_nodes_k]
            if (len(sub_graph) + len(set(nodes_can_add))) <= N:
                for n in nodes_can_add:
                    nodes_if_choosed[node_dict[n]] = 1
                    sub_graph.append(n)
                # sub_graph.extend(nodes_can_add)
            else:
                # 邻居节点按照度排序
                # new_sorted_nodes = get_degree_sorted_list(list(set(nodes_can_add)))
                random.shuffle(nodes_can_add)
                new_sorted_nodes = nodes_can_add.copy()
                # print(len(sub_graph))
                for n in new_sorted_nodes:
                    nodes_if_choosed[node_dict[n]] = 1
                    sub_graph.append(n)
                    if len(set(sub_graph)) == N:
                        break
            now_nodes = neigh_nodes_k.copy()
            neigh_nodes_k = []
            for n in now_nodes:
                neigh_nodes_k.extend(node_neigh[n])

        all_subgraph.append(list(set(sub_graph)))
        # print(list(set(sub_graph)))
    print(len(all_subgraph))
    nodes_nums = {}
    #子图数量可能会太多，删掉重复的
    for subgraph in all_subgraph[::-1]:
        for n in subgraph:
            if n not in nodes_nums.keys():
                nodes_nums[n] = 1
            else:
                nodes_nums[n] = nodes_nums[n] + 1
    if_del = []
    del_all_graph = copy.deepcopy(all_subgraph[::-1])
    not_del = 0
    for index, subgraph in enumerate(all_subgraph[::-1]):
        # print(index)
        # 遍历所有的子图，判断能不能删除，能删除就当场删除，并重新计算nodes_nums
        flag = 1
        for n in subgraph:
            # 不能删
            if nodes_nums[n] <= 1:
                flag = 0
                break
        if flag:
            del_all_graph.pop(not_del)
        else:
            not_del += 1
        # print(len(del_all_graph),'*')
        nodes_nums = {}
        # 子图数量可能会太多，删掉重复的
        for subgraph in del_all_graph:
            for n in subgraph:
                if n not in nodes_nums.keys():
                    nodes_nums[n] = 1
                else:
                    nodes_nums[n] = nodes_nums[n] + 1
    print(len(del_all_graph))
    import pandas as pd

    data = pd.DataFrame(del_all_graph)
    data.head()
    data.to_csv( "../data/SubGraph/N" + str(N) + "/N" + str(N) + "_" + str(idex) + "_" + str(len(del_all_graph)) + ".csv",index=False)


