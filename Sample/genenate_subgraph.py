from load import *
import load
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch

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

rna_num = {}
num_rna = {}
curnode = 0
for name in g.ntypes:
    tmp={}
    for i in range(g.number_of_nodes(name)):
        tmp[i]=curnode
        num_rna[curnode]=(name,i)
        curnode+=1
    rna_num[name]=tmp

#去掉真实网络中去点度为0的孤立节点
new_nodes = [_ for _, v in nx.degree(gragh) if v != 0]
# plt.subplot(211)
graph_new = gragh.subgraph(new_nodes)

#对联通子图进行判断
N = 50
# N = 100
# N = 150
# N = 150
# N = 250
# N = 300
# N = 400

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
idexs = [2,3,4,5,6]
lenths = [338,342,342,337,335]
for idex,lenth in zip(idexs,lenths):
    print(idex,lenth)
    degree0_nodes = list(graph_new.nodes())
    lncnum = 0
    minum = 0
    mnum = 0
    for n in degree0_nodes:
        if num_rna[n][0] == "lncrna":
            lncnum += 1
        elif num_rna[n][0] == "mirna":
            minum += 1
        elif num_rna[n][0] == "mrna":
            mnum += 1
    # print(lncnum, minum, mnum)
    # print(len(graph_new))
    nodesss = list(graph_new.nodes())
    graphid_2_degree0 = torch.zeros((len(nodesss), 1))
    for i in range(len(nodesss)):
        graphid_2_degree0[i, 0] = torch.tensor(nodesss[i])
    torch.save(graphid_2_degree0, "./N"+str(N)+"/graphid_2_degree0_" + str(N) + "_" + str(idex) + "_add_ep" + ".pt")

    aaa = torch.load("./N"+str(N)+"/graphid_2_degree0_" + str(N) + "_" + str(idex) + "_add_ep" + ".pt")
    # print(len(graph_new.nodes()))
    # 获取新图的节点
    new_sort_nodes = list(graph_new.nodes())
    node_dict = {}
    node_dict_reverse = {}
    # 记录新节点在邻接矩阵中的index下标
    for i, n in enumerate(new_sort_nodes):
        node_dict[n] = i
        node_dict_reverse[i] = n
    new_A = np.array(nx.adjacency_matrix(graph_new).todense())
    # 改变邻接矩阵
    # print(len(new_A))

    # 读取各个子图，测试RGCN
    subgraphs = pd.read_csv("../data/SubGraph/N"+str(N)+"/N"+str(N)+"_"+str(idex)+"_"+str(lenth)+".csv")

    all_subgraphs = subgraphs.iloc[0:lenth, :]
    rna_num = {}
    num_rna = {}
    curnode = 0
    for name in g.ntypes:
        tmp = {}
        for i in range(g.number_of_nodes(name)):
            tmp[i] = curnode
            num_rna[curnode] = (name, i)
            curnode += 1
        rna_num[name] = tmp

    degree0_2_subgraph = {}
    path = "N" + str(N) + "/" + str(idex)
    for k in range(len(all_subgraphs)):
        lnc_lnc = []
        lnc_mi = []
        lnc_m = []
        mi_lnc = []
        mi_mi = []
        mi_m = []
        m_lnc = []
        m_mi = []
        m_m = []
        sub1 = list(all_subgraphs.iloc[k, :])
        rna_num = [0, 0, 0]
        node_type_id = {}
        lnc_num = 0
        mi_num = 0
        m_num = 0

        degree0_2_subgraph[k] = {}
        cursor = 0
        for f, node in enumerate(sub1):
            type = num_rna[node][0]
            degree0_2_subgraph[k][cursor] = node_dict[node]

            if type == "lncrna":
                node_type_id[node] = (type, lnc_num)
                lnc_num += 1
            elif type == "mirna":
                node_type_id[node] = (type, mi_num)
                mi_num += 1
            elif type == "mrna":
                node_type_id[node] = (type, m_num)
                m_num += 1
            cursor += 1
        # print(lnc_num, mi_num, m_num, end=" ")
        for index1, node in enumerate(sub1):
            if num_rna[node][0] == "lncrna":
                # 判断邻居属性
                for index, node1 in enumerate(sub1):
                    # 子图内的邻居节点node,i
                    if new_A[node_dict[node], node_dict[node1]] == 1:
                        if num_rna[node1][0] == "lncrna":
                            lnc_lnc.append([node_type_id[node][1], node_type_id[node1][1]])
                        elif num_rna[node1][0] == "mirna":
                            lnc_mi.append([node_type_id[node][1], node_type_id[node1][1]])
                        elif num_rna[node1][0] == "mrna":
                            lnc_m.append([node_type_id[node][1], node_type_id[node1][1]])
            elif num_rna[node][0] == "mirna":
                for index, node1 in enumerate(sub1):
                    # 子图内的邻居节点node,i
                    if new_A[node_dict[node], node_dict[node1]] == 1:
                        if num_rna[node1][0] == "lncrna":
                            mi_lnc.append([node_type_id[node][1], node_type_id[node1][1]])
                        elif num_rna[node1][0] == "mirna":
                            mi_mi.append([node_type_id[node][1], node_type_id[node1][1]])
                        elif num_rna[node1][0] == "mrna":
                            mi_m.append([node_type_id[node][1], node_type_id[node1][1]])
            elif num_rna[node][0] == "mrna":
                for index, node1 in enumerate(sub1):
                    # 子图内的邻居节点node,i
                    if new_A[node_dict[node], node_dict[node1]] == 1:
                        if num_rna[node1][0] == "lncrna":
                            m_lnc.append([node_type_id[node][1], node_type_id[node1][1]])
                        elif num_rna[node1][0] == "mirna":
                            m_mi.append([node_type_id[node][1], node_type_id[node1][1]])
                        elif num_rna[node1][0] == "mrna":
                            m_m.append([node_type_id[node][1], node_type_id[node1][1]])
        # print(k, len(sub1), len(lnc_lnc), len(lnc_mi), len(lnc_m), len(mi_lnc), len(mi_mi), len(mi_m), len(m_lnc),
        #       len(m_mi), len(m_m))
        try:
            data_lnc_lnc = pd.DataFrame(lnc_lnc).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_lnc_lnc.csv",
                                                        index=False)
            data_lnc_mi = pd.DataFrame(lnc_mi).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_lnc_mi.csv",
                                                      index=False)
            data_lnc_m = pd.DataFrame(lnc_m).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_lnc_m.csv",
                                                    index=False)
            data_mi_lnc = pd.DataFrame(mi_lnc).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_mi_lnc.csv",
                                                      index=False)
            data_mi_mi = pd.DataFrame(mi_mi).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_mi_mi.csv",
                                                    index=False)
            data_mi_m = pd.DataFrame(mi_m).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_mi_m.csv",
                                                  index=False)
            data_m_lnc = pd.DataFrame(m_lnc).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_m_lnc.csv",
                                                    index=False)
            data_m_mi = pd.DataFrame(m_mi).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_m_mi.csv",
                                                  index=False)
            data_m_m = pd.DataFrame(m_m).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_m_m.csv", index=False)
        except OSError:
            os.mkdir("../data/SubGraph/" + path + "/" + str(k) + "/")
            data_lnc_lnc = pd.DataFrame(lnc_lnc).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_lnc_lnc.csv",
                                                        index=False)
            data_lnc_mi = pd.DataFrame(lnc_mi).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_lnc_mi.csv",
                                                      index=False)
            data_lnc_m = pd.DataFrame(lnc_m).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_lnc_m.csv",
                                                    index=False)
            data_mi_lnc = pd.DataFrame(mi_lnc).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_mi_lnc.csv",
                                                      index=False)
            data_mi_mi = pd.DataFrame(mi_mi).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_mi_mi.csv",
                                                    index=False)
            data_mi_m = pd.DataFrame(mi_m).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_mi_m.csv",
                                                  index=False)
            data_m_lnc = pd.DataFrame(m_lnc).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_m_lnc.csv",
                                                    index=False)
            data_m_mi = pd.DataFrame(m_mi).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_m_mi.csv",
                                                  index=False)
            data_m_m = pd.DataFrame(m_m).to_csv("../data/SubGraph/" + path + "/" + str(k) + "/sub_m_m.csv", index=False)
    torch.save(degree0_2_subgraph, "./N"+str(N)+"/degree0_2_subgraph_" + str(N) + "_" + str(idex) + "_" + ".pt")
    aaa = torch.load("./N"+str(N)+"/degree0_2_subgraph_" + str(N) + "_" + str(idex) + "_" + ".pt")
    a = set([])
    state = torch.zeros((3164, 1))
    for i in range(lenth):
        for j in range(N):
            state[aaa[i][j]] = 1
            a.add(aaa[i][j])
    print(torch.sum(state))
    print(len(a))

