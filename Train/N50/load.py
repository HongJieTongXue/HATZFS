import random
import dgl
import torch
import numpy as np
import argparse
import pandas as pd
import os
import networkx as nx
def load_degree0_graph(path='../../../data/Degree0_graph/'):
    graph_dict={}
    if os.path.getsize(path + '/lnc_lnc.csv')>=3:
        a = pd.read_csv(path+'/lnc_lnc.csv')
        graph_dict[('lncrna', 'lnc_lnc', 'lncrna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))
        # print((torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist())))

    if os.path.getsize(path + '/lnc_mi.csv')>=3:
        a = pd.read_csv(path + '/lnc_mi.csv')
        graph_dict[('lncrna', 'lnc_mi', 'mirna')]=(torch.tensor(a['0']),torch.tensor(a['1']))
        # print((torch.tensor(a['0'].tolist())))
    if os.path.getsize(path + '/lnc_m.csv')>=3:
        a = pd.read_csv(path + '/lnc_m.csv')
        graph_dict[('lncrna', 'lnc_m', 'mrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))
        # print((torch.tensor(a['0'].tolist())))
    if os.path.getsize(path+'/mi_lnc.csv')>=3:
        a=pd.read_csv(path+'/mi_lnc.csv')
        graph_dict[('mirna', 'mi_lnc', 'lncrna')]=(torch.tensor(a['0']), torch.tensor(a['1']))
        # print(torch.tensor(torch.tensor(a['1'].tolist())))
    if os.path.getsize(path + '/mi_mi.csv')>=3:
        a = pd.read_csv(path+'/mi_mi.csv')
        graph_dict[('mirna', 'mi_mi', 'mirna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))
    if os.path.getsize(path + '/mi_m.csv')>=3:
        a = pd.read_csv(path+'/mi_m.csv')
        graph_dict[('mirna', 'mi_m', 'mrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))

    if os.path.getsize(path + '/m_lnc.csv')>=3:
        a = pd.read_csv(path+'/m_lnc.csv')
        graph_dict[('mrna', 'm_lnc', 'lncrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))
        # print((torch.tensor(a['1'].tolist())))
    if os.path.getsize(path + '/m_mi.csv')>=3:
        a = pd.read_csv(path + '/m_mi.csv')
        graph_dict[('mrna', 'm_mi', 'mirna')] = ( torch.tensor(a['0']),torch.tensor(a['1']))
    if os.path.getsize(path + '/m_m.csv')>=3:
        a = pd.read_csv(path+'/m_m.csv')
        graph_dict[('mrna', 'm_m', 'mrna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))


    g = dgl.heterograph(graph_dict)
    adjacency_matrix = torch.from_numpy(
        np.array(nx.adjacency_matrix(dgl.to_homogeneous(g).to_networkx().to_undirected()).todense(), dtype=float))

    node1 = []
    node2 = []
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i, j]:
                node1.append(i)
                # node1.append(j)
                node2.append(j)
                # node2.append(i)
    gragh = dgl.graph((torch.tensor(node1), torch.tensor(node2)))

    # print(g)
    return gragh

#加载子图并转为GAT能用的形式
def load_subk_GAT_graph(k,path='../../../data/SubGraph/N50/1/'):
    graph_dict={}
    path = path + str(k)
    if os.path.getsize(path + '/sub_lnc_lnc.csv')>=3:
        a = pd.read_csv(path+'/sub_lnc_lnc.csv')
        graph_dict[('lncrna', 'lnc_lnc', 'lncrna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))
        # print((torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist())))

    if os.path.getsize(path + '/sub_lnc_mi.csv')>=3:
        a = pd.read_csv(path + '/sub_lnc_mi.csv')
        graph_dict[('lncrna', 'lnc_mi', 'mirna')]=(torch.tensor(a['0']),torch.tensor(a['1']))
        # print((torch.tensor(a['0'].tolist())))
    if os.path.getsize(path + '/sub_lnc_m.csv')>=3:
        a = pd.read_csv(path + '/sub_lnc_m.csv')
        graph_dict[('lncrna', 'lnc_m', 'mrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))
        # print((torch.tensor(a['0'].tolist())))
    if os.path.getsize(path+'/sub_mi_lnc.csv')>=3:
        a=pd.read_csv(path+'/sub_mi_lnc.csv')
        graph_dict[('mirna', 'mi_lnc', 'lncrna')]=(torch.tensor(a['0']), torch.tensor(a['1']))
        # print(torch.tensor(torch.tensor(a['1'].tolist())))
    if os.path.getsize(path + '/sub_mi_mi.csv')>=3:
        a = pd.read_csv(path+'/sub_mi_mi.csv')
        graph_dict[('mirna', 'mi_mi', 'mirna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))
    if os.path.getsize(path + '/sub_mi_m.csv')>=3:
        a = pd.read_csv(path+'/sub_mi_m.csv')
        graph_dict[('mirna', 'mi_m', 'mrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))

    if os.path.getsize(path + '/sub_m_lnc.csv')>=3:
        a = pd.read_csv(path+'/sub_m_lnc.csv')
        graph_dict[('mrna', 'm_lnc', 'lncrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))
        # print((torch.tensor(a['1'].tolist())))
    if os.path.getsize(path + '/sub_m_mi.csv')>=3:
        a = pd.read_csv(path + '/sub_m_mi.csv')
        graph_dict[('mrna', 'm_mi', 'mirna')] = ( torch.tensor(a['0']),torch.tensor(a['1']))
    if os.path.getsize(path + '/sub_m_m.csv')>=3:
        a = pd.read_csv(path+'/sub_m_m.csv')
        graph_dict[('mrna', 'm_m', 'mrna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))


    g = dgl.heterograph(graph_dict)
    adjacency_matrix = torch.from_numpy(
        np.array(nx.adjacency_matrix(dgl.to_homogeneous(g).to_networkx().to_undirected()).todense(), dtype=float))

    node1 = []
    node2 = []
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i, j]:
                node1.append(i)
                # node1.append(j)
                node2.append(j)
                # node2.append(i)
    gragh = dgl.graph((torch.tensor(node1), torch.tensor(node2)))

    # print(g)
    return gragh


def load_subk_hetero_graph(k,path='../../../data/SubGraph/N50/1/'):
    graph_dict={}
    path = path + str(k)
    if os.path.getsize(path + '/sub_lnc_lnc.csv')>=3:
        a = pd.read_csv(path+'/sub_lnc_lnc.csv')
        graph_dict[('lncrna', 'lnc_lnc', 'lncrna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))
        # print((torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist())))

    if os.path.getsize(path + '/sub_lnc_mi.csv')>=3:
        a = pd.read_csv(path + '/sub_lnc_mi.csv')
        graph_dict[('lncrna', 'lnc_mi', 'mirna')]=(torch.tensor(a['0']),torch.tensor(a['1']))
        # print((torch.tensor(a['0'].tolist())))
    if os.path.getsize(path + '/sub_lnc_m.csv')>=3:
        a = pd.read_csv(path + '/sub_lnc_m.csv')
        graph_dict[('lncrna', 'lnc_m', 'mrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))
        # print((torch.tensor(a['0'].tolist())))
    if os.path.getsize(path+'/sub_mi_lnc.csv')>=3:
        a=pd.read_csv(path+'/sub_mi_lnc.csv')
        graph_dict[('mirna', 'mi_lnc', 'lncrna')]=(torch.tensor(a['0']), torch.tensor(a['1']))
        # print(torch.tensor(torch.tensor(a['1'].tolist())))
    if os.path.getsize(path + '/sub_mi_mi.csv')>=3:
        a = pd.read_csv(path+'/sub_mi_mi.csv')
        graph_dict[('mirna', 'mi_mi', 'mirna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))
    if os.path.getsize(path + '/sub_mi_m.csv')>=3:
        a = pd.read_csv(path+'/sub_mi_m.csv')
        graph_dict[('mirna', 'mi_m', 'mrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))

    if os.path.getsize(path + '/sub_m_lnc.csv')>=3:
        a = pd.read_csv(path+'/sub_m_lnc.csv')
        graph_dict[('mrna', 'm_lnc', 'lncrna')] = (torch.tensor(a['0']), torch.tensor(a['1']))
        # print((torch.tensor(a['1'].tolist())))
    if os.path.getsize(path + '/sub_m_mi.csv')>=3:
        a = pd.read_csv(path + '/sub_m_mi.csv')
        graph_dict[('mrna', 'm_mi', 'mirna')] = ( torch.tensor(a['0']),torch.tensor(a['1']))
    if os.path.getsize(path + '/sub_m_m.csv')>=3:
        a = pd.read_csv(path+'/sub_m_m.csv')
        graph_dict[('mrna', 'm_m', 'mrna')] = (torch.tensor(a['0'].tolist()), torch.tensor(a['1'].tolist()))


    g = dgl.heterograph(graph_dict)


    # print(g)
    return g

def load_random_hetero_graph(path='../data/random_network',Specified_quantity={} ):
    graph_dict={}
    a=pd.read_csv(path+'/mi_lnc.csv')
    graph_dict[('mirna', 'mi_lnc', 'lncrna')]=(torch.tensor(a['mirna']), torch.tensor(a['lncrna']))
    graph_dict[('lncrna', 'lnc_mi', 'mirna')]=(torch.tensor(a['lncrna']),torch.tensor(a['mirna']))
    a = pd.read_csv(path+'/mi_m.csv')
    graph_dict[('mirna', 'mi_m', 'mrna')] = (torch.tensor(a['mirna']), torch.tensor(a['mrna']))
    graph_dict[('mrna', 'm_mi', 'mirna')] = ( torch.tensor(a['mrna']),torch.tensor(a['mirna']))
    a = pd.read_csv(path+'/m_lnc.csv')
    graph_dict[('lncrna', 'm_lnc', 'mrna')] = (torch.tensor(a['lncrna']), torch.tensor(a['mrna']))
    graph_dict[('mrna', 'lnc_m', 'lncrna')] = (torch.tensor(a['mrna']), torch.tensor(a['lncrna']))

    a = pd.read_csv(path+'/mi_mi.csv')
    graph_dict[('mirna', 'mi_mi', 'mirna')] = (torch.tensor(a['mirna1'].tolist()+a['mirna2'].tolist()), torch.tensor(a['mirna2'].tolist()+a['mirna1'].tolist()))
    a = pd.read_csv(path+'/m_m.csv')
    graph_dict[('mrna', 'm_m', 'mrna')] = (torch.tensor(a['mrna1'].tolist()+a['mrna2'].tolist()), torch.tensor(a['mrna2'].tolist()+a['mrna1'].tolist()))
    a = pd.read_csv(path+'/lnc_lnc.csv')
    graph_dict[('lncrna', 'lnc_lnc', 'lncrna')] = (torch.tensor(a['lncrna1'].tolist()+a['lncrna2'].tolist()), torch.tensor(a['lncrna2'].tolist()+a['lncrna1'].tolist()))

    if len(Specified_quantity)==0:
        g = dgl.heterograph(graph_dict)
    else:
        g = dgl.heterograph(graph_dict,num_nodes_dict=Specified_quantity)
    print(g)
    return g

def load_all_node(Specified_quantity={}):
    network_path = "./data/PAAD_network2"
    graph_dict={}
    a=pd.read_csv(network_path+'/mi_lnc.csv')
    graph_dict[('mirna', 'mi_lnc', 'lncrna')]=(torch.tensor(a['mirna']), torch.tensor(a['lncrna']))
    graph_dict[('lncrna', 'lnc_mi', 'mirna')]=(torch.tensor(a['lncrna']),torch.tensor(a['mirna']))
    a = pd.read_csv(network_path+'/mi_m.csv')
    graph_dict[('mirna', 'mi_m', 'mrna')] = (torch.tensor(a['mirna']), torch.tensor(a['mrna']))
    graph_dict[('mrna', 'm_mi', 'mirna')] = ( torch.tensor(a['mrna']),torch.tensor(a['mirna']))
    a = pd.read_csv(network_path+'/m_lnc.csv')
    graph_dict[('lncrna', 'm_lnc', 'mrna')] = (torch.tensor(a['lncrna']), torch.tensor(a['mrna']))
    graph_dict[('mrna', 'lnc_m', 'lncrna')] = (torch.tensor(a['mrna']), torch.tensor(a['lncrna']))

    a = pd.read_csv(network_path+'/mi_mi.csv')
    graph_dict[('mirna', 'mi_mi', 'mirna')] = (torch.tensor(a['mirna1'].tolist()+a['mirna2'].tolist()), torch.tensor(a['mirna2'].tolist()+a['mirna1'].tolist()))
    a = pd.read_csv(network_path+'/m_m.csv')
    graph_dict[('mrna', 'm_m', 'mrna')] = (torch.tensor(a['mrna1'].tolist()+a['mrna2'].tolist()), torch.tensor(a['mrna2'].tolist()+a['mrna1'].tolist()))
    a = pd.read_csv(network_path+'/lnc_lnc.csv')
    graph_dict[('lncrna', 'lnc_lnc', 'lncrna')] = (torch.tensor(a['lncrna1'].tolist()+a['lncrna2'].tolist()), torch.tensor(a['lncrna2'].tolist()+a['lncrna1'].tolist()))

    g = dgl.heterograph(graph_dict,num_nodes_dict=Specified_quantity)

    #加载k-mer特征
    k_mer_path = "./data/paad_rna_feature"
    g.nodes["mirna"].data['mirna_kmer'] = torch.tensor(np.array(pd.read_csv(k_mer_path + '/mirna_kmer.csv').iloc[:,1:]))
    g.nodes["mrna"].data['mrna_kmer'] = torch.tensor(np.array(pd.read_csv(k_mer_path + '/mrna_kmer.csv').iloc[:,1:]))
    g.nodes["lncrna"].data['lncrna_kmer'] = torch.tensor(np.array(pd.read_csv(k_mer_path + '/lncrna_kmer.csv').iloc[:,1:]))

    return g