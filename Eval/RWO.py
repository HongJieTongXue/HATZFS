# coding: utf-8
import networkx as nx
import numpy as np
import random
import os
import networkx as nx
import math
import operator
import matplotlib.pyplot as plp
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List

# Libraries for graphs
import community
import networkx as nx

# Libraries for matrix computations
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from scipy.stats import pearsonr


import torch
import warnings
import argparse
import numpy as np
from torch import nn, optim
from load import load_random_hetero_graph
import os
import mvc_env
import networkx as nx
from load import *
from DQN import DQN
import dgl
import pickle
from model import RGCNDQNModel
import matplotlib.pyplot as plt
from eval import *
import sys


from load import load_random_hetero_graph
import dgl
sys.path.append('../')
warnings.filterwarnings("ignore")


# load network
# input file name format:
#     [NetName].txt
# input file format:
#     [gene1]\t[gene2]\r\n
def load_network(netpath, netName, s):
    os.chdir(netpath)
    a = open(netName, "r")
    G1 = nx.Graph()
    for i in a:
        n = i.strip().split("\t")
        G1.add_edge('_'.join([s, n[0]]), '_'.join([s, n[1]]))
    a.close()
    number = list(G1.nodes())
    numNodes = len(number)
    numEdges = G1.number_of_edges()
    return G1, number, numNodes, numEdges


# mapping
def mapping(G1, G2):
    matrix_mappingID = {}
    matrix_mappingName = {}
    matrix_mappingType = {}
    numAllGene = 0
    for x in G1.nodes():
        matrix_mappingID[x] = numAllGene
        matrix_mappingName[numAllGene] = x
        matrix_mappingType[numAllGene] = 'None'
        numAllGene = numAllGene + 1
    for x in G2.nodes():
        matrix_mappingID[x] = numAllGene
        matrix_mappingName[numAllGene] = x
        matrix_mappingType[numAllGene] = 'None'
        numAllGene = numAllGene + 1
    return matrix_mappingID, matrix_mappingName, matrix_mappingType, numAllGene


# load homology information
# input file name format:
#     [homoName].txt
# input file format:
#     [s1Gene]\t[s2Gene]\r\n
def load_homo_info(homoPath, homoName):
    tupleHomoInfo = []
    os.chdir(homoPath)
    f = open(homoName, 'r')
    for line in f.readlines():
        list = line.strip().split('\t')
        tupleHomoInfo.append((list[0], list[1]))
    f.close()
    return (tupleHomoInfo)


# initialization
def intro_initial_RW(G1, G2, count, matrix_mappingID):
    RW = np.zeros(shape=(count, count))
    for (x, y) in G1.edges():
        RW[matrix_mappingID[x]][matrix_mappingID[y]] = 1
        RW[matrix_mappingID[y]][matrix_mappingID[x]] = 1
    for (x, y) in G2.edges():
        RW[matrix_mappingID[x]][matrix_mappingID[y]] = 1
        RW[matrix_mappingID[y]][matrix_mappingID[x]] = 1
    return RW


# inter-initial RW
def inter_initial_RW(RW, tupleHomoInfo,
                     matrix_mappingID, matrix_mappingType):
    count = 0
    for (s1g, s2g) in tupleHomoInfo:
        if s1g in matrix_mappingID and s2g in matrix_mappingID:
            RW[matrix_mappingID[s1g]][matrix_mappingID[s2g]] = 1
            RW[matrix_mappingID[s2g]][matrix_mappingID[s1g]] = 1
            matrix_mappingType[matrix_mappingID[s2g]] = 'homo'
            matrix_mappingType[matrix_mappingID[s1g]] = 'homo'
            count = count + 1
    return RW, matrix_mappingType, count


# create probability transfer matrix
def Pmatrix(a1, a2, RW, matrix_mappingType, numAllGene, numNodeS1):
    Pr = np.zeros(shape=(numAllGene, numAllGene))
    for i in range(numAllGene):
        degree = sum(RW[i, :])
        homo = 0
        normal = 0
        if i < numNodeS1:
            across = sum(RW[i, numNodeS1:numAllGene])
            for x in range(numNodeS1):
                if matrix_mappingType[x] == 'homo' and RW[i][x] > 0:
                    homo = homo + 1
                if matrix_mappingType[x] == 'None' and RW[i][x] > 0:
                    normal = normal + 1

        if i >= numNodeS1:
            across = sum(RW[i, 0:numNodeS1])
            for x in range(numNodeS1, numAllGene):
                if matrix_mappingType[x] == 'homo' and RW[i][x] > 0:
                    homo = homo + 1
                if matrix_mappingType[x] == 'None' and RW[i][x] > 0:
                    normal = normal + 1
        alfa = a1
        beta = a2
        if degree > 0:
            if homo == 0:
                beta = 0
            if across == 0:
                alfa = 0
            if homo == 0 and normal == 0:
                alfa = 1
            for j in range(numAllGene):
                if i < numNodeS1 and j < numNodeS1:
                    if matrix_mappingType[j] == 'homo' and homo > 0:
                        Pr[i][j] = RW[i][j] * (1 - alfa) * beta / homo
                    if matrix_mappingType[j] == 'None' and normal > 0:
                        Pr[i][j] = RW[i][j] * (1 - alfa) * (1 - beta) / normal
                # break
                else:
                    if i >= numNodeS1 and j >= numNodeS1:
                        if matrix_mappingType[j] == 'homo' and homo > 0:
                            Pr[i][j] = RW[i][j] * (1 - alfa) * beta / homo
                        if matrix_mappingType[j] == 'None' and normal > 0:
                            Pr[i][j] = RW[i][j] * (1 - alfa) * (1 - beta) / normal

                    else:
                        if across > 0:
                            Pr[i][j] = RW[i][j] * alfa / across

    return Pr


# RWO
def output_RWO(rho, tau, RW, matrix_mappingType, numAllGene, numNodeS1,
               matrix_mappingName):
    P = Pmatrix(rho, tau, RW, matrix_mappingType, numAllGene, numNodeS1)
    Pt = np.ones(shape=(1, numAllGene))
    for i in range(100):
        Pt = np.dot(Pt, P)
    score_dict = {}
    for i in range(numAllGene):
        score_dict[matrix_mappingName[i]] = Pt[0][i]
    return score_dict


def RWO(G1, G2, tupleHomoInfo, rho=0.1, tau=0.4):

    # mapping
    print('mapping...')
    matrix_mappingID, matrix_mappingName, matrix_mappingType, numAllGene \
        = mapping(G1, G2)

    # load homo information
    # input file name format:
    #     [homoName].txt
    # input file format:
    #     [s1Gene]\t[s2Gene]\r\n
    ''' print('load homology information...')
    tupleHomoInfo = load_homo_info(homoPath, homoName)
    print('    Homology information:')
    print('    Number of information: '+str(len(tupleHomoInfo)))'''

    # initializaition
    print('Initializaition...')
    RW = intro_initial_RW(G1, G2, numAllGene, matrix_mappingID)
    RW, matrix_mappingType, count \
        = inter_initial_RW(RW, tupleHomoInfo, matrix_mappingID, matrix_mappingType)

    # RWO
    print('RWO...')
    score_dict=output_RWO(rho, tau, RW, matrix_mappingType, numAllGene, 3049, matrix_mappingName)
    return score_dict


def RWO_sort():
    init_graph = load_random_hetero_graph('../data/PAAD_network',
                                          Specified_quantity={'lncrna': 1500, 'mirna': 168, 'mrna': 2519})
    G = dgl.to_homogeneous(init_graph).to_networkx().to_undirected()

    # nodes1 = [n for n, v in nx.degree(G) if v != 0]
    # nodes0 = [n for n, v in nx.degree(G) if v == 0]
    nodes1 = torch.load("degree_nodes.pt")
    G = G.subgraph(nodes1)
    # G1 = G.subgraph(nodes1[:690] + nodes1[690 + 168:])
    G1 = G.subgraph(nodes1[:643] + nodes1[643 + 168:])
    G2 = G.subgraph(nodes1[643:643 + 168])
    tupleHomoInfo = []
    for i in G.edges():
        if (i[0] in nodes1[:643] + nodes1[643 + 168:]) & (i[1] in nodes1[643:643 + 168]):
            tupleHomoInfo += [i]
        elif (i[1] in nodes1[:643] + nodes1[643 + 168:]) & (i[0] in nodes1[643:643 + 168]):
            tupleHomoInfo += [i]
    score = RWO(G1, G2, tupleHomoInfo, rho=0.1, tau=0.4)
    for i in score.keys():
        print (score[i])
    rwos = [(v, score[v]) for v in score]
    sorted_rwos = sorted(rwos, key=lambda x: (x[1], x[0]), reverse=True)
    action_choosed = [x[0] for x in sorted_rwos]
    torch.save(action_choosed, "RWO.pt")
    return action_choosed



