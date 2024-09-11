import torch
from load import *

station_sub_action = torch.load("../../../Sample/degree0_2_subgraph_300_2_.pt")
degree0_graph = load_degree0_graph()
adj = torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(degree0_graph).to_networkx().to_undirected()).todense(),dtype=float))

all_neibor = {}
#计算每个子图每个节点的邻居
for i in range(156):
    all_neibor[i] = {}
    for j in range(300):

        node = station_sub_action[i][j]
        neiborgh = list(torch.nonzero(adj[node,:]))

        all_neibor[i][j] = neiborgh
torch.save(all_neibor,"all_neibor_N300-2.pt")