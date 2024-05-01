import torch
from load import *

station_sub_action = torch.load("../../../Sample/N50/degree0_2_subgraph_50_6_.pt")
degree0_graph = load_degree0_graph()
adj = torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(degree0_graph).to_networkx().to_undirected()).todense(),dtype=float))

all_neibor = {}
#计算每个子图每个节点的邻居
for i in range(335):
    all_neibor[i] = {}
    for j in range(50):

        node = station_sub_action[i][j]
        neiborgh = list(torch.nonzero(adj[node,:]))

        all_neibor[i][j] = neiborgh
torch.save(all_neibor,"all_neibor_N50-6.pt")