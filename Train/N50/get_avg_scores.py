
import torch

node_scores = {}
length = 10
N = 50
idex = 20
degree_nodes = torch.load("../../Sample/graphid_2_degree0_250.pt")

# indexs = ['old_6']
# indexs = ['old_5']
# indexs = ['old_2','old_3','old_4','old_5','old_6']
# indexs = ['old_1','old_3','old_4','old_5','old_6']
# indexs = ['old_1','old_2','old_4','old_5','old_6']
# indexs = ['old_1','old_2','old_3','old_5','old_6']
# indexs = ['old_1','old_2','old_3','old_4','old_6']
indexs = ['old_1','old_2','old_3','old_4','old_5']
# indexs = ['old_20']
for i in indexs:
    res = torch.load("./"+str(i)+"/N"+str(N)+"_"+str(i)+"_allscores.pt",map_location=torch.device("cpu"))
    # res = torch.load("./"+"old_2"+"/N"+str(N)+"_"+"old_test_"+str(idex)+"_allscores.pt",map_location=torch.device("cpu"))
    for node in res.keys():
        # t = int(degree_nodes[node, 0])
        if node not in node_scores.keys():
            node_scores[node] = res[node][0]
        else:
            node_scores[node] = node_scores[node] + res[node][0]
node_scores = sorted(node_scores.items(),key=lambda x:x[1], reverse=True)
# print(node_scores)
action_choosed = [int(degree_nodes[node, 0]) for node,_ in node_scores]
torch.save(action_choosed,"N50_avg_5.pt")
degree0_nodes_ispaad = torch.load("all_nodes_ispaad_new.pt")

lnc_choosed = [i for i in action_choosed if i >= 0 and i < 1500]
mi_choosed = [i for i in action_choosed if i >= 1500 and i < 1668]
m_choosed = [i for i in action_choosed if i >= 1668]
print(len(action_choosed), end=" ")
print(len(lnc_choosed), end=" ")
print(len(mi_choosed), end=" ")
print(len(m_choosed))
res = []
count = 0
for ind in range(0, 1000, 100):
    count_t = 0
    for node in action_choosed[ind:ind + 100]:
        if degree0_nodes_ispaad[node]:
            count_t += 1
    count += count_t
    res.append(count)
print(res)

# 计算可控节点集有多少与paad有关
len_all = 0
lnc_all = 0
mi_all = 0
m_all = 0
for node in action_choosed:
    if degree0_nodes_ispaad[node]:
        len_all += 1
for node in lnc_choosed:
    if degree0_nodes_ispaad[node]:
        lnc_all += 1
for node in mi_choosed:
    if degree0_nodes_ispaad[node]:
        mi_all += 1
for node in m_choosed:
    if degree0_nodes_ispaad[node]:
        m_all += 1
print(len_all, lnc_all, mi_all, m_all)
l_nall = []
mi_nall = []
m_nall = []
res = 0
for i in range(0, 200, 50):
    count = 0
    for lnc in lnc_choosed[i:i + 50]:
        if degree0_nodes_ispaad[lnc]:
            count += 1
    res += count
    l_nall.append(res)
res = 0
for i in range(0, 50, 10):
    count = 0
    for mi in mi_choosed[i:i + 10]:
        if degree0_nodes_ispaad[mi]:
            count += 1
    res += count
    mi_nall.append(res)
res = 0
for i in range(0, 700, 100):
    count = 0
    for m in m_choosed[i:i + 100]:
        if degree0_nodes_ispaad[m]:
            count += 1
    res += count
    m_nall.append(res)

print(l_nall)
print(mi_nall)
print(m_nall)

