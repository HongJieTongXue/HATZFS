# HATZFS predicts pancreatic cancer driver biomarkers by hierarchical reinforcement learning and zero-forcing set
![image](https://github.com/HongJieTongXue/HATZFS/blob/master/data/github_img.png)

## 1.Overview
The code for paper "HATZFS predicts pancreatic cancer driver biomarkers by hierarchical reinforcement learning and zero-forcing set". The repository is organized as follows:
* `data/` contains all the datasets used in this paper
* `Sample/` generates subgraphs and saves subgraph data
* `Train/` trains models and predicts biomarkers
## 2.Dependencies
* pandas == 2.0.3
* numpy == 1.21.1
* networkx == 3.1
* torch == 2.0.1+cu118
* dgl == 1.1.2+cu118
## 3.Quick Start
We provide a sample for generating subgraphs of size 50 and training the model:
1. Download and upzip our data and code files
2. Run  `genenate_subgraph1.py` and `genenate_subgraph2.py` (genenate subgraph)
3. Run `get_neibor_linux.py` (genenate neighbor layer)
4. Run `hdqn.py` (train model)
5. Run `get_sorted_big_cooesed.py` and `get_avg_scores.py` (predict biomarkers)

## 4.Contacts
If you have any questions, please email Jie Hong (hongjie22@mails.jlu.edu.cn)
