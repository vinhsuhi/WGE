##### Table of Content

1. [Introduction](#two-view-graph-neural-networks-for-knowledge-graph-completion)
1. [Getting Started](#getting-started)
    - [Datasets](#datasets)
    - [Installation](#installation)
1. [Experiments](#experiments)
    - [Training & Testing](#training-and-testing)


# Two-view Graph Neural Networks for Knowledge Graph Completion

We present an effective graph neural network (GNN)-based knowledge graph embedding model, which we name WGE, to capture entity- and relation-focused graph structures. Given a knowledge graph, WGE builds a single undirected entity-focused graph that views entities as nodes. WGE also constructs another single undirected graph from relation-focused constraints, which views entities and relations as nodes. WGE then proposes a GNN-based architecture to better learn vector representations of entities and relations from these two single entity- and relation-focused graphs. WGE feeds the learned entity and relation representations into a weighted score function to return the triple scores for knowledge graph completion. Experimental results show that WGE outperforms strong baselines on seven benchmark datasets for knowledge graph completion.  

<img src="./figs/model.png" width="800">


Details of the model architecture and experimental results can be found in [our following paper](https://arxiv.org/abs/2112.09231):

```
@inproceedings{wge,
    title     = {{Two-view Graph Neural Networks for Knowledge Graph Completion}},
    author    = {Vinh Tong and Dai Quoc Nguyen and Dinh Phung and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 20th Extended Semantic Web Conference},
    year      = {2023}
}
```
**Please CITE** our paper whenever our model implementation is used to help produce published results or incorporated into other software.

## Getting Started

### Datasets
LitWD, CodEx and FB15k237 datasets are stored in data.zip. 
Please xtract the zip file before running the code.

### Installation:
```
# clone the repo
git clone https://github.com/vinhsuhi/WGE.git
cd WGE

# install dependencies
pip install -r requirements.txt
```


## Experiments
### Training and Testing
```
python main.py --dataset codex-s --lr 0.0005 --beta 0.2 --emb_dim 256
python main.py --dataset codex-m --lr 0.0005 --beta 0.2 --emb_dim 256
python main.py --dataset codex-l --lr 0.0001 --beta 0.2 --emb_dim 256
```

