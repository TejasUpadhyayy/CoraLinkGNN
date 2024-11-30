
# **Cora Citation Network**

The Cora dataset is a graph dataset that represents a network of scientific publications and their citations. It is widely used in machine learning research for tasks involving graph structures, such as node classification, clustering, and link prediction.

## **Dataset Details:**
- **Nodes:** 2,708 scientific papers.
- **Edges:** 5,429 citation relationships.
- **Features:** A 1,433-dimensional feature vector representing the bag-of-words representation of each paper.
- **Classes:** Papers are classified into one of 7 categories:
  - Case-Based
  - Genetic Algorithms
  - Neural Networks
  - Probabilistic Methods
  - Reinforcement Learning
  - Rule Learning
  - Theory

## **Structure:**
The dataset is structured as a single graph:
- **Adjacency Matrix (A):** Encodes the connections between nodes (citations).
- **Feature Matrix (X):** Stores feature vectors for each node (bag-of-words).
- **Label Vector (Y):** Contains the ground truth label for each node.

## **Preprocessing:**
The dataset was preprocessed as follows:
1. Added self-loops to the adjacency matrix to improve model learning.
2. Split into:
   - **Training set (85%)**
   - **Validation set (10%)**
   - **Testing set (5%)**
3. Negative sampling: Non-existent edges were sampled to create a balanced dataset for link prediction tasks.

## **Access:**
The dataset is available for download from:
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid)
- [Kaggle](https://www.kaggle.com/cora-dataset)

## **Visualization**
Below is an example of the Cora citation network. Nodes represent papers, and edges represent citation relationships.

*(Include a visualization image here if available, e.g., cora_graph.png)*

## **Citation:**
If you use this dataset, please cite:
```
@article{mccallum2000automating,
  title={Automating the construction of internet portals with machine learning},
  author={McCallum, Andrew and Nigam, Kamal and Rennie, Jason and Seymore, Kristie},
  journal={Information Retrieval},
  volume={3},
  number={2},
  pages={127--163},
  year={2000},
  publisher={Springer}
}
```
