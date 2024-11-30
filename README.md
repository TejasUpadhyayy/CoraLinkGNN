
# **Link Prediction on the Cora Dataset Using Graph Neural Networks**

## **Overview**
This project explores **link prediction** using **Graph Neural Networks (GNNs)**, focusing on the **Cora citation dataset**. The task involves predicting missing or potential future links (edges) in the graph. The implementation compares three state-of-the-art GNN architectures:
- **Graph Convolutional Networks (GCN)**
- **GraphSAGE**
- **Graph Attention Networks (GAT)**

This repository is inspired by the research paper **"Predicting the Future of AI with AI: High-quality Link Prediction in an Exponentially Growing Knowledge Network"** and adapts its techniques to a static graph scenario.

---

## **Project Highlights**
- **Dataset**: The **Cora citation dataset** contains:
  - **Nodes**: 2,708 scientific papers.
  - **Edges**: 5,429 citation links.
  - **Node Features**: Bag-of-words representation (1,433 features per node).
  - **Classes**: Seven categories of research topics.
- **Task**: Predict whether an edge exists between two nodes (link prediction).
- **Key Metrics**:
  - **AUC (Area Under the ROC Curve)**: Evaluates the classification performance.
  - **AP (Average Precision)**: Measures the precision-recall balance.
- **Best Model**: GAT achieved the highest performance with a Test AUC of **0.9550** and Test AP of **0.9510**.

---

## **Theoretical Background**

### **Graph Neural Networks (GNNs)**
GNNs extend traditional neural networks to graph-structured data by learning node embeddings that capture both local and global graph structures. They work by propagating and aggregating information between neighboring nodes.

**General Framework**:
\[
h_v^{(l+1)} = \text{AGGREGATE}\big(\{f(h_u^{(l)}, h_v^{(l)}, e_{uv}) \mid u \in \mathcal{N}(v)\}\big)
\]
Where:
- \( h_v^{(l)} \): Embedding of node \( v \) at layer \( l \).
- \( \mathcal{N}(v) \): Neighboring nodes of \( v \).
- \( f \): Message-passing function (e.g., edge-weighted summation).
- \( \text{AGGREGATE} \): Aggregation function (e.g., mean, max, or sum).

---

### **Link Prediction**
The link prediction task involves estimating the probability \( P(u, v) \) that an edge exists between two nodes \( u \) and \( v \). The prediction is based on the embeddings learned by the GNN.

**Decoder Equation**:
\[
P(u, v) = \sigma(z_u \cdot z_v)
\]
Where:
- \( z_u, z_v \): Node embeddings for \( u \) and \( v \).
- \( \sigma \): Sigmoid activation function to normalize predictions.

---

### **Architectures**

#### **1. Graph Convolutional Networks (GCN)**
GCNs perform spectral convolutions to capture local neighborhood information.

**Key Equation**:
\[
H^{(l+1)} = \sigma\big(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}\big)
\]
Where:
- \( \hat{A} = A + I \): Adjacency matrix with self-loops.
- \( \hat{D} \): Degree matrix of \( \hat{A} \).
- \( W^{(l)} \): Trainable weights at layer \( l \).

#### **2. GraphSAGE**
GraphSAGE introduces inductive learning by sampling fixed-size neighborhoods.

**Key Equation**:
\[
h_v^{(l+1)} = \sigma\big(W^{(l)} \cdot \text{AGGREGATE}(\{h_u^{(l)} \mid u \in \mathcal{N}(v)\})\big)
\]
Common aggregators include:
- **Mean pooling**
- **Max pooling**
- **LSTM pooling**

#### **3. Graph Attention Networks (GAT)**
GAT uses attention mechanisms to dynamically assign importance to neighbors.

**Attention Coefficients**:
\[
e_{uv} = \text{LeakyReLU}\big(a^\top [W h_u \| W h_v]\big)
\]
\[
\alpha_{uv} = \frac{\exp(e_{uv})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{vk})}
\]

**Node Update**:
\[
h_v^{(l+1)} = \sigma\big(\sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u\big)
\]

---

## **Implementation**

### **Dataset Preparation**
1. **Splitting Edges**:
   - Training: 85%
   - Validation: 10%
   - Testing: 5%
2. **Negative Sampling**:
   - Negative edges are sampled to balance the dataset for binary classification.

### **Training**
- **Loss Function**: Binary Cross-Entropy Loss
\[
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \big[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\big]
\]
Where:
- \( y_i \): Ground truth label.
- \( \hat{y}_i \): Predicted probability.

- **Optimizer**: Adam
- **Evaluation Metrics**: AUC, AP

---

## **Results**

| Model       | Test AUC | Test AP |
|-------------|----------|---------|
| **GCN**     | 0.9500   | 0.9470  |
| **GraphSAGE** | 0.9520 | 0.9490  |
| **GAT**     | 0.9550   | 0.9510  |

### **Analysis**
- **GAT** achieved the highest accuracy due to its ability to assign dynamic weights to neighboring nodes.
- **GraphSAGE** outperformed GCN because of its flexible aggregation strategies.

---

## **Conclusion**
This project demonstrates the effectiveness of GNNs in link prediction tasks. The results highlight the importance of advanced architectures like GAT in capturing complex relationships in graph data. This implementation shows:
1. **High Accuracy**: Achieved a Test AUC of **0.9550** with GAT.
2. **Scalability**: Models like GraphSAGE can handle dynamic, inductive tasks.
3. **Applications**: These techniques are broadly applicable to social networks, recommendation systems, and biological networks.

---

## **Future Work**
- Extend the models to handle **dynamic graphs**.
- Incorporate **edge attributes** for richer representations.
- Apply these techniques to **larger, real-world datasets**.

---

## **How to Run**
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook linkpredict2.ipynb
   ```

---

## **References**
1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
2. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs.
3. Veličković, P., et al. (2018). Graph Attention Networks.
4. **Predicting the Future of AI with AI**: High-quality Link Prediction in an Exponentially Growing Knowledge Network.

---
