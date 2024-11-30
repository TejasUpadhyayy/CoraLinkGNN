
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

![image](https://github.com/user-attachments/assets/b3b400e6-5320-41d4-9f45-896e5aa9d7be)


---

### **Link Prediction**
The link prediction task involves estimating the probability \( P(u, v) \) that an edge exists between two nodes \( u \) and \( v \). The prediction is based on the embeddings learned by the GNN.

![image](https://github.com/user-attachments/assets/a749ce3f-588e-4ce0-abb1-e25b5a3733c2)


---

### **Architectures**

#### **1. Graph Convolutional Networks (GCN)**
GCNs perform spectral convolutions to capture local neighborhood information.

![image](https://github.com/user-attachments/assets/7c67baf4-aede-424c-b873-bc09cd1d1e53)


#### **2. GraphSAGE**
GraphSAGE introduces inductive learning by sampling fixed-size neighborhoods.

![image](https://github.com/user-attachments/assets/61f504b1-2679-485e-817d-40aa4a6b8170)

Common aggregators include:
- **Mean pooling**
- **Max pooling**
- **LSTM pooling**

#### **3. Graph Attention Networks (GAT)**
GAT uses attention mechanisms to dynamically assign importance to neighbors.

![image](https://github.com/user-attachments/assets/b934fc22-c861-4758-9d32-c1022cf967ac)


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
![image](https://github.com/user-attachments/assets/3d06dded-df67-4191-be94-455116e70f9e)


---

## **Summary of Results**

| Model       | Test AUC | Test AP |
|-------------|----------|---------|
| **GCN**     | 0.9268   | 0.9324  |
| **GraphSAGE** | 0.7394 | 0.7423  |
| **GAT**     | 0.5803   | 0.5754  |

### **Best Model**
- **GCN** emerged as the best-performing model in the second notebook, achieving the highest **Test AUC (0.9268)** and **Test AP (0.9324)**.

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
