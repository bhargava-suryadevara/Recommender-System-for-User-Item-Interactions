
# LightGCN Preprocessing & Modeling Pipeline Documentation

This model has got a score of 0.0349 in the leaderboard.(Submission file is also added here)

This document explains the purpose of each part of the provided code and describes the full workflow implemented in the **LightGCN.ipynb** notebook.  
The explanation covers preprocessing, model construction, training, evaluation, and recommendation generation.

---

# 🔹 1. Overview

The notebook implements a full **LightGCN-based recommendation pipeline**.  
It includes:

- Dataset loading  
- Preprocessing 
- Graph construction  
- LightGCN model implementation  
- BPR Loss  
- Training loop  
- NDCG@20 evaluation  
- Recommendation generation  
- Multiple diagnostic plots (loss, NDCG, CPU/memory usage, etc.)

---

# 🔹 2. Preprocessing Steps

### ✔ Load the dataset  
The interactions are read from:

```
train-1.txt
```

Each line has:

```
user_id item1 item2 item3 ...
```

### ✔ Map IDs → Integer indices  
To build embeddings, user IDs and item IDs are mapped to continuous integer indices:

- `user2idx` – maps raw user IDs → integer index  
- `item2idx` – maps raw item IDs → integer index  
- `idx2user`, `idx2item` – reverse lookup lists  

### ✔ Build sparse interaction matrix (CSR)  
A sparse matrix is created:

```
interaction_matrix[user, item] = 1
```

Used later for popularity calculation and evaluation.

### ✔ Create `user_items` dictionary  
Stores:

```
user_items[user] = {item1, item2, ...}
```

Required for sampling positives and avoiding already-seen items.

### ✔ Optional: Train/Validation Split  
A 90/10 split is created per user:

- Ensures every user has at least 1 train item  
- Validation set used for NDCG@20 evaluation  

---

# 🔹 3. Graph Construction (Required for LightGCN)

The notebook constructs a **normalized adjacency matrix** for user-item graph:

```
User nodes: 0 … num_users-1
Item nodes: num_users … num_users+num_items-1
```

The adjacency matrix is normalized using:

```
D^(-1/2) * A * D^(-1/2)
```

Converted to a PyTorch sparse tensor for efficient propagation.

---

# 🔹 4. LightGCN Model

The model contains:

### ✔ Embeddings  
A learned embedding for each user and item node.

### ✔ Propagation Layers  
Each layer performs:

```
E_(k+1) = A_norm * E_k
```

Final representation is the **average** of embeddings across all layers (as in the LightGCN paper).

### ✔ BPR Loss  
The model is trained using the Bayesian Personalized Ranking Loss:

```
log(sigmoid(score_positive - score_negative))
```

Regularization is also applied.

---

# 🔹 5. Training Pipeline

### ✔ Mini-batch sampling  
For each batch:

- Sample a user  
- Pick one positive item (seen)  
- Pick one negative item (not seen)

### ✔ Optimizer  
Adam with learning rate 0.001

### ✔ Early stopping  
Stops after 8 epochs with no improvement in validation NDCG@20.

### ✔ Metrics captured

- Loss  
- NDCG@20  
- CPU usage  
- Memory usage  
- Epoch time  

All stored in `history`.

---

# 🔹 6. Evaluation — NDCG@20

### ✔ Why NDCG?  
Ranking-based metric that measures how well top-K recommendations match actual user preferences.

### ✔ How it works  
For each validation user:

1. Score all items  
2. Mask already-seen items  
3. Sort scores  
4. Compute NDCG@20  
5. Average across validation users  

---

# 🔹 7. Diagnostic Visualizations

The notebook generates many plots, including:

- Training loss per epoch  
- NDCG@20 progression  
- CPU usage per epoch  
- Memory usage per epoch  
- Epoch time  
- User interaction histogram  
- Item popularity histogram  
- Sparsity heatmap of interaction matrix  

These help visually understand model learning and dataset characteristics.

---

# 🔹 8. Recommendation Generation

Final recommendations are produced using:

```
scores = alpha * collaborative_filtering_score 
       + (1 - alpha) * popularity_score
```

- `alpha = 0.8` → mostly CF, slight popularity smoothing  
- For each user, items already seen are masked  
- Top 20 items written to:

```
submission.txt
```

---

# 🔹 9. Summary of What the Notebook Does

The notebook performs the **entire LightGCN pipeline end-to-end**:

### ✔ Preprocess raw dataset  
### ✔ Build graph and LightGCN model  
### ✔ Train with BPR Loss  
### ✔ Evaluate using NDCG@20  
### ✔ Generate submission file  
### ✔ Produce multiple visual diagnostics  


