# SVD-Popularity Hybrid Recommender System

## Overview
This project implements an efficient Recommender System designed for implicit user-item interaction datasets. It utilizes **Truncated SVD** for latent factor analysis and integrates a **Global Popularity Bias** to improve recommendation quality for sparse users (Cold Start).

**Key Features:**
* **High Efficiency:** Uses `scipy.sparse` CSR matrices to handle large-scale data with minimal memory.
* **Hybrid Approach:** Combines Matrix Factorization with popularity weighting.
* **Performance:** Optimized for **NDCG@20** evaluation metrics.
