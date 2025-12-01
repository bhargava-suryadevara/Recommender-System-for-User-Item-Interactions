
# Data Analysis

This section provides an overview of the interaction dataset used for the recommendation system. The dataset consists of implicit user–item interactions, where each record indicates that a user has interacted with one or more items. The goal of this analysis is to understand the scale, sparsity, and distribution characteristics of the data before modeling.

---

## Dataset Summary

```
============================================================
                  DATASET SUMMARY REPORT
============================================================
Total Users:                       31,668
Total Items:                       38,048
Total Interactions:             1,237,259
------------------------------------------------------------
Matrix Density:                0.00102685
Matrix Sparsity:               99.8973%
============================================================
```

### Interpretation

- The dataset is **large-scale** with **31.6K users** and **38K items**.
- A total of **1.23M interactions** are observed.
- Matrix density of **0.10%** means that **99.897%** of all possible user–item combinations are *missing*.  
  This extreme sparsity is typical for recommendation datasets and motivates the use of graph-based models such as LightGCN.

---

## User Interaction Statistics

```
------------------------------------------------------------
User Interaction Statistics
------------------------------------------------------------
Users with >0 interactions:        31,668
Min interactions per user:             16
Mean interactions per user:         39.07
Median interactions per user:       25.00
Max interactions per user:           1848
------------------------------------------------------------
```

### Interpretation

- Every user in the dataset has interacted with at least **16 items**, indicating good minimum coverage.
- The **average user** interacts with about **39 items**, but the **median** is lower (**25**), showing a skewed distribution.
- The most active users have up to **1,848 interactions**, suggesting the presence of *power users* who engage with many items.
- This long-tailed user activity distribution is typical for real-world platforms.

---

## Item Popularity Statistics

```
------------------------------------------------------------
Item Popularity Statistics
------------------------------------------------------------
Items with >0 interactions:        38,048
Min interactions per item:              1
Mean interactions per item:         32.52
Median interactions per item:       17.00
Max interactions per item:           1258
------------------------------------------------------------
```

### Interpretation

- All **38K items** receive at least **1 interaction**, so no cold-start items exist.
- The **mean** item popularity is **32.5**, while the **median** is much lower (**17**).  
  This again indicates **long-tail behavior**, where:
  - a small number of items are highly popular (up to **1,258 interactions**)
  - most items receive modest engagement  
- This distribution is ideal for recommendation models, as it contains both niche and trending items.

---

## Summary

- The dataset is **huge** and **very sparse**, typical for recommendation systems.
- Both users and items show **highly skewed, long-tailed distributions**.
- These characteristics justify the use of:
  - sparse matrix representations,
  - graph-based collaborative filtering (e.g., LightGCN),
  - and ranking metrics like NDCG@20 for evaluation.

