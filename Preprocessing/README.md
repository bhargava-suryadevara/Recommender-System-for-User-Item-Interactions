
# Dataset Preprocessing Documentation

## 1. Overview

The dataset consists of implicit user–item interactions stored in a text file where each line has the format:

```
user_id item_id1 item_id2 item_id3 ...
```

Each item on a user’s line represents an interaction such as a click, view, or purchase.  
The goal of preprocessing is to convert this raw text format into numerical structures suitable for recommendation models such as **LightGCN**.

This preprocessing involves:

- Mapping user and item string IDs to integer indices  
- Building a sparse interaction matrix  
- Creating fast lookup dictionaries  
- (Optionally) performing train/validation splitting  

No additional feature extraction or data cleaning is performed.

---

## 2. Preprocessing Steps

### Step 1 — Load Raw Dataset

The raw text file is read line-by-line.  
Each line contains:

- A **user ID** (string)
- A list of **item IDs** the user interacted with

The parser splits each line into:

- `user_id`
- `item_ids = [item1, item2, …]`

This step performs no modification or filtering; it only reads raw data.

---

### Step 2 — Encode User IDs

The dataset stores user IDs as strings, which must be converted to integer indices for model training.

Example:

| Raw User ID | Mapped User Index |
|-------------|-------------------|
| `"U10002"`  | 0                 |
| `"U58344"`  | 1                 |

Two structures are created:

- `user2idx` → dictionary mapping user ID → integer index  
- `idx2user` → list mapping index → original user ID  

This guarantees:

- Contiguous user indices  
- Efficient embedding lookup  

---

### Step 3 — Encode Item IDs

Item IDs are handled similarly:

| Raw Item ID | Mapped Item Index |
|--------------|-------------------|
| `"I395"`      | 0                 |
| `"I9912"`     | 1                 |

Two structures are produced:

- `item2idx` → item ID → index  
- `idx2item` → index → raw item ID  

---

### Step 4 — Build Sparse User–Item Interaction Matrix

For each interaction, we add:

```
interaction_matrix[user_index, item_index] = 1
```

This matrix:

- Has shape `(num_users, num_items)`  
- Stores only the observed interactions  
- Is constructed in **COO** and converted to **CSR** format  
- Is memory-efficient and ideal for recommender models

Example conceptual matrix:

```
      Items →
U0    [0 0 1 0 1 0 ...]
U1    [1 0 0 0 0 0 ...]
U2    [0 0 0 1 1 1 ...]
```

---

### Step 5 — Build `user_items` Dictionary

A lookup dictionary is created:

```
user_items[user_index] = {item_1, item_2, item_2, ...}
```

This is used later for:

- Negative sampling  
- Recommendation filtering (exclude already-seen items)  
- Train/validation splits  

---

### Step 6 (Optional) — Train/Validation Split

To evaluate models, interactions can be split **per user**:

- A fraction (e.g., 10%) → **validation set**
- Remaining interactions → **training set**

This ensures:

- Every user has at least 1 training interaction  
- Validation is done on unseen items  

Example:

```
User 45 interacted with: [3, 9, 22, 41, 50]

Train: [3, 9, 41, 50]
Validation: [22]
```

Outputs:

- `train_user_items`
- `val_user_items`

---

## 3. What Is Not Done

This preprocessing intentionally does **not** include:

- Removing duplicates (none expected in format)
- Feature engineering
- Normalization or scaling  
- Timestamp handling  
- Data filtering  
- Statistical analysis  

This keeps the dataset in pure implicit-feedback form for LightGCN-style models.

---

## 4. Summary

The preprocessing pipeline converts raw text interactions into:

- Integer-encoded user IDs  
- Integer-encoded item IDs  
- A sparse CSR user–item matrix  
- A dictionary mapping users to their interacted items  
- (Optional) train/validation splits  

These structures are ready for use in graph-based recommender models.

