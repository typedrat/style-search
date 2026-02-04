# Training Similarity Functions

This document describes approaches for learning personalized similarity functions from triplet judgments.

## Current Approach: Weighted Euclidean Distance

**Parameters:** 1,024 (one weight per embedding dimension)

The model learns a weight vector `w` and computes distance as:

```
d(x, y) = sqrt(sum(w_i * (x_i - y_i)^2))
```

Weights are initialized to 1.0 (standard Euclidean) and constrained positive via softplus. Training uses triplet margin loss.

**Pros:**
- Simple and interpretable
- Few parameters, works with limited data
- Weights reveal which dimensions matter for your preferences

**Cons:**
- Can only scale dimensions, not rotate or combine them
- Assumes dimensions are independent

**Triplets needed:** 100-500

---

## Active Learning

To make labeling more efficient, the system can suggest triplets using uncertainty + diversity sampling instead of random selection.

### Uncertainty Sampling

Triplets where the current model predicts similar distances to both options are more informative:

```
uncertainty = 1 / (1 + |d(anchor, A) - d(anchor, B)|)
```

High uncertainty means the model is unsure—your judgment will teach it something new.

### Diversity Sampling

To avoid redundant comparisons:

- **Anchor penalty** — Penalize reusing the same anchor repeatedly
- **Cluster coverage** — Reward triplets that span different clusters (via KMeans on embeddings)

### Combined Score

```
score = 0.6 * uncertainty + 0.4 * diversity
```

The system samples 100 random candidate triplets and returns the one with the highest combined score.

### Warm Updates

After each judgment, the model runs a few gradient steps on recent triplets to stay current. Every 20 triplets, a full retrain runs in the background to consolidate learning.

---

## Multi-Dataset Training

When you have the same items rendered by different base models (e.g., different diffusion models), you can combine triplet judgments while keeping embeddings scoped:

```bash
python -m style_search.train_similarity dataset1 dataset2 -o combined.safetensors
```

Each dataset's IDs are prefixed internally (`dataset1:artist_name`), so triplets only compare items within their own embedding space. The combined training learns weights that generalize across both contexts—dimensions that only matter for one model get diluted, while consistently important dimensions get reinforced.

---

## Alternative Approaches

### 1. Mahalanobis Distance

**Parameters:** ~500k (full) or 1,024 (diagonal)

Learns a positive semi-definite matrix M:

```
d(x, y) = sqrt((x - y)^T M (x - y))
```

Full Mahalanobis can capture correlations between dimensions. Diagonal Mahalanobis is equivalent to weighted Euclidean.

**Pros:**
- Can model dimension correlations
- Well-studied metric learning approach

**Cons:**
- Full matrix has ~500k parameters (infeasible without massive data)
- Low-rank approximations add complexity

**Triplets needed:**
- Diagonal: 100-500
- Low-rank (k=32): 5,000-10,000
- Full: 50,000+

---

### 2. Learned Projection + Cosine Similarity

**Parameters:** 32k-131k (depending on target dimension)

Project embeddings to a smaller space, then use cosine similarity:

```
d(x, y) = 1 - cos(Wx, Wy)
```

**Pros:**
- Learns a task-specific embedding space
- Can combine dimensions in useful ways
- Cosine similarity is scale-invariant

**Cons:**
- Many more parameters than weighted distance
- Harder to interpret

**Triplets needed:**
- Project to 32 dims: 3,000-5,000
- Project to 128 dims: 10,000-20,000

---

### 3. MLP (Neural Network)

**Parameters:** 10k-100k+ (architecture dependent)

Feed concatenated or differenced embeddings through a neural network:

```
similarity = MLP(concat(x, y)) or MLP(x - y)
```

**Pros:**
- Most expressive, can learn arbitrary similarity functions
- Can capture complex nonlinear relationships

**Cons:**
- Needs the most data
- Black box, not interpretable
- Easy to overfit

**Triplets needed:** 10,000-50,000+

---

### 4. Attention-Based Weighting

**Parameters:** 2k-10k

Learn query-dependent attention over dimensions:

```
w = softmax(W_attn @ anchor)
d(x, y) = sqrt(sum(w_i * (x_i - y_i)^2))
```

**Pros:**
- Weights can vary based on the anchor image
- More expressive than fixed weights, fewer params than MLP

**Cons:**
- More complex to implement
- Moderate data requirements

**Triplets needed:** 1,000-5,000

---

## Summary

| Approach | Parameters | Triplets Needed | Expressiveness |
|----------|------------|-----------------|----------------|
| Weighted Euclidean | 1k | 100-500 | Low |
| Diagonal Mahalanobis | 1k | 100-500 | Low |
| Attention-Based | 2k-10k | 1k-5k | Medium |
| Learned Projection | 32k-131k | 3k-20k | Medium-High |
| Low-Rank Mahalanobis | 65k | 5k-10k | Medium |
| MLP | 10k-100k+ | 10k-50k+ | High |
| Full Mahalanobis | 500k | 50k+ | High |

## Recommendations

1. **Start with weighted Euclidean** (current approach). It works with limited data and provides interpretable results.

2. **At ~1,000 triplets**, consider attention-based weighting for query-dependent similarity.

3. **At ~5,000+ triplets**, learned projection becomes viable and can capture richer structure.

4. **Avoid MLP and full Mahalanobis** unless you have tens of thousands of triplets.
