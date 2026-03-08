# Distance Metrics and k-NN

A hands-on Python notebook demonstrating the core distance metrics used in machine learning, and how they power the k-Nearest Neighbors (k-NN) classification algorithm.

---

## Overview

This notebook walks through four key concepts with simple examples and visualizations:

| (a) | L1 Distance — Manhattan Distance |


| (b) | L2 Distance — Euclidean Distance |


| (c) | Cosine Distance — Angle Between Vectors |


| (d) | k-Nearest Neighbors (k-NN) Classification |

---

## Concepts Covered

### (a) L1 Distance (Manhattan Distance)
Measures the sum of absolute differences between two points.

```
L1 = Σ |xᵢ - yᵢ|
```

**Example result:** L1 Distance between `[1,2]` and `[4,6]` = `7`

---

### (b) L2 Distance (Euclidean Distance)
Measures the straight-line distance between two points.

```
L2 = √Σ (xᵢ - yᵢ)²
```

**Example result:** L2 Distance between `[1,2]` and `[4,6]` = `5.0`

---

### (c) Cosine Distance
Measures the angle between two vectors — useful for comparing direction, not magnitude. Great for text and high-dimensional data.

```
Cosine Similarity = (a · b) / (||a|| × ||b||)
Cosine Distance = 1 - Cosine Similarity
```

**Example result:** Vectors `[1,2,3]` and `[2,4,6]` are parallel → Cosine Similarity = `1.0`, Distance = `0.0`

---

### (d) k-Nearest Neighbors (k-NN)
Classifies a new data point by finding the `k` closest training examples and taking a majority vote.

- Uses `sklearn.neighbors.KNeighborsClassifier`
- `k=3` in the demo
- Training data: 5 labeled 2D points (Class 0 and Class 1)
- Test point `[4, 4]` → **Predicted class: 0**

---


## 👤 Author

**kopiyo** — Part of a machine learning portfolio project.
