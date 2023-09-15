### Clustering

**Learning Goals:**
- Discuss and discover use cases for clustering across multiple industries.
- Recognize common clustering algorithms.
- Understand how the k-means clustering algorithm works.
- Implement k-means clustering in Python.
- Handle outliers using IQR.
- Practice scaling data.
- Strategies for handling missing values.
- Plot clusters.
- Make use of discovered clusters later in the data science pipeline.

**About Clustering:**
- Clustering is an unsupervised machine learning methodology used to group and identify similar observations when no labels are available.
- It is often used as a preprocessing or exploratory step in the data science pipeline.
- Clustering helps identify groups of data points that are similar to each other, even when there are no predefined labels.
- Common clustering methodologies include partitioned-based clustering (K-Means), hierarchical clustering, and density-based clustering (DBSCAN).
- Clustering can be used for text classification, geographic analysis, marketing, anomaly detection, image processing, and more.

**Vocabulary:**
- **Euclidean Distance:** The shortest distance between two points in n-dimensional space.
- **Manhattan Distance:** The distance between two points is the sum of the absolute differences of their Cartesian coordinates.
- **Cosine Similarity:** Measures the cosine of the angle between two vectors to define similarity.
- **Sparse Matrix:** A matrix in which most elements are zero.
- **Dense Matrix:** A matrix where most elements are nonzero.

**Common Clustering Algorithms:**
- **K-Means:** The most popular clustering algorithm that stores K centroids and assigns data points to clusters based on proximity to centroids.
- **Hierarchical Clustering:** Clusters are formed by merging or splitting clusters in a hierarchy.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Clusters objects based on the density of the space around them.

**K-Means:**
- Key Hyperparameters:
  - Number of Clusters (k).
  - Random State (for reproducibility).
- Pros:
  - Scales well with data.
  - Creates refined clusters.
  - Centroids can be recomputed.
- Cons:
  - Naive use of mean value for cluster center.
  - Fails for non-circular clusters.
  - Sensitive to initial conditions and data order.

**Hierarchical Clustering:**
- Bottom-up and top-down approaches.
- Key Hyperparameters:
  - Linkage (ward, average, complete, single).
  - Affinity (Euclidean, Manhattan, Cosine).
- Pros:
  - Intuitive for deciding cluster numbers.
  - Relatively easy to implement.
- Cons:
  - Doesn't scale well with large datasets.
  - Less adaptable.
  - Sensitive to outliers and data order.

**DBSCAN:**
- Key Hyperparameters:
  - Epsilon (eps) and MinPts.
  - Distance Metrics (Euclidean, Manhattan, Cosine).
- Pros:
  - No need to pre-set the number of clusters.
  - Identifies outliers.
  - Handles arbitrarily sized and shaped clusters.
- Cons:
  - Doesn't scale well with large datasets.
  - Challenging with clusters of varying density.
  - Less suitable for high-dimensional data.
