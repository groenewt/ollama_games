"""Strategy clustering for allocation games.

Uses dimensionality reduction (PCA, t-SNE) and clustering (K-Means)
to identify strategy archetypes from allocation patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from math import log2


@dataclass
class ClusterResult:
    """Result of strategy clustering."""
    cluster_labels: List[int]           # Cluster ID per allocation
    archetype_names: List[str]          # Human-readable names per cluster
    cluster_centers: np.ndarray         # Cluster centroids in feature space
    features_2d: np.ndarray             # Reduced 2D coordinates for visualization
    inertia: float                      # K-Means inertia (lower = tighter clusters)
    silhouette_score: float             # Clustering quality (-1 to 1, higher = better)


@dataclass
class StrategyArchetype:
    """Describes a strategy archetype."""
    name: str                           # e.g., "Concentrated", "Uniform"
    description: str                    # Human-readable description
    avg_concentration: float            # Average HHI
    avg_allocation: List[float]         # Typical allocation pattern
    count: int                          # Number of allocations in this archetype


class StrategyClusterer:
    """Clusters allocation strategies to identify archetypes."""

    def __init__(self, random_state: int = 42):
        """Initialize clusterer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def extract_features(
        self,
        allocations: List[List[float]],
        budget: float = 100.0
    ) -> np.ndarray:
        """Convert allocations to feature vectors for clustering.

        Features include:
        - Normalized allocation proportions
        - Concentration index (HHI)
        - Entropy
        - Max field proportion
        - Symmetry score

        Args:
            allocations: List of allocation vectors
            budget: Allocation budget for normalization

        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []

        for alloc in allocations:
            if not alloc or sum(alloc) == 0:
                continue

            # Normalize to proportions
            total = sum(alloc)
            props = [a / total for a in alloc]

            # Basic features: proportions
            feature_vec = list(props)

            # HHI (concentration)
            hhi = sum(p ** 2 for p in props)
            feature_vec.append(hhi)

            # Entropy
            entropy = 0.0
            for p in props:
                if p > 0:
                    entropy -= p * log2(p)
            max_entropy = log2(len(alloc)) if len(alloc) > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            feature_vec.append(norm_entropy)

            # Max allocation proportion
            max_prop = max(props)
            feature_vec.append(max_prop)

            # Symmetry score: how different from mirror image
            # (detects left-heavy vs right-heavy strategies)
            reversed_props = list(reversed(props))
            symmetry = 1 - sum(abs(p1 - p2) for p1, p2 in zip(props, reversed_props)) / 2
            feature_vec.append(symmetry)

            # Variance of allocations
            mean_prop = 1 / len(props)
            variance = sum((p - mean_prop) ** 2 for p in props) / len(props)
            feature_vec.append(variance)

            features.append(feature_vec)

        return np.array(features) if features else np.array([])

    def reduce_dimensions(
        self,
        features: np.ndarray,
        method: str = 'pca',
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce feature dimensions for visualization.

        Args:
            features: Feature matrix (n_samples, n_features)
            method: 'pca' or 'tsne'
            n_components: Number of output dimensions

        Returns:
            Reduced matrix (n_samples, n_components)
        """
        if features.size == 0 or len(features) < 2:
            return np.array([])

        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            # t-SNE needs perplexity < n_samples
            perplexity = min(30, len(features) - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=max(5, perplexity),
                random_state=self.random_state,
                n_iter=1000,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

        return reducer.fit_transform(features_scaled)

    def cluster_strategies(
        self,
        features: np.ndarray,
        n_clusters: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Cluster strategies using K-Means.

        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters

        Returns:
            Tuple of (labels, cluster_centers, inertia)
        """
        if features.size == 0 or len(features) < n_clusters:
            return np.array([]), np.array([]), 0.0

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Fit K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(features_scaled)

        # Transform centers back to original scale
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        return labels, centers, kmeans.inertia_

    def label_archetypes(
        self,
        cluster_centers: np.ndarray,
        num_fields: int
    ) -> List[str]:
        """Generate human-readable archetype labels from cluster centers.

        Args:
            cluster_centers: Cluster centroids (n_clusters, n_features)
            num_fields: Number of allocation fields

        Returns:
            List of archetype names
        """
        labels = []

        for center in cluster_centers:
            if len(center) < num_fields + 2:
                labels.append("Unknown")
                continue

            # Extract key features from center
            props = center[:num_fields]
            hhi = center[num_fields] if len(center) > num_fields else 0
            norm_entropy = center[num_fields + 1] if len(center) > num_fields + 1 else 0

            # Classify based on concentration
            if hhi > 0.5:
                # Find which field is concentrated
                max_idx = np.argmax(props)
                labels.append(f"Concentrated (F{max_idx + 1})")
            elif hhi < 0.25:
                labels.append("Uniform/Spread")
            elif norm_entropy > 0.8:
                labels.append("Diversified")
            else:
                # Check for asymmetry patterns
                first_half = sum(props[:len(props)//2])
                second_half = sum(props[len(props)//2:])
                if first_half > second_half + 0.2:
                    labels.append("Front-Heavy")
                elif second_half > first_half + 0.2:
                    labels.append("Back-Heavy")
                else:
                    labels.append("Hedged")

        return labels

    def analyze_allocations(
        self,
        allocations: List[List[float]],
        budget: float = 100.0,
        n_clusters: int = 4,
        dim_reduction: str = 'pca'
    ) -> ClusterResult:
        """Full clustering analysis pipeline.

        Args:
            allocations: List of allocation vectors
            budget: Allocation budget
            n_clusters: Number of clusters
            dim_reduction: 'pca' or 'tsne'

        Returns:
            ClusterResult with all analysis
        """
        # Extract features
        features = self.extract_features(allocations, budget)

        if features.size == 0:
            return ClusterResult(
                cluster_labels=[],
                archetype_names=[],
                cluster_centers=np.array([]),
                features_2d=np.array([]),
                inertia=0.0,
                silhouette_score=0.0,
            )

        # Adjust n_clusters if needed
        n_samples = len(features)
        n_clusters = min(n_clusters, n_samples)

        if n_clusters < 2:
            # Can't cluster with less than 2 clusters
            features_2d = self.reduce_dimensions(features, dim_reduction)
            return ClusterResult(
                cluster_labels=[0] * n_samples,
                archetype_names=["Single Cluster"],
                cluster_centers=np.mean(features, axis=0, keepdims=True),
                features_2d=features_2d,
                inertia=0.0,
                silhouette_score=0.0,
            )

        # Cluster
        labels, centers, inertia = self.cluster_strategies(features, n_clusters)

        # Reduce dimensions for visualization
        features_2d = self.reduce_dimensions(features, dim_reduction)

        # Label archetypes
        num_fields = len(allocations[0]) if allocations else 0
        archetype_names = self.label_archetypes(centers, num_fields)

        # Calculate silhouette score
        silhouette = 0.0
        if n_clusters > 1 and len(set(labels)) > 1:
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            silhouette = silhouette_score(features_scaled, labels)

        return ClusterResult(
            cluster_labels=list(labels),
            archetype_names=archetype_names,
            cluster_centers=centers,
            features_2d=features_2d,
            inertia=inertia,
            silhouette_score=silhouette,
        )

    def get_archetype_summaries(
        self,
        allocations: List[List[float]],
        cluster_result: ClusterResult,
        budget: float = 100.0
    ) -> List[StrategyArchetype]:
        """Get detailed summaries of each archetype.

        Args:
            allocations: Original allocation vectors
            cluster_result: Result from analyze_allocations
            budget: Allocation budget

        Returns:
            List of StrategyArchetype objects
        """
        from .allocation import AllocationAnalyzer

        analyzer = AllocationAnalyzer()
        archetypes = []

        labels = cluster_result.cluster_labels
        names = cluster_result.archetype_names

        for cluster_id, name in enumerate(names):
            # Get allocations in this cluster
            cluster_allocs = [
                alloc for alloc, label in zip(allocations, labels)
                if label == cluster_id
            ]

            if not cluster_allocs:
                continue

            # Calculate average allocation
            num_fields = len(cluster_allocs[0])
            avg_alloc = [
                sum(a[i] for a in cluster_allocs) / len(cluster_allocs)
                for i in range(num_fields)
            ]

            # Calculate average concentration
            concentrations = [
                analyzer.calculate_hhi(a, budget)
                for a in cluster_allocs
            ]
            avg_concentration = sum(concentrations) / len(concentrations)

            # Generate description
            if avg_concentration > 0.5:
                max_field = avg_alloc.index(max(avg_alloc))
                description = f"Concentrates resources on field {max_field + 1}"
            elif avg_concentration < 0.25:
                description = "Spreads resources evenly across all fields"
            else:
                description = "Moderately balanced allocation strategy"

            archetypes.append(StrategyArchetype(
                name=name,
                description=description,
                avg_concentration=avg_concentration,
                avg_allocation=avg_alloc,
                count=len(cluster_allocs),
            ))

        return archetypes


def find_optimal_clusters(
    features: np.ndarray,
    max_clusters: int = 8,
    random_state: int = 42
) -> int:
    """Find optimal number of clusters using elbow method + silhouette.

    Args:
        features: Feature matrix
        max_clusters: Maximum clusters to try
        random_state: Random seed

    Returns:
        Optimal number of clusters
    """
    if len(features) < 3:
        return min(2, len(features))

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    max_k = min(max_clusters, len(features) - 1)
    if max_k < 2:
        return 2

    silhouettes = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(features_scaled, labels)
            silhouettes.append((k, score))

    if not silhouettes:
        return 2

    # Return k with highest silhouette score
    best_k = max(silhouettes, key=lambda x: x[1])[0]
    return best_k
