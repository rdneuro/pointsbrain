"""Point-set geometry: subsampling, kNN graphs, density and outliers.

Coordinate-only helpers for unstructured point sets (no faces). The generic
vertex-array geometry (centroids, PCA alignment, Procrustes, neighbour
queries, Hausdorff / Chamfer distances, convex hulls) is reused from
:mod:`spectralbrain.core.base`.
"""

from __future__ import annotations

import numpy as np
from spectralbrain.core.base import knn_search
from spectralbrain.runtime import (
    Points,
    ScalarMap,
    SparseMatrix,
    get_logger,
    resolve_seed,
)

logger = get_logger(__name__)


def farthest_point_sampling(
    points: Points,
    n_samples: int,
    *,
    seed: int | None = None,
) -> tuple[Points, np.ndarray]:
    """Farthest-point sampling (FPS) for uniform subsampling.

    Iteratively selects the point farthest from the current set,
    producing an approximately uniform subset.  Essential for
    standardising point-cloud density from volumetric segmentations.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    n_samples : int
        Number of points to select.
    seed : int, optional
        RNG seed for the initial point.  ``None`` falls back to the
        library-wide seed set by :func:`~spectralbrain.utils.seed_everything`.

    Returns
    -------
    sampled : ndarray, shape (n_samples, 3)
        Selected points.
    indices : ndarray, shape (n_samples,)
        Indices into *points*.

    Notes
    -----
    Complexity: O(N * n_samples).  For N > 100 k, consider the
    approximate FPS in Open3D or PyTorch3D.
    """
    N = points.shape[0]
    if n_samples >= N:
        return points.copy(), np.arange(N)

    rng = np.random.default_rng(resolve_seed(seed))
    indices = np.zeros(n_samples, dtype=np.int64)
    indices[0] = rng.integers(N)

    # min_dist[i] = distance from point i to the closest selected point.
    min_dist = np.full(N, np.inf, dtype=np.float64)

    for j in range(1, n_samples):
        last = points[indices[j - 1]]  # (3,)
        dist = np.sum((points - last) ** 2, axis=1)  # (N,)
        min_dist = np.minimum(min_dist, dist)
        indices[j] = np.argmax(min_dist)

    return points[indices], indices


def _knn_excluding_self(points: Points, k: int) -> tuple[np.ndarray, np.ndarray]:
    """kNN self-query returning the *k* nearest **other** points.

    Queries ``k + 1`` neighbours and drops the query point itself (which
    also handles exact duplicates robustly by removing the self index
    wherever it appears in the row).
    """
    N = points.shape[0]
    kq = min(k + 1, N)
    distances, indices = knn_search(points, k=kq)
    distances = np.asarray(distances).reshape(N, kq)
    indices = np.asarray(indices).reshape(N, kq)
    self_idx = np.arange(N)[:, None]
    is_self = indices == self_idx
    # Drop exactly one entry per row: the self index if present, else the last.
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), kq - 1)
    keep = np.ones((N, kq), dtype=bool)
    keep[np.arange(N), drop] = False
    k_eff = kq - 1
    return distances[keep].reshape(N, k_eff), indices[keep].reshape(N, k_eff)


def compute_adjacency_from_knn(
    points: Points,
    k: int = 20,
    *,
    symmetric: bool = True,
) -> SparseMatrix:
    """Build a kNN adjacency matrix (binary or weighted).

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        Number of neighbours.
    symmetric : bool
        Symmetrise the adjacency (``A = max(A, A.T)``).

    Returns
    -------
    SparseMatrix, shape (N, N)
        Binary adjacency (no self-loops; each point linked to its *k*
        nearest *other* points).
    """
    import scipy.sparse as sp

    _distances, indices = _knn_excluding_self(points, k)
    N = points.shape[0]
    k_eff = indices.shape[1]

    rows = np.repeat(np.arange(N), k_eff)
    cols = indices.ravel()
    data = np.ones(N * k_eff, dtype=np.float64)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A.setdiag(0)
    A.eliminate_zeros()
    if symmetric:
        A = A.maximum(A.T)
    return A


def estimate_point_density(
    points: Points,
    k: int = 6,
) -> ScalarMap:
    """Estimate local point density via k-NN distance.

    Density at each point is approximated as ``k / V_k`` where
    ``V_k = (4/3)pi r_k^3`` and ``r_k`` is the distance to the k-th
    nearest neighbour.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        Neighbour count for density estimation.

    Returns
    -------
    density : ndarray, shape (N,)
        Points per mm^3 (approximate).
    """
    distances, _ = _knn_excluding_self(points, k)
    r_k = distances[:, -1]  # (N,) distance to k-th NN (self excluded)
    r_k = np.clip(r_k, 1e-10, None)  # avoid div by zero
    volume_k = (4 / 3) * np.pi * r_k**3
    return k / volume_k


def detect_density_outliers(
    points: Points,
    k: int = 6,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """Flag points with anomalously low or high local density.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        Neighbour count.
    threshold_sigma : float
        Number of standard deviations from the mean log-density
        to flag as outlier.

    Returns
    -------
    outlier_mask : ndarray, shape (N,), bool
        True for outlier points.
    """
    density = estimate_point_density(points, k=k)
    log_d = np.log(density + 1e-30)
    z = (log_d - log_d.mean()) / (log_d.std() + 1e-30)
    return np.abs(z) > threshold_sigma


__all__: list[str] = [
    "compute_adjacency_from_knn",
    "detect_density_outliers",
    "estimate_point_density",
    "farthest_point_sampling",
]
