"""Synthetic point clouds for tests, demos and surrogate studies.

:class:`SyntheticPointCloud` draws spheres, ellipsoids, blobs, multi-cluster
sets or reference-matched clouds; :func:`example_point_cloud` is the quick
one-liner. Noise distributions come from
:mod:`spectralbrain.statistics.surrogates` (22 parametric families or a
reference-matched empirical distribution).
"""

from __future__ import annotations

import numpy as np
from spectralbrain.statistics.surrogates import _match_distribution, _sample_distribution


class SyntheticPointCloud:
    """Generate synthetic point clouds (sphere, ellipsoid, blob, multi-cluster).

    Parameters
    ----------
    reference : ndarray, optional
    distribution, dist_params, seed : as above.
    """

    def __init__(self, reference=None, distribution=None, dist_params=None, seed=None):
        """Initialise the point cloud generator.

        Parameters
        ----------
        reference : ndarray, optional
            Reference point cloud for distribution matching.
        distribution : str, optional
            Noise distribution (default ``"normal"``).
        dist_params : dict, optional
            Distribution-specific parameters.
        seed : int, optional
            RNG seed.
        """
        self.reference = reference
        self.distribution = distribution or "normal"
        self.dist_params = dist_params or {}
        self.rng = np.random.default_rng(seed)

    def sphere(self, n_points=1000, radius=50.0, noise=0.5):
        """Generate random points on a sphere surface.

        Parameters
        ----------
        n_points : int
        radius : float
        noise : float
            Gaussian jitter magnitude.

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        pts = _random_sphere_points(n_points, radius, self.rng)
        if noise > 0:
            pts += self._noise_3d(n_points) * noise
        return pts

    def ellipsoid(self, n_points=1000, radii=(50, 30, 20), noise=0.5):
        """Generate random points on an ellipsoid surface.

        Parameters
        ----------
        n_points : int
        radii : tuple of float
            Semi-axis lengths (a, b, c).
        noise : float

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        pts = _random_sphere_points(n_points, 1.0, self.rng) * np.array(radii)
        if noise > 0:
            pts += self._noise_3d(n_points) * noise
        return pts

    def blob(self, n_points=1000, center=(0, 0, 0), scale=(10, 10, 10)):
        """Generate a Gaussian blob point cloud.

        Parameters
        ----------
        n_points : int
        center : tuple of float
        scale : tuple of float

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        return self.rng.normal(center, scale, (n_points, 3))

    def multi_cluster(self, n_points=1000, n_clusters=5, spread=30.0, cluster_std=5.0):
        """Generate a multi-cluster point cloud.

        Parameters
        ----------
        n_points : int
        n_clusters : int
        spread : float
            Spatial extent for cluster centres.
        cluster_std : float
            Standard deviation within each cluster.

        Returns
        -------
        ndarray, shape (~n_points, 3)
        """
        per = n_points // n_clusters
        centers = self.rng.uniform(-spread, spread, (n_clusters, 3))
        return np.vstack([self.rng.normal(c, cluster_std, (per, 3)) for c in centers])

    def from_reference(self, n_points=None):
        """Generate points matching the reference distribution.

        Parameters
        ----------
        n_points : int, optional
            Number of points (default: same as reference).

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        if self.reference is None:
            raise ValueError("No reference data.")
        return _match_distribution(
            self.reference, (n_points or self.reference.shape[0], 3), self.rng
        )

    def _noise_3d(self, n):
        """Generate 3D noise samples using the configured distribution."""
        if self.reference is not None:
            return _match_distribution(
                self.reference - self.reference.mean(axis=0), (n, 3), self.rng
            )
        return _sample_distribution(
            self.distribution, (n, 3), rng=self.rng, params={**self.dist_params, "scale": 1}
        )


def _random_sphere_points(n, radius, rng):
    """Sample *n* uniformly distributed points on a sphere surface.

    Uses the Marsaglia method (uniform z + uniform azimuth).

    Parameters
    ----------
    n : int
        Number of points.
    radius : float
        Sphere radius.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r_xy = np.sqrt(1 - z**2)
    return radius * np.column_stack([r_xy * np.cos(phi), r_xy * np.sin(phi), z])


def example_point_cloud(
    n_points: int = 1000,
    shape: str = "sphere",
    seed: int = 0,
) -> np.ndarray:
    """Quick point cloud for testing.

    Parameters
    ----------
    n_points : int
    shape : str
        ``"sphere"``, ``"ellipsoid"``, ``"blob"``, ``"multi_cluster"``.
    seed : int

    Returns
    -------
    ndarray, shape (n_points, 3)
    """
    gen = SyntheticPointCloud(seed=seed)
    if shape == "sphere":
        return gen.sphere(n_points)
    elif shape == "ellipsoid":
        return gen.ellipsoid(n_points)
    elif shape == "blob":
        return gen.blob(n_points)
    elif shape == "multi_cluster":
        return gen.multi_cluster(n_points)
    raise ValueError(f"Unknown shape: {shape!r}")


__all__: list[str] = [
    "SyntheticPointCloud",
    "example_point_cloud",
]
