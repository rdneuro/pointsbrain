"""Point-cloud geometry and Laplacians (regression tests moved from SpectralBrain 0.0.7)."""

from __future__ import annotations

import numpy as np
import pytest
import spectralbrain as sb

import pointsbrain as pb
from pointsbrain import BrainPointCloud
from pointsbrain.geometry import (
    compute_adjacency_from_knn,
    estimate_point_density,
    farthest_point_sampling,
)


def _fib_sphere(n: int = 1500, r: float = 1.0) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def test_pointcloud_sigma_never_leaks_into_eigsh():
    pc = BrainPointCloud(_fib_sphere(600))
    with pytest.warns(DeprecationWarning):
        a = pc.decompose(k=6, laplacian_method="knn", sigma=0.3)
    with pytest.warns(DeprecationWarning):
        b = pc.decompose(k=6, laplacian_method="knn", sigma=0.3)
    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, atol=1e-10)
    assert a.eigenvalues[0] < 1e-8  # smallest eigenvalues, not ones near a shift


def test_pointcloud_cache_keyed_on_method_and_params():
    pc = BrainPointCloud(_fib_sphere(600))
    knn = pc.decompose(k=5, laplacian_method="knn")
    bn = pc.decompose(k=5, laplacian_method="belkin_niyogi")
    assert bn.metadata["laplacian_method"] == "belkin_niyogi"
    assert not np.allclose(knn.eigenvalues, bn.eigenvalues)
    narrow = pc.decompose(k=5, laplacian_method="knn", kernel_sigma=0.05)
    assert not np.allclose(knn.eigenvalues, narrow.eigenvalues)
    # explicitly built Laplacian is reused by a bare decompose()
    pc.compute_laplacian("knn", k=12)
    d = pc.decompose(k=5)
    assert d.metadata["laplacian_method"] == "knn"
    assert dict(d.metadata["laplacian_params"])["k"] == 12


def test_pointcloud_eigsh_sigma_alias():
    pc = BrainPointCloud(_fib_sphere(400))
    d = pc.decompose(k=4, laplacian_method="knn", eigsh_sigma=-0.05)
    assert d.eigenvalues[0] < 1e-8


def test_belkin_niyogi_approximates_sphere_spectrum():
    pc = BrainPointCloud(_fib_sphere(2000))
    lam = pc.decompose(k=5, laplacian_method="belkin_niyogi", area=4 * np.pi).eigenvalues
    np.testing.assert_allclose(lam[1:4], 2.0, rtol=0.05)


def test_knn_adjacency_has_no_self_loops():
    p = np.random.default_rng(0).normal(size=(100, 3))
    A = compute_adjacency_from_knn(p, k=5)
    assert A.diagonal().sum() == 0
    assert (np.asarray((A > 0).sum(axis=1)).ravel() >= 5).all()


def test_point_density_uses_kth_other_neighbour():
    p = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
    dens = estimate_point_density(p, k=1)
    assert dens[0] == pytest.approx(1 / ((4 / 3) * np.pi * 1.0**3))


def test_seed_everything_makes_library_rngs_deterministic():
    from spectralbrain.runtime import set_global_seed

    p = np.random.default_rng(1).normal(size=(200, 3))
    try:
        sb.seed_everything(7)
        a = farthest_point_sampling(p, 10)[1]
        sb.seed_everything(7)
        b = farthest_point_sampling(p, 10)[1]
        np.testing.assert_array_equal(a, b)
    finally:
        set_global_seed(None)


# -- integration with SpectralBrain ------------------------------------


def test_pointcloud_satisfies_spectralbrain_protocol():
    from spectralbrain.core.base import GeometricObject, SpectralDecomposition

    pc = BrainPointCloud(_fib_sphere(500))
    assert isinstance(pc, GeometricObject)
    assert isinstance(pc.decompose(k=6, laplacian_method="knn"), SpectralDecomposition)


def test_spectralbrain_descriptors_run_on_a_cloud_decomposition():
    pc = BrainPointCloud(_fib_sphere(800, r=10.0))
    dec = pc.decompose(k=20, laplacian_method="belkin_niyogi", area=4 * np.pi * 100)
    hks = sb.compute_hks(dec, n_times=8)
    wks = sb.compute_wks(dec, n_energies=8)
    dna = sb.compute_shapedna(dec)
    assert hks.shape == (800, 8) and wks.shape == (800, 8)
    assert np.isfinite(hks).all() and np.isfinite(wks).all() and np.isfinite(dna).all()


def test_robust_laplacian_matches_mesh_spectrum_on_a_sphere():
    pytest.importorskip("robust_laplacian")
    v, f = sb.utils.example_sphere(n_lat=30, n_lon=60, radius=10.0)
    mesh_l = sb.BrainMesh(v, f).decompose(k=6).eigenvalues
    pc_l = BrainPointCloud(v).decompose(k=6, laplacian_method="robust").eigenvalues
    np.testing.assert_allclose(pc_l[1:4], mesh_l[1:4], rtol=0.05)


def test_synthetic_and_example_clouds():
    for shape in ("sphere", "ellipsoid", "blob", "multi_cluster"):
        p = pb.example_point_cloud(n_points=300, shape=shape, seed=0)
        assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
    with pytest.raises(ValueError):
        pb.example_point_cloud(shape="torus")
