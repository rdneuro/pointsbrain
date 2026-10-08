"""TractSeg bundle masks as point clouds (moved from SpectralBrain 0.0.7)."""

from __future__ import annotations

import numpy as np
import pytest

import pointsbrain as pb


def _blob(center, radius, shape=(32, 32, 32)):
    """Binary spherical blob volume."""
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    m = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2 <= radius**2
    return m.astype(np.uint8)


def _write_tractseg(root, bundles):
    """Write a minimal TractSeg ``bundle_segmentations`` directory."""
    nib = pytest.importorskip("nibabel")
    seg = root / "tractseg_output" / "bundle_segmentations"
    seg.mkdir(parents=True)
    for name, (center, radius) in bundles.items():
        vol = _blob(center, radius)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(seg / f"{name}.nii.gz"))
    return root / "tractseg_output"


def test_load_tractseg_pointcloud(tmp_path):
    """A bundle mask loads as a BrainPointCloud of its voxels."""
    pytest.importorskip("nibabel")
    ts = _write_tractseg(tmp_path, {"CST_left": ((16, 16, 16), 6)})
    clouds = pb.load_tractseg(ts)
    cloud = clouds["CST_left"]
    assert cloud.points.shape[1] == 3
    assert cloud.n_points > 0
    assert cloud.metadata["bundle"] == "CST_left"


def test_load_tractseg_bundle_selection(tmp_path):
    """Selecting a subset of bundles returns only those."""
    pytest.importorskip("nibabel")
    ts = _write_tractseg(
        tmp_path,
        {"CST_left": ((16, 16, 16), 6), "AF_left": ((10, 20, 16), 5), "CC": ((16, 16, 20), 4)},
    )
    out = pb.load_tractseg(ts, bundles=["CST_left", "CC"])
    assert sorted(out) == ["CC", "CST_left"]


def test_empty_mask_is_skipped(tmp_path):
    """An all-zero mask is logged and skipped, not raised."""
    nib = pytest.importorskip("nibabel")
    seg = tmp_path / "tractseg_output" / "bundle_segmentations"
    seg.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.zeros((16, 16, 16), np.uint8), np.eye(4)), str(seg / "EMPTY.nii.gz")
    )
    out = pb.load_tractseg(tmp_path / "tractseg_output")
    assert out == {}


def test_cloud_and_mesh_of_the_same_bundle(tmp_path):
    """pointsbrain gives the voxel cloud, SpectralBrain the isosurface of the same mask."""
    import spectralbrain as sb

    pytest.importorskip("nibabel")
    ts = _write_tractseg(tmp_path, {"CST_left": ((16, 16, 16), 7)})
    cloud = pb.load_tractseg(ts)["CST_left"]
    mesh = sb.load_tractseg(ts)["CST_left"]
    assert cloud.n_points == cloud.metadata["n_voxels"] > 0
    assert mesh.n_vertices > 0
