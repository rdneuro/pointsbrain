"""Point-cloud I/O: volumetric labels, raw T1w and TractSeg masks -> point clouds.

- :func:`labels_to_pointcloud` -- voxel centres of a label in world space.
- :func:`raw_to_pointcloud` -- raw T1w -> skull-strip -> segment -> points
  (containers from :mod:`spectralbrain.io.preprocess`).
- :func:`load_tractseg_bundle` / :func:`load_tractseg` -- TractSeg bundle
  masks as :class:`~pointsbrain.clouds.BrainPointCloud`.

Everything else (format detection, NIfTI/GIfTI/FreeSurfer loaders, meshing,
parcellation, cohorts) is SpectralBrain's :mod:`spectralbrain.io`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from spectralbrain.io.preprocess import segment, skull_strip
from spectralbrain.io.tractseg import _DEFAULT_SUBDIR, discover_tractseg_bundles
from spectralbrain.runtime import (
    PathLike,
    Points,
    get_logger,
)

# ======================================================================
# VOLUMETRIC LABEL -> POINT CLOUD
# ======================================================================


def labels_to_pointcloud(
    label_volume: np.ndarray,
    affine: np.ndarray,
    label_id: int,
    *,
    jitter: bool = False,
    jitter_scale: float = 0.25,
    seed: int | None = None,
) -> Points:
    """Extract a point cloud from a volumetric segmentation.

    Given a 3D integer label volume (e.g. FreeSurfer ``aseg.mgz``)
    and a target label ID, returns the world-space (RAS) coordinates
    of all voxels with that label.

    This is the pathway 3 from the SpectralBrain I/O diagram:
    volumetric segmentation -> point cloud -> spectral descriptors.

    Parameters
    ----------
    label_volume : ndarray, shape (X, Y, Z)
        Integer label volume.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine matrix.
    label_id : int
        Target label (e.g. 17 for left hippocampus in aseg).
    jitter : bool
        Add sub-voxel Gaussian jitter to break the grid pattern.
        Useful for point-cloud Laplacian estimation, where a
        regular grid causes degenerate eigenvalues.
    jitter_scale : float
        Standard deviation of the jitter in voxel units (default
        0.25 -- i.e. +/-0.25 voxels).
    seed : int, optional
        RNG seed for reproducible jitter.

    Returns
    -------
    points : ndarray, shape (N, 3)
        World-space coordinates of the extracted voxels.

    Raises
    ------
    ValueError
        If *label_id* is not found in the volume.

    Examples
    --------
    >>> data, affine = sb.io.load_nifti("aseg.mgz")
    >>> hippo_L = pb.labels_to_pointcloud(data, affine, label_id=17)
    >>> hippo_L.shape
    (4231, 3)
    """
    label_volume = np.asarray(label_volume)
    affine = np.asarray(affine, dtype=np.float64)

    if label_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {label_volume.shape}")

    if np.issubdtype(label_volume.dtype, np.floating):
        # Label maps stored with scl_slope/inter (or resampled) load as float;
        # exact equality would silently miss voxels such as 16.9999997.
        mask = np.isclose(label_volume, label_id, rtol=0.0, atol=1e-3)
    else:
        mask = label_volume == label_id
    if not mask.any():
        raise ValueError(
            f"Label {label_id} not found in volume.  "
            f"Unique labels (first 20): {np.unique(label_volume)[:20]!r}..."
        )

    # Voxel indices -> (N, 3) int array.
    ijk = np.argwhere(mask).astype(np.float64)  # (N, 3)

    if jitter:
        rng = np.random.default_rng(seed)
        ijk += rng.normal(scale=jitter_scale, size=ijk.shape)

    # Apply affine: world = affine @ [i, j, k, 1]^T
    ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
    ijk_h = np.hstack([ijk, ones])  # (N, 4)
    xyz = (affine @ ijk_h.T).T[:, :3]  # (N, 3)

    logger.info("Extracted %d points for label %d", xyz.shape[0], label_id)
    return xyz


# ======================================================================
# RAW T1w -> POINT CLOUD
# ======================================================================


def raw_to_pointcloud(
    input_path: PathLike,
    label_id: int,
    *,
    output_dir: PathLike | None = None,
    gpu: bool | None = None,
    jitter: bool = True,
    jitter_scale: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    """End-to-end: raw T1w -> skull-strip -> segment -> point cloud.

    Chains :func:`skull_strip`, :func:`segment`, and
    :func:`pointsbrain.io.labels_to_pointcloud` into a single call.

    Parameters
    ----------
    input_path : PathLike
        Raw T1w NIfTI.
    label_id : int
        Target structure label (e.g. 17 for left hippocampus).
    output_dir : PathLike, optional
        Working directory for intermediate files.
    gpu : bool or None
        GPU passthrough.
    jitter : bool
        Add sub-voxel jitter to the point cloud.
    jitter_scale : float
        Jitter magnitude in voxel units.
    seed : int, optional
        RNG seed.

    Returns
    -------
    points : ndarray, shape (N, 3)
        World-space point cloud for the target structure.

    Examples
    --------
    >>> hippo = raw_to_pointcloud("sub-01_T1w.nii.gz", label_id=17)
    >>> hippo.shape
    (4231, 3)
    """
    from spectralbrain.io.loaders import load_nifti

    inp = Path(input_path)
    wdir = Path(output_dir) if output_dir else inp.parent
    wdir.mkdir(parents=True, exist_ok=True)

    stem = inp.name.split(".")[0]  # "sub-01_T1w.nii.gz" -> "sub-01_T1w"

    # Step 1: skull-strip
    brain = skull_strip(inp, wdir / f"{stem}_brain.nii.gz", gpu=gpu)

    # Step 2: segment
    seg = segment(brain, wdir / f"{stem}_synthseg.nii.gz", gpu=gpu)

    # Step 3: extract point cloud
    data, affine = load_nifti(seg)
    points = labels_to_pointcloud(
        data,
        affine,
        label_id,
        jitter=jitter,
        jitter_scale=jitter_scale,
        seed=seed,
    )
    return points


logger = get_logger(__name__)


# ======================================================================
# TRACTSEG BUNDLE MASKS AS POINT CLOUDS
# ======================================================================


def load_tractseg_bundle(
    mask_path: PathLike,
    *,
    level: float = 0.5,
    jitter: bool = False,
    jitter_scale: float = 0.25,
    seed: int | None = None,
) -> Any:
    """Load one TractSeg bundle mask as a point cloud of its voxel centres.

    Parameters
    ----------
    mask_path : PathLike
        A binary (or probabilistic) bundle mask NIfTI.
    level : float
        Threshold separating inside/outside the bundle (default ``0.5``;
        appropriate for binary masks and TractSeg probability maps).
    jitter, jitter_scale, seed :
        Optional sub-voxel jitter to break the regular grid (helps the
        point-cloud Laplacian estimate).

    Returns
    -------
    BrainPointCloud
        With ``metadata["bundle"]``, ``metadata["source"]`` and
        ``metadata["n_voxels"]`` set. For the isosurface MESH of the same
        mask use :func:`spectralbrain.io.load_tractseg_bundle`.
    """
    from spectralbrain.io.loaders import load_nifti

    from pointsbrain.clouds import BrainPointCloud

    path = Path(mask_path)
    vol, affine = load_nifti(path)
    vol = np.asarray(vol)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(
            f"{path.name}: expected a 3-D bundle mask, got shape {vol.shape} "
            "(multi-bundle 4-D outputs must be split per bundle first)."
        )
    binary = (vol > level).astype(np.uint8)
    if binary.sum() == 0:
        raise ValueError(f"Empty mask (no voxels above {level}) in {path.name}")

    meta = {"bundle": path.name.split(".")[0], "source": str(path)}
    pts = labels_to_pointcloud(
        binary,
        affine,
        label_id=1,
        jitter=jitter,
        jitter_scale=jitter_scale,
        seed=seed,
    )
    return BrainPointCloud(pts, metadata={**meta, "n_voxels": int(binary.sum())})


def load_tractseg(
    tractseg_dir: PathLike,
    *,
    bundles: list[str] | None = None,
    subdir: str | None = _DEFAULT_SUBDIR,
    return_failed: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, str]]:
    """Load all (or selected) bundles of one subject's TractSeg output as point clouds.

    Parameters
    ----------
    tractseg_dir : PathLike
        TractSeg output directory for a subject.
    bundles : list of str, optional
        Restrict to these bundle names. Defaults to all masks found.
    subdir : str, optional
        Mask subdirectory (default ``"bundle_segmentations"``).
    return_failed : bool
        If True, also return ``{bundle_name: error message}`` for the bundles
        that were skipped.
    **kwargs
        Forwarded to :func:`load_tractseg_bundle` (``level``, ``jitter``, ...).

    Returns
    -------
    dict of {bundle_name: BrainPointCloud}
        Bundles that fail to load (e.g. empty masks) are logged and skipped;
        programming errors (``TypeError`` etc.) are re-raised.
    """
    from spectralbrain.io.group import PROGRAMMING_ERRORS

    files = discover_tractseg_bundles(tractseg_dir, bundles=bundles, subdir=subdir)
    out: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for name, path in files.items():
        try:
            out[name] = load_tractseg_bundle(path, **kwargs)
        except PROGRAMMING_ERRORS:
            raise
        except Exception as exc:
            logger.error("FAILED %s: %s", name, exc)
            failed[name] = str(exc)
    if failed:
        logger.warning("Skipped %d TractSeg bundle(s): %s", len(failed), ", ".join(failed))
    logger.info("Loaded %d/%d TractSeg bundles as point clouds.", len(out), len(files))
    if return_failed:
        return out, failed
    return out


__all__: list[str] = [
    "labels_to_pointcloud",
    "load_tractseg",
    "load_tractseg_bundle",
    "raw_to_pointcloud",
]
