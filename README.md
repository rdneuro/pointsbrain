# pointsbrain

Spectral analysis of brain structures as **point clouds** -- the point-cloud
sibling of [SpectralBrain](https://github.com/rdneuro/spectralbrain).

SpectralBrain is mesh-only. pointsbrain owns everything specific to unstructured
point sets and nothing else:

- **Construction** -- voxel centres of a label (`labels_to_pointcloud`,
  `BrainPointCloud.from_volume`), FreeSurfer segmentations, atlas volumes,
  surface vertex sets, TractSeg bundle masks (`load_tractseg`), tractography
  streamlines, raw T1w through containers (`raw_to_pointcloud`).
- **Point-cloud Laplacians** -- kNN graph, Belkin-Niyogi heat kernel and the
  Sharp-Crane robust (tufted) Laplacian.
- **Point-set geometry** -- farthest-point sampling, kNN adjacency, density,
  outlier removal, normals, local curvature, clustering.
- **Synthetic clouds** and **point-cloud renderers** (vedo / Open3D).

`BrainPointCloud.decompose()` returns SpectralBrain's own `SpectralDecomposition`,
and `BrainPointCloud` satisfies SpectralBrain's `GeometricObject` protocol. From
that point on, every SpectralBrain descriptor, distance, statistic and figure
applies unchanged:

```python
import pointsbrain as pb
import spectralbrain as sb

clouds = pb.load_tractseg("sub-01/tractseg_output")        # {bundle: BrainPointCloud}
dec = clouds["CST_left"].decompose(k=60)                    # robust Laplacian
hks = sb.compute_hks(dec, n_times=100)                      # SpectralBrain from here on
dna = sb.compute_shapedna(dec, normalize="area")
```

## Installation

```bash
pip install pointsbrain            # pulls spectralbrain>=0.1
pip install "pointsbrain[viz]"     # vedo + Open3D renderers
```

Requires Python 3.11-3.12.

## Layout

```
pointsbrain/
  clouds.py      BrainPointCloud and its Laplacians
  geometry.py    farthest-point sampling, kNN adjacency, density, outliers
  io.py          labels / raw T1w / TractSeg masks -> point clouds
  synthetic.py   SyntheticPointCloud, example_point_cloud
  viz.py         point-cloud renderers (vedo, Open3D)
```

## Tutorials

`tutorials/01_pointclouds_and_tracts.ipynb` -- point clouds and white-matter
tracts: the robust Laplacian, HKS/WKS on bundles, BKS/iBKS and their numerical
limits, bundles compared by their spectra. It runs on the TractSeg subject
bundled in `tutorials/data/` and is a companion to the SpectralBrain series.

## Origin

The code was part of SpectralBrain up to 0.0.7 (`core.pointclouds`,
`viz.geometry.points`, the point-cloud paths of `io` and `utils`) and moved here
when SpectralBrain 0.1.0 became mesh-only. Its regression tests moved with it.

## License

MIT -- see [`LICENSE`](LICENSE).
