# Changelog

## [0.1.0] - Unreleased

Initial release: the point-cloud material split out of SpectralBrain 0.0.7.

- `BrainPointCloud` (from `spectralbrain.core.pointclouds`) with kNN,
  Belkin-Niyogi and robust Laplacians; it returns SpectralBrain's
  `SpectralDecomposition` and satisfies its `GeometricObject` protocol.
- `farthest_point_sampling`, `compute_adjacency_from_knn`,
  `estimate_point_density`, `detect_density_outliers` (from
  `spectralbrain.core.base`).
- `labels_to_pointcloud`, `raw_to_pointcloud` (from `spectralbrain.io`), and
  `load_tractseg` / `load_tractseg_bundle` returning point clouds (the former
  `output="pointcloud"` path).
- `SyntheticPointCloud`, `example_point_cloud` (from
  `spectralbrain.statistics.surrogates` / `spectralbrain.utils.datasets`).
- `pointsbrain.viz` (from `spectralbrain.viz.geometry.points`).
- Tutorial 01 (SpectralBrain's former tutorial 06) and the moved regression
  tests, plus integration tests against SpectralBrain.
