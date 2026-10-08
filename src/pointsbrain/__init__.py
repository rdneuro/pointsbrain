"""pointsbrain -- spectral analysis of brain structures as point clouds.

Sibling of SpectralBrain (which is mesh-only). pointsbrain owns the
point-cloud geometry -- construction from label volumes, raw T1w, TractSeg
masks or surface vertices; point-cloud Laplacians (kNN, Belkin-Niyogi,
Sharp-Crane robust); subsampling, density and outlier handling -- and hands a
:class:`spectralbrain.core.base.SpectralDecomposition` to SpectralBrain, which
supplies every descriptor, distance, statistic and figure::

    import pointsbrain as pb
    import spectralbrain as sb

    pc = pb.BrainPointCloud.from_volume(aseg, affine, label_id=17)
    dec = pc.decompose(k=100)             # robust / kNN / Belkin-Niyogi Laplacian
    hks = sb.compute_hks(dec, n_times=50)  # SpectralBrain from here on
"""

from pointsbrain.clouds import BrainPointCloud
from pointsbrain.geometry import (
    compute_adjacency_from_knn,
    detect_density_outliers,
    estimate_point_density,
    farthest_point_sampling,
)
from pointsbrain.io import (
    labels_to_pointcloud,
    load_tractseg,
    load_tractseg_bundle,
    raw_to_pointcloud,
)
from pointsbrain.synthetic import SyntheticPointCloud, example_point_cloud

__version__ = "0.1.0"

__all__ = [
    "BrainPointCloud",
    "SyntheticPointCloud",
    "__version__",
    "compute_adjacency_from_knn",
    "detect_density_outliers",
    "estimate_point_density",
    "example_point_cloud",
    "farthest_point_sampling",
    "labels_to_pointcloud",
    "load_tractseg",
    "load_tractseg_bundle",
    "raw_to_pointcloud",
]
