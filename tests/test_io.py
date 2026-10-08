"""Point-cloud I/O (regression tests moved from SpectralBrain 0.0.7)."""

from __future__ import annotations

import numpy as np

import pointsbrain as pb


def test_labels_to_pointcloud_float_labels():
    vol = np.zeros((4, 4, 4), np.float32)
    vol[1, 1, 1] = 16.9999997
    pts = pb.labels_to_pointcloud(vol, np.eye(4), 17)
    assert pts.shape == (1, 3)
