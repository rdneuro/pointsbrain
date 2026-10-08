"""Point-cloud renderers: import-level checks (rendering needs vedo / open3d)."""

from pointsbrain import viz


def test_resolve_cmap_z_score():
    assert viz._resolve_cmap("z_score", None) == "RdBu_r"


def test_public_renderers_present():
    for name in (
        "plot_point_cloud",
        "plot_mls_reconstruction",
        "plot_clusters",
        "plot_point_cloud_panel",
        "plot_warp",
        "plot_voronoi",
        "plot_point_cloud_o3d",
    ):
        assert callable(getattr(viz, name))
