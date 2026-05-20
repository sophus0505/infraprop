import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import stripy
from scipy.interpolate import RectBivariateSpline

# from plot_config import *


def get_azimuth_inclination_from_voronoi(N):
    # Create geodesic mesh
    mesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=N)

    # Get Voronoi points (triangle centroids), in radians
    vor_lons, vor_lats = mesh.voronoi_points()

    # Convert to azimuth and inclination (in degrees)
    azimuth = np.degrees(vor_lons) % 360 - 180
    latitudes = np.degrees(vor_lats)  # latitude
    inclination = 90 - latitudes  # zenith angle

    return azimuth, inclination


def get_azimuth_inclination_from_voronoi_positive_inc(N, min_inc=35):
    # Create geodesic mesh for a source at the ground (only positive inclinations)
    # min_inc is the minimum inclination from the vertical (so 35 corresponds to inclinations in [35, 90] --> 0, 55)
    mesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=N)

    vor_lons, vor_lats = mesh.voronoi_points()
    mask = vor_lats >= 0
    vor_lons = vor_lons[mask]
    vor_lats = vor_lats[mask]

    # Convert to azimuth and inclination (in degrees)
    azimuth = np.degrees(vor_lons) % 360 - 180
    latitude = np.degrees(vor_lats)  # latitude
    inclination = 90 - latitude  # zenith angle

    # breakpoint()
    mask = (inclination >= min_inc) & (inclination <= 90.0)

    return azimuth[mask], inclination[mask]


def plot_azimuth_inclination_grid(azimuth, inclination, cmap="viridis"):
    # Plot azimuth vs inclination sampling grid as a 2D scatter plot.
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(azimuth, inclination, c="k", cmap=cmap, s=30)

    ax.set_xlim(0, 360)
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Inclination (degrees)")
    ax.set_title("Azimuth-Inclination Sampling Grid")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def get_lat_lon_from_mesh(N):
    # Create geodesic mesh
    mesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=N)
    lats, lons = np.degrees(mesh.lats), np.degrees(mesh.lons)

    return lats, lons


def plot_lat_lon_grid(lons, lats):
    lons, lats = np.degrees(lons), np.degrees(lats)  # Convert radians to degrees
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(lons, lats, c="k", s=30)

    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    plt.show()


def plot_mesh(N):
    mesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=N)
    proj_map = ccrs.Orthographic(central_latitude=0, central_longitude=0)
    proj_flat = ccrs.PlateCarree(central_longitude=0)

    lons = np.degrees(mesh.lons)
    lats = np.degrees(mesh.lats)

    vor_lons, vor_lats = mesh.voronoi_points()
    vlons = np.degrees(vor_lons)
    vlats = np.degrees(vor_lats)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj_map)

    ax.axis("off")
    ax.triplot(
        lons,
        lats,
        mesh.simplices,
        c="black",
        zorder=1,
        transform=proj_flat,
        linewidth=0.5,
    )
    # ax.scatter(lons, lats, c='black', marker='.', s=0.1, zorder=2, transform=proj_flat)
    ax.scatter(vlons, vlats, marker=".", zorder=3, s=5, transform=proj_flat)

    # # gridlines
    # gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False

    # # relabel latitude as inclination (0° north → 180° south)
    # gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    # gl.yformatter = mticker.FuncFormatter(lambda lat, pos: f"{90 - lat:.0f}°")
    plt.show()


def plot_mesh_flat(N):
    # plot a flash mesh over inclination and azimuth
    azs, incs = get_azimuth_inclination_from_voronoi_positive_inc(N)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(azs, incs, marker=".", zorder=3, s=5)
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Inclination (degrees)")
    plt.show()


if __name__ == "__main__":
    N = 3
    # plot_mesh_N(N)

    # Get azimuth and inclination
    az_inc_dir = "/staff/sophus/Documents/ismonpy/ismonpy/utils/az_incs_voronoi/"
    for i in range(5):
        azimuth, inclination = get_azimuth_inclination_from_voronoi(i)
        fname = os.path.join(az_inc_dir, f"voronoi_{i}.dat")
        np.savetxt(fname, np.column_stack((azimuth, inclination)))

    # Example values for coloring (e.g., turning heights)
    # values = np.random.rand(len(azimuth)) * 100  # Random values for demonstration
    # Plot the azimuth-inclination grid
    # plot_azimuth_inclination_grid(azimuth, inclination)

    plot_mesh_flat(N)
