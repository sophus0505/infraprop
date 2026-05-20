# utils/plot_map.py

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from cartopy.io import shapereader as shpreader

EXTENT_UKRAINE = [22, 41, 44, 53]  # lon_min, lon_max, lat_min, lat_max
LAND, OCEAN = "#EEF2F5", "#F6F8FB"
# UK_FILL, UK_EDGE = "#E1E7EC", "#7B8794"
UK_FILL, UK_EDGE = "k", "k"


def ne_records(name, res="10m", cat="cultural"):
    return shpreader.Reader(
        shpreader.natural_earth(resolution=res, category=cat, name=name)
    ).records()


def add_geom(ax, geom, *, face="none", edge="none", lw=0.0, alpha=1.0, zorder=None):
    kwargs = dict(facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha)
    if zorder is not None:
        kwargs["zorder"] = zorder
    ax.add_geometries([geom], ccrs.PlateCarree(), **kwargs)


def draw_base(ax):
    ax.set_facecolor(OCEAN)

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m", facecolor=LAND, edgecolor="none"
        ),
        zorder=0,
    )

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "lakes", "10m", facecolor=OCEAN, edgecolor="none"
        ),
        zorder=1,
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        linewidth=0.6,
        edgecolor="#9AA4AF",
        zorder=3,
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        linewidth=0.6,
        edgecolor="#9AA4AF",
        zorder=3,
    )

    gl = ax.gridlines(
        ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="#C7D0DB",
        alpha=0.7,
        linestyle=(0, (4, 4)),
        zorder=2,
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 12}

    ax.tick_params()


def highlight_ukraine(ax, add_fill, lw=0.8):
    # Fill Ukraine polygon
    for r in ne_records("admin_0_countries"):
        if (r.attributes.get("ADMIN") or r.attributes.get("SOVEREIGNT")) == "Ukraine":
            if add_fill:
                add_geom(ax, r.geometry, face=UK_FILL, zorder=4)
            add_geom(ax, r.geometry, edge=UK_EDGE, lw=lw, zorder=5, alpha=0.8)
            break

    # If your Natural Earth version separates Crimea as "disputed", fill it too
    for r in ne_records("admin_0_disputed_areas"):
        if (
            "crimea"
            in (r.attributes.get("NAME") or r.attributes.get("name") or "").lower()
        ):
            if add_fill:
                add_geom(ax, r.geometry, face=UK_FILL, zorder=4)
            add_geom(ax, r.geometry, edge=UK_EDGE, lw=lw, zorder=5, alpha=0.8)


def draw_regions(ax):
    for r in ne_records("admin_1_states_provinces_lines"):
        if (
            r.attributes.get("admin")
            or r.attributes.get("ADM0_NAME")
            or r.attributes.get("geonunit")
        ) == "Ukraine":
            add_geom(ax, r.geometry, edge="#8A96A3", lw=0.55, alpha=0.85, zorder=6)


def draw_stations(ax, extent, station_type="is"):
    if station_type == "is":
        st_names = ["MAAG", "GRDI", "KPDI"]
        latlons = [(50.7014, 29.2301), (50.59925, 29.447125), (48.563745, 26.45660)]
    elif station_type == "seismic": 
        st_names = ["Malyn"]
        latlons = [(50.6573, 29.2057)]
    else: 
        raise Warning(f"{station_type} not one of ['is', 'seismic']")
    halo = [pe.withStroke(linewidth=3, foreground="white", alpha=0.9)]

    for i, st_n in enumerate(st_names):
        lat, lon = latlons[i]

        if (extent[0] < lon < extent[1]) and (extent[2] < lat < extent[3]):
            ax.scatter(
                lon,
                lat,
                marker="v",
                transform=ccrs.PlateCarree(),
                s=100,
                zorder=1000,
                path_effects=halo,
            )
            dlon = 0.3
            dlat = -0.3
            # dlon = 0.2
            # dlat = -0.15
            if st_n == "MAAG":
                dlon = -1.25
                # dlon = -0.8
            ax.text(
                lon + dlon,
                lat + dlat,
                st_n,
                transform=ccrs.PlateCarree(),
                fontsize=14,
                path_effects=halo,
                zorder=1001,
            )


def add_map_ukraine(
    ax=None,
    *,
    extent=None,
    projection=None,
    add_fill=True,
    add_regions=True,
    add_stations=True,
    ukr_lw=0.8,
    station_type="is",
):
    """
    Add a Ukraine basemap to an axis so data can be plotted on top later.
    Returns (fig, ax).
    """

    if extent is None:
        extent = EXTENT_UKRAINE

    if ax is None:
        if projection is None:
            projection = ccrs.Mercator()

        fig, ax = plt.subplots(
            figsize=(9, 6),
            subplot_kw={"projection": projection},
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    draw_base(ax)
    highlight_ukraine(ax, add_fill, ukr_lw)

    if add_regions:
        draw_regions(ax)

    if add_stations:
        draw_stations(ax, extent, station_type)

    return fig, ax

def add_general_map(
    src_lat,
    src_lon,
    max_dist_km,
    ax=None,
    *,
    projection=None,
):
    """
    Add a simple map centered at (src_lat, src_lon) with extent based on max_dist_km.
    
    Returns (fig, ax)
    """
    import numpy as np

    # --- compute extent ---
    dlat = max_dist_km / 111.0
    dlon = max_dist_km / (111.0 * np.cos(np.radians(src_lat)))

    extent = [
        src_lon - dlon,
        src_lon + dlon,
        src_lat - dlat,
        src_lat + dlat,
    ]

    # --- create axis if needed ---
    if ax is None:
        if projection is None:
            projection = ccrs.Mercator()

        fig, ax = plt.subplots(
            figsize=(9, 6),
            subplot_kw={"projection": projection},
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    # --- apply extent ---
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # --- draw base map ---
    draw_base(ax)

    # --- mark source location ---
    ax.scatter(
        src_lon,
        src_lat,
        c="red",
        s=80,
        marker="*",
        transform=ccrs.PlateCarree(),
        zorder=1000,
    )

    return fig, ax
if __name__ == "__main__":
    import os 
    import pandas as pd 
    import numpy as np 
    import cartopy.crs as ccrs
    from datetime import timedelta 
    from obspy import UTCDateTime
    from ismonpy.detection import detection
    from ismonpy.association import association

    station = "MAAG"
    start_date = UTCDateTime("2022-03-01T00:00:00")
    end_date = start_date + timedelta(hours=6)

    detector = detection.InfraPyDetector(station)
    detector.set_station_info(station, start_date, end_date)
    ass = association.Associator(detector)

    dir_events = "/staff/sophus/Documents/norad_ukr/infrapy_testing/data/dataset_ML_validated/"
    seismic_path = os.path.join(dir_events, "events_seismic_feb_march.csv")
    acoustic_path = os.path.join(dir_events, "events_acoustic_feb_march.csv")
    
    seismic_events = ass.load_events(plot=False, return_all=True, file_path=seismic_path)
    infrasound_events = ass.load_events(plot=False, return_all=True, file_path=acoustic_path)
    seismic_events = seismic_events[seismic_events["has_IS"] == False]
    zoom_extent = [28, 32.3, 50, 52]
    fig, ax = add_map_ukraine(extent=zoom_extent, add_fill=False)
    ax.scatter(seismic_events["longitude"].to_numpy(), 
               seismic_events["latitude"].to_numpy(),
               c = "#ad233e",
               alpha=0.75,
               s = 20,
               label="Seismic events without IS",
               transform = ccrs.PlateCarree())
    ax.scatter(infrasound_events["longitude"].to_numpy(), 
               infrasound_events["latitude"].to_numpy(),
               c = "#00335e",
               alpha=0.75,
               s = 50,
               label="Seismic events with IS",
               transform = ccrs.PlateCarree())
    # plt.legend(fontsize=14, markerscale=2)

    plt.savefig("/staff/sophus/Documents/ismonpy/test_scripts/figures/ben_presentation_figs/ukraine_events_zoom.png", transparent=True)
    plt.show()
