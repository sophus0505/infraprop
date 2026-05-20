import os
import subprocess

import numpy as np 
import matplotlib.pyplot as plt 
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from matplotlib.collections import LineCollection
from obspy import UTCDateTime
from pyproj import Geod
from infraprop.atmos import g2s_profile
from infraprop.raytracing import tau_p  
from infraprop.utils import cmaps, plot_maps

geod = Geod(ellps="WGS84")

def get_az_dist_between_points(src_loc, rcv_loc):
    az, baz, dist_m = geod.inv(src_loc[1], src_loc[0], rcv_loc[1], rcv_loc[0])
    return az, baz, dist_m

def plot_1d_atm(src_loc, rcv_loc, date, atmos_dir, show=True, block=False):
    atm_prof = g2s_profile.G2Sprofile(src_loc[0], src_loc[1], atmos_dir)
    atm_path = atm_prof.get_single_profile(date, atmos_dir)
    # atm_prof.plot_atm_profile(date)

    # calculate the rays from the src to the receiver
    az_deg, baz_deg, dist_m = get_az_dist_between_points(src_loc, rcv_loc)
    taup = tau_p.TauP() 
    taup.plot_atm_profile(atm_path, az_deg=az_deg)

def plot_1d_ray_fan(src_loc, rcv_loc, date, atmos_dir, show=True, block=False): 
    atm_prof = g2s_profile.G2Sprofile(src_loc[0], src_loc[1], atmos_dir)
    atm_path = atm_prof.get_single_profile(date, atmos_dir)
    # atm_prof.plot_atm_profile(date)


    # calculate the rays from the src to the receiver
    az_deg, baz_deg, dist_m = get_az_dist_between_points(src_loc, rcv_loc)

    inc_degs = np.array([80])
    az_degs = np.arange(-180, 180, 1)
    az_degs, inc_degs = np.meshgrid(az_degs, inc_degs)
    az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()

    taup = tau_p.TauP() 
    taup.plot_ray_paths_2d(az_degs, inc_degs, atm_path, lat0=src_loc[0], lon0=src_loc[1], add_ukr=False)


def plot_1d_wavefield_fan(src_loc, rcv_loc, date, atmos_dir, freq, show=True, block=False):
    atm_prof = g2s_profile.G2Sprofile(src_loc[0], src_loc[1], atmos_dir)
    atm_path = atm_prof.get_single_profile(date, atmos_dir)
    az_deg, baz_deg, dist_m = get_az_dist_between_points(src_loc, rcv_loc)

    cmd = ["ePape", 
           "--multiprop", 
           "--azimuth_start", str(-180),
           "--azimuth_end", str(175),
           "--azimuth_step", str(5),
           "--starter", "self",
           "--atmosfile", atm_path, 
           "--freq", str(freq),
           "--maxrange_km", str(dist_m/1000 + 100),
           ]

    subprocess.run(cmd, cwd=atmos_dir)
    wf_data = np.loadtxt(os.path.join(atmos_dir, "tloss_multiprop.pe"))
    
    degs_unique = np.sort(np.unique(wf_data[:, 1]))
    rngs = wf_data[wf_data[:, 1] == degs_unique[0], 0]
    p = (wf_data[:, 2] + 1j*wf_data[:, 3]).reshape(len(degs_unique), len(rngs)).T 
    tl = 20 * np.log10(np.abs(p))

    RADS, RNGS = np.meshgrid(np.radians(degs_unique), rngs)
    azimuths = np.degrees(RADS)
    distances_m = RNGS * 1000
    LONS, LATS, _ = geod.fwd(src_loc[1]*np.ones_like(RADS), src_loc[0]*np.ones_like(RADS), azimuths, distances_m)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True, subplot_kw={"projection": ccrs.Robinson(central_longitude=src_loc[0])})
    # ax = fig.add_subplot(projection=ccrs.Robinson(central_longitude=src_loc[0]))
    plot_maps.add_general_map(src_loc[0], src_loc[1], max_dist_km=np.nanmax(distances_m)/1000 + 100, ax=ax)

    ax.scatter(
        rcv_loc[1],
        rcv_loc[0],
        c="k",
        s=120,
        marker="*",
        transform=ccrs.PlateCarree(),
        zorder=1000,
    )
    pc = ax.pcolormesh(LONS, LATS, tl, cmap=cmc.acton_r, shading="nearest", transform=ccrs.PlateCarree(), vmax=0, vmin=-100)
    fig.colorbar(pc, ax=ax, label="TL [dB]", orientation="horizontal", shrink=0.8)
    
    if show:
        plt.show(block=block)

def plot_1d_rays(src_loc: tuple[float, float], rcv_loc: tuple[float, float], date: UTCDateTime, atmos_dir: str, show = True, block=False):
    # src and recv loc should be (lat, lon)
    # the atmos_dir is used to save the atmospheric profile. 
    print("Plotting rays") 
    atm_prof = g2s_profile.G2Sprofile(src_loc[0], src_loc[1], atmos_dir)
    atm_path = atm_prof.get_single_profile(date, atmos_dir)
    az_deg, baz_deg, dist_m = get_az_dist_between_points(src_loc, rcv_loc)

    # calculate the rays from the src to the receiver
    inc_degs = 90 - np.arange(10, 60, 1)
    az_degs  = np.array([az_deg])
    az_degs, inc_degs = np.meshgrid(az_degs, inc_degs)
    az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()
    taup = tau_p.TauP()
    rays = taup.get_ray_paths(atm_path, az_degs, inc_degs, gw=False, min_duct_height=500)

    x = rays["x"] / 1000
    z_rays = rays["z"] / 1000
    max_x = dist_m / 1000 + 100

    bounce_lengths = np.nanmax(x, axis=1)
    bounce_lengths[bounce_lengths <= 0] = -np.inf
    bounce_lengths[np.isnan(bounce_lengths)] = -np.inf
    n_bounces  = np.ceil(max_x / bounce_lengths).astype(int)
    max_bounces = n_bounces.max()
    offsets  = np.arange(max_bounces)[None, :] * bounce_lengths[:, None]
    x_tiled  = x[:, None, :] + offsets[:, :, None]
    z_tiled  = np.broadcast_to(z_rays[:, None, :], x_tiled.shape)
    needed   = np.arange(max_bounces)[None, :] < n_bounces[:, None]
    mask     = (x_tiled <= max_x) & needed[:, :, None]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True, width_ratios=(0.25, 1))
    axs = axs.flatten()

    segments = []
    colors = []

    order = np.argsort(inc_degs)

    for i in order:
        for j in range(max_bounces):   
            m = mask[i, j, :]
            if not m.any():
                continue

            xij = x_tiled[i, j, :]
            zij = z_tiled[i, j, :]

            seg = np.column_stack([xij[m], zij[m]])  # shape (N, 2)

            if len(seg) < 2:
                continue

            segments.append(seg)
            colors.append(90 - inc_degs[i])  # one value per ray
    
    cmap = cmaps.divergent_cmap().reversed()
    norm = mcolors.Normalize(vmin=90 - inc_degs.max(), vmax=90 - inc_degs.min())
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        alpha=0.75,  
    )

    lc.set_array(np.array(colors))
    lc.set_linewidth(1)

    axs[1].add_collection(lc)
    axs[1].scatter(dist_m/1000, 0, c="r", marker="^", s=100)
    fig.colorbar(lc, label="Inclination [Deg]")

    z, u, v, c = atm_prof.get_atmo_vars(atm_path)
    c_eff = c + u * np.sin(np.radians(az_deg)) + v * np.cos(np.radians(az_deg))
    axs[0].plot(c, z, label="c", c="k", lw=2, alpha=0.7)
    axs[0].plot(c_eff, z, label="c eff.", c="r", lw=2, alpha=0.7)
    axs[0].axvline(c_eff[0], z.min(), z.max(), ls="--", c="k", alpha=0.7)
    axs[0].legend(loc="lower right")

    axs[0].set_xlabel("Sound speed [m/s]")
    axs[0].set_ylabel("Altitude [km]")
        
    axs[0].set_ylim(0, z.max())
    axs[1].set_ylim(0, z.max())
    axs[1].set_xlim(0, max_x)
    axs[1].set_xlabel("Range [km]")
    
    if show:
        plt.show(block=block)

def plot_1d_wavefield(src_loc, rcv_loc, date, atmos_dir, freq, show=False, block=False): 
    atm_prof = g2s_profile.G2Sprofile(src_loc[0], src_loc[1], atmos_dir)
    atm_path = atm_prof.get_single_profile(date, atmos_dir)
    az_deg, baz_deg, dist_m = get_az_dist_between_points(src_loc, rcv_loc)

    cmd = ["ePape", 
           "--singleprop", 
           "--azimuth", str(az_deg), 
           "--starter", "self",
           "--atmosfile", atm_path, 
           "--freq", str(freq),
           "--maxrange_km", str(dist_m/1000 + 100),
           "--write_2d_tloss"
           ]

    subprocess.run(cmd, cwd=atmos_dir)
    wf_data = np.loadtxt(os.path.join(atmos_dir, "tloss_2d.pe"))

    r = np.unique(wf_data[:, 0])
    z = np.unique(wf_data[:, 1])

    R, Z = np.meshgrid(r, z, indexing="ij")

    reP = wf_data[:, 2]
    imP = wf_data[:, 3]

    p = np.sqrt(reP**2 + imP**2).reshape(len(r), len(z))
    p = p /np.max(p)
    p = 20*np.log10(p)

    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True, width_ratios=(0.25, 1))
    axs = axs.flatten()

    pc = axs[1].pcolormesh(R, Z, p, cmap=cmc.acton_r, 
                           shading="nearest", 
                           rasterized=True, 
                           vmin=-125,
                           vmax=0,
                           )
    fig.colorbar(pc, label="TL [dB]")


    z, u, v, c = atm_prof.get_atmo_vars(atm_path)
    c_eff = c + u * np.sin(np.radians(az_deg)) + v * np.cos(np.radians(az_deg))
    axs[0].plot(c, z, label="c", c="k", lw=2, alpha=0.7)
    axs[0].plot(c_eff, z, label="c eff.", c="r", lw=2, alpha=0.7)
    axs[0].axvline(c_eff[0], z.min(), z.max(), ls="--", c="k", alpha=0.7)
    axs[0].legend(loc="lower right")

    axs[0].set_xlabel("Sound speed [m/s]")
    axs[0].set_ylabel("Altitude [km]")
    axs[1].set_xlabel("Range [km]")
        
    axs[0].set_ylim(0, z.max())
    axs[1].set_ylim(0, z.max())
    

    if show:
        plt.show(block=block)

    

def plot_1d_combined(src_loc, rcv_loc, date, atmos_dir, show=False, fig=None):
    atm_prof = g2s_profile.G2Sprofile(src_loc[0], src_loc[1], atmos_dir)
    atm_path = atm_prof.get_single_profile(date, atmos_dir)
    az_deg, baz_deg, dist_m = get_az_dist_between_points(src_loc, rcv_loc)

    taup = tau_p.TauP()
    taup_results = taup.evaluate_tau_p_integrals_profile(atm_path, gw=False)
    az_grid      = taup_results["az_degs"]
    inc_grid     = taup_results["inc_degs"]
    range_grid   = taup_results["ranges"]
    time_grid    = taup_results["travel_times"]
    celerity_grid    = taup_results["celerities"]
    az_dev_grid      = taup_results["az_deviations"]
    duct_height_grid = taup_results["ducting_heights"]
    phase_vel_grid   = taup_results["phase_velocities"]
    
    inc_degs = 90 - np.arange(10, 60, 1)
    az_degs  = np.array([az_deg])
    az_degs, inc_degs = np.meshgrid(az_degs, inc_degs)
    az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()
    rays = taup.get_ray_paths(atm_path, az_degs, inc_degs, gw=False, min_duct_height=500)

    x = rays["x"] / 1000
    z_rays = rays["z"] / 1000
    max_x = dist_m / 1000 + 100

    bounce_lengths = np.nanmax(x, axis=1)
    bounce_lengths[bounce_lengths <= 0] = -np.inf
    bounce_lengths[np.isnan(bounce_lengths)] = -np.inf
    n_bounces  = np.ceil(max_x / bounce_lengths).astype(int)
    max_bounces = n_bounces.max()
    offsets  = np.arange(max_bounces)[None, :] * bounce_lengths[:, None]
    x_tiled  = x[:, None, :] + offsets[:, :, None]
    z_tiled  = np.broadcast_to(z_rays[:, None, :], x_tiled.shape)
    needed   = np.arange(max_bounces)[None, :] < n_bounces[:, None]
    mask     = (x_tiled <= max_x) & needed[:, :, None]

    z_atm, u, v, c = atm_prof.get_atmo_vars(atm_path)
    c_eff = c + u * np.sin(np.radians(az_deg)) + v * np.cos(np.radians(az_deg))

    az_ticks  = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    az_labels = ["S", "SW", "W", "NW", "N", "NE", "E", "SE", "S"]
    
    if fig is None:     
        fig = plt.figure(figsize=(12, 14))
    gs  = fig.add_gridspec(2, 1, height_ratios=(1.5, 1))

    # Top block: replicate original plot_atm_profile logic exactly
    gs_top = gs[0].subgridspec(3, 2, hspace=0.6, wspace=0.4)
    top_axes = np.array([[fig.add_subplot(gs_top[r, c]) for c in range(2)] for r in range(3)])

    top_plots = [
        (top_axes[0, 0], phase_vel_grid,      "App. Velocity [m/s]",    [300, 600]),
        (top_axes[0, 1], duct_height_grid,    "Turning Height [km]",    [0, 120]),
        (top_axes[1, 0], celerity_grid,       "Celerity [m/s]",         [200, 360]),
        (top_axes[1, 1], range_grid,          "Range [km]",             [0, 500]),
        (top_axes[2, 0], az_dev_grid,         "Azimuth Deviation [°]",  [-10, 10]),
        (top_axes[2, 1], time_grid / 60,      "Travel Time [min]",      [0, 30]),
    ]
    for ax, grid, title, clim in top_plots:
        kwargs = dict(cmap=cmaps.divergent_cmap(), shading="auto", rasterized=True)
        if clim:
            kwargs["vmin"], kwargs["vmax"] = clim
        pc = ax.pcolormesh(az_grid, 90 - inc_grid, grid, **kwargs)
        fig.colorbar(pc, ax=ax, label=title)
        ax.set_xlabel("Backazimuth [°]")
        ax.set_ylabel("Inclination [°]")
        ax.axvline(az_deg, c="k", lw=2, alpha=0.7, ls="--")
        ax.set_title(title)
        ax.set_xticks(az_ticks)
        ax.set_xticklabels(az_labels)

    # Bottom block
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=(0.25, 1))
    ax_c   = fig.add_subplot(gs_bot[0])
    ax_ray = fig.add_subplot(gs_bot[1], sharey=ax_c)

    ax_c.plot(c, z_atm, c="k", lw=2, alpha=0.7, label="c")
    ax_c.plot(c_eff, z_atm, c="r", lw=2, alpha=0.7, label="c eff.")
    ax_c.axvline(c_eff[0], ls="--", c="k", alpha=0.7)
    ax_c.legend(loc="lower right", fontsize="small")
    ax_c.set_ylabel("Altitude [km]")
    ax_c.set_xlabel("Sound Speed [m/s]")
    ax_c.set_ylim(0, z_atm.max())

    lines_x = []
    lines_z = []

    for i in range(x_tiled.shape[0]):
        for j in range(max_bounces):
            m = mask[i, j, :]
            if not m.any():
                continue
            xij = x_tiled[i, j, :].copy()
            zij = z_tiled[i, j, :].copy()
            xij[~m] = np.nan
            zij[~m] = np.nan

            lines_x.append(xij)
            lines_z.append(zij)

    # Stack into one array
    lines_x = np.concatenate(lines_x)
    lines_z = np.concatenate(lines_z)

    ax_ray.plot(lines_x, lines_z, c="k", lw=0.3, alpha=0.7)

    ax_ray.set_xlim(0, max_x)
    ax_ray.set_xlabel("Range [km]")
    ax_ray.tick_params(labelleft=False) 

    if show:
        plt.show()
    return fig

if __name__ == "__main__":
    locations = {
        "teheran": (35.41, 51.20),
        "IS19": (11.45, 43.18),
        "IS31": (50.41, 58.03),
        "IS43": (56.71, 37.22),
        "IS48": (35.80, 9.32),
    }

    latlons = {
        "MAAG": (50.7014, 29.2301),
        "GRDI": (50.5993, 29.4471),
        "KPDI": (48.5630, 26.4562),
    }
    src_loc = locations["teheran"]
    # src_loc = latlons["MAAG"]
    # rcv_loc = latlons["KPDI"]
    # rcv_loc = locations["IS19"]
    rcv_loc = locations["IS31"]
    # rcv_loc = locations["IS43"]
    # rcv_loc = locations["IS48"]
    
    date = UTCDateTime("2026-04-01T12:00")
    date = UTCDateTime("2026-04-08T12:00")
    # date = UTCDateTime("2022-03-01T00:00:00")
    tmp_atm_dir = "/staff/sophus/packages/infraview/data/tmp_atmos_dir/"
    freq = 0.5
    plot_1d_atm(src_loc,rcv_loc, date, tmp_atm_dir, show=True, block=False)
    plot_1d_wavefield_fan(src_loc, rcv_loc, date, tmp_atm_dir, freq=freq)
    plot_1d_ray_fan(src_loc, rcv_loc, date, tmp_atm_dir)
    plot_1d_wavefield(src_loc, rcv_loc, date, tmp_atm_dir, freq=freq, show=True)
    plot_1d_rays(src_loc, rcv_loc, date, tmp_atm_dir, show=True)
    plt.show() 

