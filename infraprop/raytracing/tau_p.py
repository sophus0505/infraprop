import os

import time
from datetime import timedelta

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.stats import gaussian_kde

from infraprop.atmos import g2s_profile
from infraprop.utils import (
    GW_perturbations,
    atmos_files,
    cmaps,
    plot_maps,
)

rng = np.random.default_rng()

class TauP:
    # class for calculating the turning heights of rays using the Tau-p formulation for atmospheric infrasound from Garcia and Drob (garcia 1998 has the most thorough explanation) 
    # The class is made to handle both single profiles and grids in the NCPAProp format (as returned by G2S)
    # Other formats can be used, but the zTuvdp values still need to be accessible --> OBS! not tested
    # TODO: add support for elevated sources
    # TODO: Add support for multi-bounce ray-tracing (still 1D atmo)
    #
    # NB: As the method originates from seismology the inclination is "upside down" compared to usual for atmospheric infrasound (measured from the vertical)
    #     Other variables should be as expected...

    def __init__(self):
        self.format = "zTuvdp"

    def get_c(self, atmo_data, format=None):
        # Sound speed on Earth
        format = self.format if format is None else format
        c = np.sqrt(0.14 * atmo_data[:, self.format.find("p")] / atmo_data[:, self.format.find("d")])
        return c

    def get_k(self, az_degs, inc_degs):
        # get the normal vector of the rays
        az, inc = np.radians(az_degs), np.radians(inc_degs)
        k = np.stack([np.sin(az), np.cos(az), np.sin(inc)], axis=1)  # (N, 2)
        # kz = np.sin(inc)  # (N,)
        return k

    def get_u_proj(self, u_vec, k):
        # get the projected winds
        # only consider the horizontal components (k[:, :2])
        u_proj = (u_vec @ k[:, :2].T).T
        return u_proj

    def get_u_transverse(self, u_vec, k):
        # Wind component perpendicular to ray direction.
        # u_proj is along k[:, :2], transverse is the cross product magnitude
        # For 2D horizontal: rotate k by 90 degrees -> [-ky, kx]
        k_perp = np.stack([-k[:, 1], k[:, 0]], axis=1)  # (n_rays, 2)
        u_transverse = u_vec @ k_perp.T  # (n_z, n_rays)
        return u_transverse

    def get_p(self, az_degs, inc_degs, c, u_vec, z0_idx):
        # get the ray parameter as given in Drob morph.
        k = self.get_k(az_degs, inc_degs)
        u_proj = self.get_u_proj(u_vec, k)
        c0 = c[z0_idx]
        # k[:, 2] is the vertical component of the ray
        p = (k[:, 2] / c0) / (1 + (k[:, 2] * u_proj[:, z0_idx]) / c0)
        return p

    def get_duct_z_index(self, tc_ray, z, min_duct_height=500):
        # Find turning point using grid search + Brent's method.
        # min_duct_height: minimum turning height in meters (default 0.5 km)
        sign_changes = np.where(np.diff(np.sign(tc_ray)))[0]

        # Filter out sign changes below minimum duct height
        sign_changes = sign_changes[z[sign_changes] >= min_duct_height]

        if sign_changes.size == 0:
            return None, None

        i = sign_changes[0]
        tc_interp = lambda z_val: np.interp(z_val, z, tc_ray)
        z_star = brentq(tc_interp, z[i], z[i + 1], xtol=1e-3)

        return i, z_star

    def get_az_inc_from_voronoi(self, N_voronoi):
        # NB: Need to run the geodesic_tesselation file in /utils/ first as stripy is not compatible with other libraries
        data_dir = "/staff/sophus/Documents/ismonpy/ismonpy/utils/az_incs_voronoi/"
        fname = f"voronoi_{N_voronoi}.dat"
        azs, incs = np.loadtxt(os.path.join(data_dir, fname), unpack=True)
        return azs, incs

    def _integrate(self, integrand_vals, z, z0_idx, bracket_idx):
        """
        Integrate from z0 to just before the turning point,
        avoiding the singularity at the upper bound.
        """
        z_slice = z[z0_idx:bracket_idx]  # exclude the singular point
        integrand_slice = integrand_vals[z0_idx:bracket_idx]
        return np.trapezoid(integrand_slice, z_slice)

    def get_characteristic_function(self, az_degs, inc_degs, c, u_vec, z0_idx):
        # This is the chi symbol in Drob
        k = self.get_k(az_degs, inc_degs)
        u_proj = self.get_u_proj(u_vec, k)
        p = self.get_p(az_degs, inc_degs, c, u_vec, z0_idx)

        sum = 1 / c**2 - (p[:, None] ** 2) / (1 - p[:, None] * u_proj) ** 2
        valid = sum > 0
        char_func = np.zeros_like(sum)
        char_func[valid] = 1 / np.sqrt(sum[valid])
        return char_func.T

    def get_tau_p_turning_conditions(self, az_degs, inc_degs, c, u_vec, z0_idx):
        # get the turning condition for all rays defined by a tesselated sphere
        k = self.get_k(az_degs, inc_degs)
        u_proj = self.get_u_proj(u_vec, k)
        c0 = c[z0_idx]
        # k[:, 2] is the vertical component of the ray
        p = (k[:, 2] / c0) / (1 + (k[:, 2] * u_proj[:, z0_idx]) / c0)
        # v = 1 / p  # horizontal phase velocity of the rays
        # condition = v[:, None] - (
        #     c + u_proj
        # )  # is zero when the horizontal phase vel is equal to the effective sound speed
        # return condition
        return 1 / c**2 - (p[:, None] ** 2) / (1 - p[:, None] * u_proj) ** 2  # this is an alternative formulation from Drob

    def compute_ducting_heights_profile(
        self,
        atm_profile: str,
        z0_idx: float = 0,
        N_voronoi: int = 3,
        azimuthal: bool = False,
        gw: bool = False,
    ):
        # compute the ducting heights for the atm_profile (assuming self.format)
        # z0_idx is the index of the source altitude (assumes 0 to be surface level)
        data = np.loadtxt(atm_profile)
        z = data[:, self.format.find("z")]
        u = data[:, self.format.find("u")]
        v = data[:, self.format.find("v")]
        u_vec = np.column_stack((u, v))
        c = self.get_c(data)

        if gw:
            _, u_pert, v_pert = GW_perturbations.get_GW_1D_profile(z_min_km=z[0], z_max_km=z[-1], dz_km=z[1] - z[0])
            r_u = rng.uniform(0, 5)
            r_v = rng.uniform(0, 5)
            u, v = u + u_pert*r_u, v + v_pert*r_v

        az_degs, inc_degs = self.get_az_inc_from_voronoi(N_voronoi)
        all_tc = self.get_tau_p_turning_conditions(az_degs, inc_degs, c, u_vec, z0_idx)

        z_m = z * 1000  # km -> m, consistent with get_duct_z_index

        duct_heights = []
        duct_az = []
        duct_inc = []
        for ray_idx in range(az_degs.size):
            bracket_idx, z_star = self.get_duct_z_index(all_tc[ray_idx], z_m)
            if z_star is None:
                continue
            duct_heights.append(z_star / 1000)  # back to km for height bins
            if azimuthal:
                duct_az.append(az_degs[ray_idx])
                duct_inc.append(inc_degs[ray_idx])

        duct_heights = np.array(duct_heights)
        total = az_degs.size

        if azimuthal:
            duct_az = np.array(duct_az)
            duct_inc = np.array(duct_inc)

            north_mask = (duct_az >= -45) & (duct_az < 45)
            east_mask = (duct_az >= 45) & (duct_az < 135)
            south_mask = (duct_az >= 135) | (duct_az < -135)
            west_mask = (duct_az >= -135) & (duct_az < -45)

            north_total = az_degs[(az_degs >= -45) & (az_degs < 45)].size
            east_total = az_degs[(az_degs >= 45) & (az_degs < 135)].size
            south_total = az_degs[(az_degs >= 135) | (az_degs < -135)].size
            west_total = az_degs[(az_degs >= -135) & (az_degs < -45)].size

            north = {
                "tropos": np.sum(duct_heights[north_mask] < 20) / north_total,
                "stratos": np.sum((duct_heights[north_mask] >= 20) & (duct_heights[north_mask] < 70)) / north_total,
                "thermos": np.sum(duct_heights[north_mask] >= 70) / north_total,
                "escape": north_total - duct_az[north_mask].size,
            }

            east = {
                "tropos": np.sum(duct_heights[east_mask] < 20) / east_total,
                "stratos": np.sum((duct_heights[east_mask] >= 20) & (duct_heights[east_mask] < 70)) / east_total,
                "thermos": np.sum(duct_heights[east_mask] >= 70) / east_total,
                "escape": east_total - duct_az[east_mask].size,
            }

            south = {
                "tropos": np.sum(duct_heights[south_mask] < 20) / south_total,
                "stratos": np.sum((duct_heights[south_mask] >= 20) & (duct_heights[south_mask] < 70)) / south_total,
                "thermos": np.sum(duct_heights[south_mask] >= 70) / south_total,
                "escape": south_total - duct_az[south_mask].size,
            }

            west = {
                "tropos": np.sum(duct_heights[west_mask] < 20) / west_total,
                "stratos": np.sum((duct_heights[west_mask] >= 20) & (duct_heights[west_mask] < 70)) / west_total,
                "thermos": np.sum(duct_heights[west_mask] >= 70) / west_total,
                "escape": west_total - duct_az[west_mask].size,
            }
            results = {
                "north": north,
                "east": east,
                "south": south,
                "west": west,
                "duct_heights": duct_heights,
                "duct_az": duct_az,
            }
            return results

        tropos = np.sum(duct_heights < 20)
        stratos = np.sum((duct_heights >= 20) & (duct_heights < 70))
        thermos = np.sum(duct_heights >= 70)
        escape = total - duct_heights.size

        results = {
            "tropos": tropos / total,
            "stratos": stratos / total,
            "thermos": thermos / total,
            "escape": escape / total,
            "duct_heights": duct_heights,
        }
        return results

    def get_ray_paths(self, atm_profile, az_degs, inc_degs, gw = False, min_duct_height=100):
        # Compute the ray path x(z), t(z) for a single ray defined by (az, inc).
        # Returns the full path from z0 to the turning point and back down.
        # TODO: extend for multiple bounces to reveal the tropospheric waveguide?

        data = np.loadtxt(atm_profile)
        z0_idx = 0
        z = data[:, self.format.find("z")]
        u = data[:, self.format.find("u")]
        v = data[:, self.format.find("v")]
        c = self.get_c(data)
        zeta = 1 / c**2

        if gw:
            _, u_pert, v_pert = GW_perturbations.get_GW_1D_profile(z_min_km=z[0], z_max_km=z[-1], dz_km=z[1] - z[0])
            r_u = rng.uniform(0, 5)
            r_v = rng.uniform(0, 5)
            u, v = u + u_pert*r_u, v + v_pert*r_v

        u_vec = np.column_stack((u, v))

        k = self.get_k(az_degs, inc_degs)
        u_proj = self.get_u_proj(u_vec, k)
        u_transverse = self.get_u_transverse(u_vec, k)
        p = self.get_p(az_degs, inc_degs, c, u_vec, z0_idx)
        char_func = self.get_characteristic_function(az_degs, inc_degs, c, u_vec, z0_idx)
        tc = self.get_tau_p_turning_conditions(az_degs, inc_degs, c, u_vec, z0_idx)
        z_m = z * 1000

        range_func = char_func * ((p / (1 - u_proj.T * p)) + u_proj.T * zeta[:, None])
        time_func = char_func * zeta[:, None]
        transverse_func = char_func * zeta[:, None] * u_transverse

        x_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        y_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        z_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        t_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        arrivals_idx = np.full(az_degs.size, np.nan)

        bracket_indices = []
        z_stars = []
        for ray_idx in range(az_degs.size):
            bracket_idx, z_star = self.get_duct_z_index(tc[ray_idx], z_m, min_duct_height=min_duct_height)
            bracket_indices.append(bracket_idx)
            z_stars.append(z_star)

        for ray_idx in range(az_degs.size):
            bracket_idx = bracket_indices[ray_idx]
            z_star = z_stars[ray_idx]
            if bracket_idx is None:
                continue
            # range_integrals[ray_idx] = np.trapezoid(range_func[:idx, ray_idx], z_m[:idx])

            valid = char_func[:bracket_idx, ray_idx] > 0
            if not np.any(valid):
                continue
            first_valid = np.argmax(valid)  # first non-zero index

            z_up = z_m[first_valid:bracket_idx]

            x_up = cumulative_trapezoid(range_func[first_valid:bracket_idx, ray_idx], z_up, initial=0)
            y_up = cumulative_trapezoid(transverse_func[first_valid:bracket_idx, ray_idx], z_up, initial=0)
            t_up = cumulative_trapezoid(time_func[first_valid:bracket_idx, ray_idx], z_up, initial=0)

            z_down = z_up[::-1]
            y_descent = cumulative_trapezoid(transverse_func[first_valid:bracket_idx, ray_idx][::-1], z_down, initial=0)
            y_down = 2 * y_up[-1] - y_up[::-1]

            x_down = 2 * x_up[-1] - x_up[::-1]
            # y_down = 2 * y_up[-1] - y_up[::-1]
            t_down = 2 * t_up[-1] - t_up[::-1]

            # save the full path
            x_tot[ray_idx, first_valid : 2 * bracket_idx - 1] = np.concatenate([x_up, x_down[1:]])
            y_tot[ray_idx, first_valid : 2 * bracket_idx - 1] = np.concatenate([y_up, y_down[1:]])
            z_tot[ray_idx, first_valid : 2 * bracket_idx - 1] = np.concatenate([z_up, z_down[1:]])
            t_tot[ray_idx, first_valid : 2 * bracket_idx - 1] = np.concatenate([t_up, t_down[1:]])
            arrivals_idx[ray_idx] = int(2*bracket_idx - 1)

        return {
            "x": x_tot,  # range (m)
            "y": y_tot,  # transverse offset (m)
            "z": z_tot,  # altitude (m)
            "t": t_tot,  # travel time (s)
            "arrival_idx": arrivals_idx.astype(int),
        }

    def plot_ray_paths_2d(self, az_degs, inc_degs, atm_profile: str, gw: bool = False, only_arrivals: bool = False, savefile = None, lat0=None, lon0=None, title=None, add_ukr=True):
        rays = self.get_ray_paths(atm_profile, az_degs, inc_degs)

        az = np.deg2rad(az_degs)[:, None]

        # Need to double check the correct translation
        X = rays["x"] * np.sin(az) - rays["y"] * np.cos(az)
        Y = rays["x"] * np.cos(az) + rays["y"] * np.sin(az)

        az_dev = np.degrees(np.arctan2(rays["y"], rays["x"]))
        
        if lat0 is None or lon0 is None:
            if "g2stxt" in atm_profile:
                date, lat0, lon0 = atmos_files.parse_ncpag2s_fname(os.path.basename(atm_profile))
            else:
                date, lat0, lon0 = atmos_files.parse_g2s_fname(os.path.basename(atm_profile))

        local_crs = ccrs.AzimuthalEquidistant(
            central_longitude=lon0,
            central_latitude=lat0,
        )
        lonlats = ccrs.PlateCarree().transform_points(local_crs, X, Y)
        lons = lonlats[:, :, 0]
        lats = lonlats[:, :, 1]

        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
        if add_ukr:
            plot_maps.add_map_ukraine(ax=ax1, add_fill=False)
        else: 
            plot_maps.add_general_map(lat0, lon0, max_dist_km=np.nanmax(rays["x"])/1000 + 100, ax=ax1)

        x = rays["x"]
        y = rays["y"]
        t = rays["t"]
        if only_arrivals: 
            arr_idxs = rays["arrival_idx"]
            lons = lons[:, arr_idxs]
            lats = lats[:, arr_idxs]
            az_dev = az_dev[:, arr_idxs]
            x = x[:, arr_idxs]
            y = y[:, arr_idxs]
            t = t[:, arr_idxs]

        sc1 = ax1.scatter(
            lons,
            lats,
            c=az_dev,
            s=0.025,
            alpha=0.9,
            cmap=cmaps.divergent_cmap(),
            vmin=-8,
            vmax=8,
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(sc1, ax=ax1, label="Azimuth deviation [Deg]", pad=0.015, shrink=0.7)
        if title is not None: 
            ax1.set_title(title)
        else: 
            ax1.set_title("Tau-p ray fan:")

        ax2 = fig.add_subplot(gs[1, 0])
        sc2 = ax2.scatter(
            x / 1000,
            y / 1000,
            c = t / 60,
            s=0.025,
            alpha=0.9,
            cmap=cmc.oslo,
            vmin=0,
            vmax=30,
        )
        fig.colorbar(sc2, ax=ax2, label="Travel time [Min.]", pad=0.015)
        ax2.set_xlabel("Range [km]")
        ax2.set_ylabel("Deviation from geodesic [km]")
        ax2.set_xlim(0, 600)
        ax2.set_ylim(-65, 65)

        if savefile:
            plt.savefig(savefile)
            plt.close()
        else:
            plt.show(block=False)

    def plot_ray_paths(self, az_deg, inc_degs, atm_profile: str, gw: bool = False):
        # Plot a fan of rays at a given azimuth for multiple inclination angles.
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
        start_time = time.perf_counter()

        az_degs = np.array([az_deg])
        az_degs, inc_degs = np.meshgrid(az_degs, inc_degs)
        az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()

        rays = self.get_ray_paths(atm_profile, az_degs, inc_degs)
        az_degs_l = np.array([(az_deg + 360) % 360 - 180])
        az_degs_l, inc_degs = np.meshgrid(az_degs_l, inc_degs)
        az_degs_l, inc_degs = az_degs_l.flatten(), inc_degs.flatten()
        rays_l = self.get_ray_paths(atm_profile, az_degs_l, inc_degs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Ray paths time: {elapsed_time:.4f} seconds")
        print(f"Ray resolution: {rays['x'].shape}")

        ax.scatter(
            rays["x"] / 1000,
            rays["z"] / 1000,
            c=rays["t"] / 60,
            s=0.025,
            alpha=0.8,
            cmap=cmc.oslo,
            vmin=0,
            vmax=30,
        )
        sc = ax.scatter(
            -rays_l["x"] / 1000,
            rays_l["z"] / 1000,
            c=rays_l["t"] / 60,
            s=0.025,
            alpha=0.8,
            cmap=cmc.oslo,
            vmin=0,
            vmax=30,
        )
        fig.colorbar(sc, label="Travel time [Min.]", pad=0.015)
        ax.set_xlabel("Range [km]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(f"Tau-p ray fan: az=+-{az_deg}°")
        ax.set_ylim(0, 150)

        plt.show(block=False)

    def get_eigenrays(self, atm_profile, az_degs, inc_degs, gw: bool = False, plot: bool = False):
        data = np.loadtxt(atm_profile)
        z0_idx = 0
        z = data[:, self.format.find("z")]
        u = data[:, self.format.find("u")]
        v = data[:, self.format.find("v")]
        c = self.get_c(data)
        zeta = 1 / c**2

        if gw:
            _, u_pert, v_pert = GW_perturbations.get_GW_1D_profile(z_min_km=z[0], z_max_km=z[-1], dz_km=z[1] - z[0])
            r_u = rng.uniform(0, 5)
            r_v = rng.uniform(0, 5)
            u, v = u + u_pert*r_u, v + v_pert*r_v
        u_vec = np.column_stack((u, v))

        k = self.get_k(az_degs, inc_degs)
        u_proj = self.get_u_proj(u_vec, k)
        u_transverse = self.get_u_transverse(u_vec, k)
        p = self.get_p(az_degs, inc_degs, c, u_vec, z0_idx)
        char_func = self.get_characteristic_function(az_degs, inc_degs, c, u_vec, z0_idx)
        tc = self.get_tau_p_turning_conditions(az_degs, inc_degs, c, u_vec, z0_idx)
        z_m = z * 1000

        range_func = char_func * ((p / (1 - u_proj.T * p)) + u_proj.T * zeta[:, None])
        time_func = char_func * zeta[:, None]
        transverse_func = char_func * zeta[:, None] * u_transverse
        taup_func = ((1 - u_proj * p[:, None]) * np.emath.sqrt(zeta - p[:, None]**2 / (1 - p[:, None]*u_proj)**2)).T

        x_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        y_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        z_tot = np.full((az_degs.size, 2 * z.size), np.nan)
        taup_tot = np.full((az_degs.size, z.size), np.nan)
        t_tot = np.full((az_degs.size, 2 * z.size), np.nan)

        bracket_indices = []
        z_stars = []
        for ray_idx in range(az_degs.size):
            bracket_idx, z_star = self.get_duct_z_index(tc[ray_idx], z_m)
            bracket_indices.append(bracket_idx)
            z_stars.append(z_star)

        for ray_idx in range(az_degs.size):
            bracket_idx = bracket_indices[ray_idx]
            z_star = z_stars[ray_idx]

            
            if bracket_idx is None:
                continue
            
            valid = char_func[:bracket_idx, ray_idx] > 0
            if not np.any(valid):
                continue
            first_valid = np.argmax(valid)

            z_up = z_m[first_valid:bracket_idx]
            z_down = z_up[::-1]

            x_up = cumulative_trapezoid(range_func[first_valid:bracket_idx, ray_idx], z_up, initial=0)
            y_up = cumulative_trapezoid(transverse_func[first_valid:bracket_idx, ray_idx], z_up, initial=0)
            t_up = cumulative_trapezoid(time_func[first_valid:bracket_idx, ray_idx], z_up, initial=0)
            taup_up = cumulative_trapezoid(taup_func[first_valid:bracket_idx, ray_idx].real, z_up, initial=0)

            x_down = 2 * x_up[-1] - x_up[::-1]
            y_down = 2 * y_up[-1] - y_up[::-1]
            t_down = 2 * t_up[-1] - t_up[::-1]

            n_up = bracket_idx - first_valid
            end_idx = first_valid + 2 * n_up - 1

            x_tot[ray_idx, first_valid:end_idx] = np.concatenate([x_up, x_down[1:]])
            y_tot[ray_idx, first_valid:end_idx] = np.concatenate([y_up, y_down[1:]])
            z_tot[ray_idx, first_valid:end_idx] = np.concatenate([z_up, z_down[1:]])
            t_tot[ray_idx, first_valid:end_idx] = np.concatenate([t_up, t_down[1:]])
            taup_tot[ray_idx, first_valid:bracket_idx] = taup_up

        if plot:
            az_unique = np.unique(az_degs)
            inc_unique = np.unique(inc_degs)
            n_az = az_unique.size
            n_inc = inc_unique.size

            taup_integrand_grid = taup_func.real.T.reshape(n_inc, n_az, z.size)
            taup_grid = taup_tot.reshape(n_inc, n_az, z.size)
            x_grid = x_tot.reshape(n_inc, n_az, 2 * z.size)
            y_grid = y_tot.reshape(n_inc, n_az, 2 * z.size)
            t_grid = t_tot.reshape(n_inc, n_az, 2 * z.size)
            p_grid = p.reshape(n_inc, n_az)

            az_idx = np.searchsorted(az_unique, 90)
            p_single = p_grid[:, az_idx]

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            mesh3 = ax3.pcolormesh(
                inc_unique,
                z,
                taup_integrand_grid[:, az_idx, :].T,
                cmap="jet",
                shading="auto"
            )
            plt.colorbar(mesh3, ax=ax3, label="τ integrand (s/m)")
            ax3.set_xlabel("Launch angle (degrees from vertical)")
            ax3.set_ylabel("Elevation (km)")
            ax3.set_title(f"Tau — azimuth {az_unique[az_idx]:.1f}°")
            plt.tight_layout()

            x_final = np.array([
                row[~np.isnan(row)][-1] / 1000 if np.any(~np.isnan(row)) else np.nan
                for row in x_grid[:, az_idx, :]
            ])
            y_final = np.array([
                row[~np.isnan(row)][-1] / 1000 if np.any(~np.isnan(row)) else np.nan
                for row in y_grid[:, az_idx, :]
            ])
            t_final = np.array([
                row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
                for row in t_grid[:, az_idx, :]
            ])
            taup_final = np.array([
                row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
                for row in taup_grid[:, az_idx, :]
            ])

            fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
            axes5[0, 0].scatter(p_single * 1000, t_final, s=1, c="k")
            axes5[0, 0].set_xlabel("Ray parameter (s/km)")
            axes5[0, 0].set_ylabel("Travel time (s)")
            axes5[0, 0].set_title("(a) Travel time vs ray parameter")

            axes5[0, 1].scatter(p_single * 1000, x_final, s=1, c="k")
            axes5[0, 1].set_xlabel("Ray parameter (s/km)")
            axes5[0, 1].set_ylabel("Range (km)")
            axes5[0, 1].set_title("(b) Range vs ray parameter")

            axes5[1, 0].scatter(p_single * 1000, y_final, s=1, c="k")
            axes5[1, 0].set_xlabel("Ray parameter (s/km)")
            axes5[1, 0].set_ylabel("Transverse offset (km)")
            axes5[1, 0].set_title("(c) Transverse offset vs ray parameter")

            axes5[1, 1].scatter(p_single * 1000, taup_final, s=1, c="k")
            axes5[1, 1].set_xlabel("Ray parameter (s/km)")
            axes5[1, 1].set_ylabel("τ (s)")
            axes5[1, 1].set_title("(d) Tau vs ray parameter")

            plt.tight_layout()
            plt.show()

        return {
            "x": x_tot,
            "y": y_tot,
            "z": z_tot,
            "t": t_tot,
            "taup": taup_tot,
        }

    def plot_arrival_map(self, atm_profile, az_degs, inc_degs, gw: bool = False, max_range_km: float = 600.0, multiple_bounces: bool = False):
        result = self.get_eigenrays(atm_profile, az_degs, inc_degs, gw=gw)

        az_unique = np.unique(az_degs)
        inc_unique = np.unique(inc_degs)
        n_az = az_unique.size
        n_inc = inc_unique.size

        x_final = np.array([
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in result["x"]
        ])
        y_final = np.array([
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in result["y"]
        ])
        t_final = np.array([
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in result["t"]
        ])

        az_rad = np.radians(az_degs)

        all_x = []
        all_y = []
        all_t = []
        all_n = []

        for n in range(1, int(max_range_km / 1) + 1):
            x_n = x_final * n
            y_n = y_final * n
            t_n = t_final * n

            total_range = np.sqrt(x_n**2 + y_n**2) / 1000
            valid = total_range <= max_range_km

            if not np.any(valid):
                break

            x_arrival = (x_n[valid] * np.cos(az_rad[valid]) - y_n[valid] * np.sin(az_rad[valid])) / 1000
            y_arrival = (x_n[valid] * np.sin(az_rad[valid]) + y_n[valid] * np.cos(az_rad[valid])) / 1000

            all_x.append(x_arrival)
            all_y.append(y_arrival)
            all_t.append(t_n[valid])
            all_n.append(np.full(valid.sum(), n))
            if not multiple_bounces: 
                break

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_t = np.concatenate(all_t)
        all_n = np.concatenate(all_n)
        all_celerity = np.sqrt((all_x * 1000)**2 + (all_y * 1000)**2) / all_t / 1000

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sc0 = axes[0].scatter(all_x, all_y, c=all_t, s=1, cmap="jet")
        plt.colorbar(sc0, ax=axes[0], label="Travel time (s)")
        axes[0].set_xlabel("East (km)")
        axes[0].set_ylabel("North (km)")
        axes[0].set_title("Arrival map — travel time")
        axes[0].set_aspect("equal")
        axes[0].scatter(0, 0, c="red", s=50, zorder=5, label="Source")
        axes[0].legend()

        sc1 = axes[1].scatter(all_x, all_y, c=all_celerity, s=1, cmap="jet")
        plt.colorbar(sc1, ax=axes[1], label="Celerity (km/s)")
        axes[1].set_xlabel("East (km)")
        axes[1].set_ylabel("North (km)")
        axes[1].set_title("Arrival map — celerity")
        axes[1].set_aspect("equal")
        axes[1].scatter(0, 0, c="red", s=50, zorder=5, label="Source")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


    def evaluate_tau_p_integrals_profile(
        self,
        atm_profile: str,
        az_resol: float = 1,
        inc_resol: float = 0.5,
        N_voronoi: int | None = None,
        gw: bool = False,
    ):
        # evaluate the tau-p integrals used in Drob et al. papers
        # Outputs a dictionary containing all the releavat results 
        # TODO: Implement the actual tau-p integral as well --> Can be usefull for finding eigenrays. 
        #       (This might be better in a separate function as you wont need all the integrals calculated here i think...)

        data = np.loadtxt(atm_profile)
        z0_idx = 0
        z = data[:, self.format.find("z")]
        u = data[:, self.format.find("u")]
        v = data[:, self.format.find("v")]
        if gw:
            _, u_pert, v_pert = GW_perturbations.get_GW_1D_profile(z_min_km=z[0], z_max_km=z[-1], dz_km=z[1] - z[0])
            r_u = rng.uniform(0, 5)
            r_v = rng.uniform(0, 5)
            u, v = u + u_pert*r_u, v + v_pert*r_v

        u_vec = np.column_stack((u, v))
        c = self.get_c(data)

        if N_voronoi is not None:
            az_degs, inc_degs = self.get_az_inc_from_voronoi(N_voronoi)
        else:
            az_degs, inc_degs = np.meshgrid(np.arange(-180, 180, az_resol), np.arange(30, 90, inc_resol))
            az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()

        k = self.get_k(az_degs, inc_degs)
        u_proj = self.get_u_proj(u_vec, k)
        u_transverse = self.get_u_transverse(u_vec, k)
        p = self.get_p(az_degs, inc_degs, c, u_vec, z0_idx)
        char_func = self.get_characteristic_function(az_degs, inc_degs, c, u_vec, z0_idx)
        zeta = 1 / c**2

        tc = 1 / p[:, None] - (c + u_proj)

        range_func = char_func * ((p / (1 - u_proj.T * p)) + u_proj.T * zeta[:, None])
        time_func = char_func * zeta[:, None]
        transverse_func = char_func * zeta[:, None] * u_transverse

        z_m = z * 1000

        range_integrals = np.full(az_degs.size, np.nan)
        time_integrals = np.full(az_degs.size, np.nan)
        transverse_integrals = np.full(az_degs.size, np.nan)
        duct_heights = np.full(az_degs.size, np.nan)

        bracket_indices = []
        z_stars = []
        for ray_idx in range(az_degs.size):
            bracket_idx, z_star = self.get_duct_z_index(tc[ray_idx], z_m)
            bracket_indices.append(bracket_idx)
            z_stars.append(z_star)

        for ray_idx in range(az_degs.size):
            idx = bracket_indices[ray_idx]
            z_star = z_stars[ray_idx]
            if idx is None:
                continue
            range_integrals[ray_idx] = np.trapezoid(range_func[:idx, ray_idx], z_m[:idx])
            time_integrals[ray_idx] = np.trapezoid(time_func[:idx, ray_idx], z_m[:idx])
            transverse_integrals[ray_idx] = np.trapezoid(transverse_func[:idx, ray_idx], z_m[:idx])
            duct_heights[ray_idx] = z_star / 1000

        range_integrals *= 2
        time_integrals *= 2
        transverse_integrals *= 2
        results = {
                "az_degs": az_degs, 
                "inc_degs": inc_degs, 
                "travel_times": time_integrals, 
                "ranges": range_integrals / 1000, 
                "az_deviations": np.degrees(np.arctan2(transverse_integrals, range_integrals)),
                "ducting_heights": duct_heights, 
                "celerities": range_integrals / time_integrals, 
                "phase_velocities": 1 / p,   
        }   

        if N_voronoi is not None:
            # If the tesselation is used, we do not have a regular grid. 
            # use az and inc as reference... 
            return results

        az_grid = np.unique(az_degs)
        inc_grid = np.unique(inc_degs)
        n_az, n_inc = az_grid.size, inc_grid.size

        range_grid = range_integrals.reshape(n_inc, n_az) / 1000  # m -> km
        time_grid = time_integrals.reshape(n_inc, n_az)
        celerity_grid = range_grid * 1000 / time_grid  # m/s
        az_dev_grid = np.degrees(
            np.arctan2(
                transverse_integrals.reshape(n_inc, n_az),
                range_integrals.reshape(n_inc, n_az),
            )
        )
        duct_height_grid = duct_heights.reshape(n_inc, n_az)

        # Phase velocity: 1/p in m/s
        phase_vel = np.full(az_degs.size, np.nan)
        valid = ~np.isnan(range_integrals)
        phase_vel[valid] = 1 / p[valid]
        phase_vel_grid = phase_vel.reshape(n_inc, n_az)

        results["az_degs"] = az_grid
        results["inc_degs"] = inc_grid
        results["ranges"] = range_grid
        results["travel_times"] = time_grid
        results["celerities"] = celerity_grid
        results["ducting_heights"] = duct_height_grid
        results["phase_velocities"] = phase_vel_grid
        results["az_deviations"] = az_dev_grid
        
        return results

    def plot_distributions(self, atm_profiles, N_voronoi=3, filter_out_thermo=True, gw=False):
        # TODO: Add the possibility of filtering the values based on priors such as az, app cel, and turning height
        if type(atm_profiles) is str: 
            atm_profiles = [atm_profiles]
        cel_vals = []
        az_dev_vals = []
        for atm_profile in atm_profiles:
            print(atm_profile)
            if "arrivals" in atm_profile:
                continue 
            elif "raypaths" in atm_profile:
                continue
            elif ".met" not in atm_profile: 
                continue
            print(f"processing {os.path.basename(atm_profile)}")
            n_reals = 10 if gw else 1
            for i in range(n_reals):
                # (
                #     az_degs,
                #     inc_degs,
                #     range_integrals,
                #     time_integrals,
                #     az_deviations,
                #     duct_heights,
                #     celerities,
                # ) = self.evaluate_tau_p_integrals_profile(atm_profile, N_voronoi=N_voronoi, gw=gw)
                taup_results = self.evaluate_tau_p_integrals_profile(atm_profile, N_voronoi=N_voronoi, gw=gw)


                mask = (taup_results["ducting_heights"] < 100) & (~np.isnan(taup_results["ducting_heights"]))
                # mask = ~np.isnan(duct_heights)

                cel_vals += list(taup_results["celerities"][mask].flatten())
                az_dev_vals += list(taup_results["az_deviations"][mask].flatten())
        
        cel_vals = np.array(cel_vals)
        az_dev_vals = np.array(az_dev_vals)
        cel_bins = np.arange(200, 360, 1)
        az_dev_bins = np.arange(-8, 8, 0.1)

        # cel_kde = gaussian_kde(cel_vals)
        # cel_kde_x = np.linspace(200, 350, 100)
        # az_dev_kde = gaussian_kde(az_dev_vals)
        
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs = axs.flatten()
        # date, lat, lon = atmos_files.parse_g2s_fname(os.path.basename(atm_profile))
        # fig.suptitle(f"Distributions for ({lat}, {lon}) at {date.strftime('%Y-%m-%dT%H')}")

        axs[0].hist(cel_vals, bins=cel_bins, density=True)
        # axs[0].plot(cel_kde_x, cel_kde(cel_kde_x))
        axs[0].set_title("Celerity")
        axs[0].set_xlabel("Celerity [m/s]")
        axs[0].set_ylabel("Count")

        axs[1].hist(az_dev_vals, bins=az_dev_bins, density=True)
        # axs[1].plot(az_dev_kde_x, az_dev_kde(az_dev_kde_x))
        axs[1].set_title("Azimuth deviation")
        axs[0].set_xlabel("Az. dev [Deg.]")
        axs[0].set_ylabel("Count")

        plt.show(block=False)

    def plot_atm_profile(self, atm_profile, gw=False, title=None, savefile=None, az_deg=None, plot_c=False, block=False):
        taup_results = self.evaluate_tau_p_integrals_profile(atm_profile, gw=gw)
        az_grid = taup_results["az_degs"]
        inc_grid = taup_results["inc_degs"]
        range_grid = taup_results["ranges"]
        time_grid = taup_results["travel_times"]
        celerity_grid = taup_results["celerities"]
        az_dev_grid = taup_results["az_deviations"]
        duct_height_grid = taup_results["ducting_heights"]
        phase_vel_grid = taup_results["phase_velocities"]

        az_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        az_labels = ["S", "SW", "W", "NW", "N", "NE", "E", "SE", "S"]

        fig, axes = plt.subplots(3, 2, figsize=(10, 8))

        if savefile is not None: 
            fig.suptitle(title)

        plots = [
            (axes[0, 0], phase_vel_grid, "App. Velocity", [300, 600]),
            (axes[0, 1], duct_height_grid, "Turning Height", [0, 120]),
            (axes[1, 0], celerity_grid, "Celerity", [200, 360]),
            (axes[1, 1], range_grid, "Range", [0, 500]),
            (axes[2, 0], az_dev_grid, "Azimuth Deviation", [-10, 10]),
            (axes[2, 1], time_grid / 60, "Travel Time", [0, 30]),
        ]

        for ax, grid, title, clim in plots:
            kwargs = dict(cmap=cmaps.divergent_cmap(), shading="auto")
            if clim is not None:
                kwargs["vmin"], kwargs["vmax"] = clim
            pc = ax.pcolormesh(az_grid, 90 - inc_grid, grid, **kwargs)
            plt.colorbar(pc, ax=ax, label=title)
            if az_deg is not None: 
                ax.axvline(az_deg, lw=2, ls="--", c="k")
            ax.set_xlabel("Backazimuth [Deg.]")
            ax.set_ylabel("Inclination [Deg.]")
            ax.set_title(title)
            ax.set_xticks(az_ticks)
            ax.set_xticklabels(az_labels)

        plt.tight_layout()
        if savefile is not None: 
            plt.savefig(savefile)   
        else:
            plt.show(block=block)

        if plot_c:
            data = np.loadtxt(atm_profile)
            z0_idx = 0
            z = data[:, self.format.find("z")]
            u = data[:, self.format.find("u")]
            v = data[:, self.format.find("v")]
            if gw:
                _, u_pert, v_pert = GW_perturbations.get_GW_1D_profile(z_min_km=z[0], z_max_km=z[-1], dz_km=z[1] - z[0])
                u, v = u + u_pert, v + v_pert
                data[:, self.format.find("u")] = u
                data[:, self.format.find("v")] = v

            c = self.get_c(data)

            fig, axs = plt.subplots(1, 2, figsize=(8, 8))
            axs = axs.flatten()
            axs[0].grid(color="k", linestyle="--", linewidth=0.5)
            axs[1].grid(color="k", linestyle="--", linewidth=0.5)

            axs[0].set_ylim(z[0], z[-1])
            axs[0].set_ylabel("Altitude [km]")
            axs[0].set_xlabel("Sound Speed [m/s]")

            axs[1].set_ylim(z[0], z[-1])
            axs[1].set_xlabel("Wind Speed [m/s]")

            axs[1].yaxis.set_ticklabels([])

            axs[0].plot(c, z, "-k", linewidth=3.0)
            axs[1].plot(u, z, "indigo", linewidth=3.0, label="Zonal")
            axs[1].plot(v, z, "orangered", linewidth=3.0, label="Merid.")
            axs[1].legend(fontsize="small")
            plt.show(block=block)

    def compute_ducting_heights_grid(
        self,
        date: UTCDateTime,
        grid_dir: str,
        N_voronoi: int = 3,
        azimuthal: bool = False,
        gw: bool = False,
    ):
        # Compute the ducting heights on a grid of profiles on the NCPAProp format
        # Example grids can be computed using ismonpy.atmos.g2s_profiles
        grid_dir = atmos_files.find_g2s_grid(date, grid_dir)

        summary_file = os.path.join(grid_dir, "summary.dat")

        coords = np.loadtxt(summary_file, usecols=(0, 1))
        fnames = np.loadtxt(summary_file, usecols=2, dtype=str)

        lats, lons = coords[:, 0], coords[:, 1]
        unique_lats = np.unique(lats)
        unique_lons = np.unique(lons)
        lat_idx = np.searchsorted(unique_lats, lats)
        lon_idx = np.searchsorted(unique_lons, lons)
        if azimuthal:
            grid_tropos_n = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_tropos_e = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_tropos_s = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_tropos_w = np.full((unique_lats.size, unique_lons.size), np.nan)

            grid_stratos_n = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_stratos_e = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_stratos_s = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_stratos_w = np.full((unique_lats.size, unique_lons.size), np.nan)

            grid_thermos_n = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_thermos_e = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_thermos_s = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_thermos_w = np.full((unique_lats.size, unique_lons.size), np.nan)
        else:
            grid_tropos = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_stratos = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_thermos = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_escape = np.full((unique_lats.size, unique_lons.size), np.nan)
            grid_mean = np.full((unique_lats.size, unique_lons.size), np.nan)

        z0_idx = 0
        for k, (lat, lon, fname) in enumerate(zip(lats, lons, fnames)):
            atm_profile = os.path.join(grid_dir, fname)
            print(f"Processing {lat = :.3f}, {lon = :.3f}")
            grid_res = self.compute_ducting_heights_profile(atm_profile, z0_idx, N_voronoi=N_voronoi, azimuthal=azimuthal, gw=gw)
            i = lat_idx[k]
            j = lon_idx[k]
            if not azimuthal:
                tropo, strato, thermo, escape, turning_heights = (
                    grid_res["tropos"],
                    grid_res["stratos"],
                    grid_res["thermos"],
                    grid_res["escape"],
                    grid_res["duct_heights"],
                )

                grid_mean[i, j] = np.mean(turning_heights)
                grid_tropos[i, j] = tropo
                grid_stratos[i, j] = strato
                grid_thermos[i, j] = thermo
                grid_escape[i, j] = escape
            else:
                north, east, south, west, duct_heights, duct_az = (
                    grid_res["north"],
                    grid_res["east"],
                    grid_res["south"],
                    grid_res["west"],
                    grid_res["duct_heights"],
                    grid_res["duct_az"],
                )
                grid_tropos_n[i, j] = north["tropos"]
                grid_tropos_e[i, j] = east["tropos"]
                grid_tropos_s[i, j] = south["tropos"]
                grid_tropos_w[i, j] = west["tropos"]

                grid_stratos_n[i, j] = north["stratos"]
                grid_stratos_e[i, j] = east["stratos"]
                grid_stratos_s[i, j] = south["stratos"]
                grid_stratos_w[i, j] = west["stratos"]

                grid_thermos_n[i, j] = north["thermos"]
                grid_thermos_e[i, j] = east["thermos"]
                grid_thermos_s[i, j] = south["thermos"]
                grid_thermos_w[i, j] = west["thermos"]

                # "north": north,
                # "east": east,
                # "south": south,
                # "west": west,
                # "duct_heights": duct_heights,
                # "duct_az": duct_az,
        if azimuthal:
            grids_tropos = {
                "north": grid_tropos_n,
                "east": grid_tropos_e,
                "south": grid_tropos_s,
                "west": grid_tropos_w,
            }
            grids_stratos = {
                "north": grid_stratos_n,
                "east": grid_stratos_e,
                "south": grid_stratos_s,
                "west": grid_stratos_w,
            }
            grids_thermos = {
                "north": grid_thermos_n,
                "east": grid_thermos_e,
                "south": grid_thermos_s,
                "west": grid_thermos_w,
            }
            results = {
                "lats": unique_lats,
                "lons": unique_lons,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "grids_tropos": grids_tropos,
                "grids_stratos": grids_stratos,
                "grids_thermos": grids_thermos,
            }
        else:
            results = {
                "lats": unique_lats,
                "lons": unique_lons,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "grid_mean": grid_mean,
                "grid_tropos": grid_tropos,
                "grid_stratos": grid_stratos,
                "grid_thermos": grid_thermos,
                "grid_escape": grid_escape,
            }
        return results

    def plot_grids_azimuthal(self, date: UTCDateTime, grid_dir: str, N_voronoi=3, gw=False):
        results = self.compute_ducting_heights_grid(date, grid_dir, N_voronoi=N_voronoi, azimuthal=True, gw=gw)
        lons, lats = results["lons"], results["lats"]
        grids_tropos = results["grids_tropos"]
        grids_stratos = results["grids_stratos"]
        grids_thermos = results["grids_thermos"]
        LONS, LATS = np.meshgrid(lons, lats)
        extent = [
            lons.min(),
            lons.max(),
            lats.min(),
            lats.max(),
        ]

        grid_dirs = ["north", "east", "south", "west"]

        grid_list = [grids_tropos, grids_stratos, grids_thermos]
        grid_names = ["tropos", "stratos", "thermos"]
        vlims = [(0, 0.4), (0, 0.6), (0.3, 0.9)]
        for i, grid in enumerate(grid_list[:]):
            name = grid_names[i]
            # fig, ax = plot_maps.add_map_ukraine(extent=extent, add_fill=False, ukr_lw=1.5)
            fig, axs = plt.subplots(
                2,
                2,  # or however many subplots you need
                figsize=(11, 8),
                subplot_kw={"projection": ccrs.Mercator()},
                constrained_layout=True,
                sharex=True,
                sharey=True,
            )
            axs = axs.flatten()
            for j, ax in enumerate(axs):
                plot_maps.add_map_ukraine(ax=ax, extent=extent, add_fill=False)
                grid = grid_list[i][grid_dirs[j]]
                pc = ax.pcolormesh(
                    LONS,
                    LATS,
                    grid,
                    shading="auto",
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                    cmap=cmaps.divergent_cmap(),
                    vmin=vlims[i][0],
                    vmax=vlims[i][1],
                )
                ax.set_title(f"{grid_dirs[j]}")
            fig.colorbar(pc, ax=axs, label="Ducting fraction", shrink=0.7, pad=0.015)
            fig.suptitle(f"{name}: {date.strftime('%Y-%m-%dT%H')}")
            plt.show(block=False)
        plt.show(block=False)

    def plot_grids(self, date: UTCDateTime, grid_dir: str, N_voronoi: int = 3, gw: bool = True):
        start_time = time.perf_counter()
        results = self.compute_ducting_heights_grid(date, grid_dir, N_voronoi=N_voronoi, gw=gw)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for calculating grid = {elapsed_time:.4f} seconds")
        lons, lats = results["lons"], results["lats"]
        grid_mean = results["grid_mean"]
        grid_tropos = results["grid_tropos"]
        grid_stratos = results["grid_stratos"]
        grid_thermos = results["grid_thermos"]
        grid_escape = results["grid_escape"]
        LONS, LATS = np.meshgrid(lons, lats)
        extent = [
            lons.min(),
            lons.max(),
            lats.min(),
            lats.max(),
        ]
        grid_list = [grid_mean, grid_tropos, grid_stratos, grid_thermos, grid_escape]
        names = ["mean", "tropos", "stratos", "thermos", "escape"]
        vlims = [(0, 150), (0, 0.4), (0, 0.6), (0.3, 0.9), (0, 0.3)]
        for i, grid in enumerate(grid_list[:]):
            name = names[i]
            fig, ax = plot_maps.add_map_ukraine(extent=extent, add_fill=False, ukr_lw=1.5)
            pc = ax.pcolormesh(
                LONS,
                LATS,
                grid,
                shading="auto",
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                cmap=cmaps.divergent_cmap(),
                # vmin=vlims[i][0],
                # vmax=vlims[i][1],
            )
            fig.colorbar(pc, label="Ducting fraction", shrink=0.7, pad=0.015)
            ax.set_title(f"{name}: {date.strftime('%Y-%m-%dT%H')}")
            plt.show(block=False)
        plt.show(block=False)


if __name__ == "__main__":
    taup = TauP()
    grid_dir = "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids"

    # grid_dir = "../../test_scripts/data/ukraine_grids/"
    # grid_date = UTCDateTime("2022-03-07T16:00:00")
    grid_date = UTCDateTime("2022-02-01T12:00:00")
    # taup_grid.compute_ducting_heights_grid(grid_date, grid_dir)
    # atm_profile = "/Users/sophus/code/norsar/ismonpy/test_scripts/data/ukraine_grids/grid_2022-03-07T16/profiles/g2stxt_2022030716_43.0000_27.5000.dat"
    # atm_profile = "/Users/sophus/code/norsar/ismonpy/test_scripts/data/ukraine_grids/grid_2022-03-07T16/profiles/g2stxt_2022030716_51.5000_28.0000.dat"
    # profile_dir = "/Users/sophus/code/norsar/ismonpy/test_scripts/data/ukraine_grids/grid_2022-03-07T16/profiles/"
    profile_dir = "../../test_scripts/data/MAAG/"
    
    # lat = 51.5
    # lon = 28
    # taup.plot_grids(grid_date, grid_dir, N_voronoi=3, gw=False)
    # taup.plot_grids_azimuthal(grid_date, grid_dir, N_voronoi=2, gw=False)
    # plt.show()

    # atm_profile = "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids/grid_2022-03-01T12/profiles/g2stxt_2022030112_44.0000_22.5000.dat"
    # atm_profile = ( "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids/grid_2022-02-01T12/profiles/g2stxt_2022020112_50.0000_29.0000.dat")
    # atm_profile = (
    #     "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids/grid_2022-02-01T12/profiles/g2stxt_2022020112_47.5000_34.0000.dat"
    # )

    station = "MAAG"
    start_date = UTCDateTime("2022-03-01T00:00:00")
    end_date = start_date + timedelta(hours=12)
    latlons = {
        "MAAG": (50.7014, 29.2301),
        "GRDI": (50.5993, 29.4471),
    }
    lat, lon = latlons[station]
    atmos_dir = f"/staff/sophus/Documents/ismonpy/test_scripts/data/{station.upper()}/"
    h_resol = 6

    inc_degs = np.array([80])
    inc_degs_range = 90 - np.arange(5, 65, 1)
    az_degs = np.arange(-180, 180, 1)
    az_degs, inc_degs = np.meshgrid(az_degs, inc_degs)
    az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()
    
    atm_profiles, atm_dates = atmos_files.list_g2s_files(atmos_dir, lat, lon, start_date, end_date, h_resol)
    for atm_profile, date in zip(atm_profiles, atm_dates):
        print(f"Runnign for {date = }")
        # savefile = f"/staff/sophus/Documents/ismonpy/test_scripts/figures/maag_profile_gif_march/{station.upper()}_{date.strftime('%Y-%m-%dT%H')}.png"
        # savefile = f"/staff/sophus/Documents/ismonpy/test_scripts/figures/maag_ray_fan_gif_march/{station.upper()}_{date.strftime('%Y-%m-%dT%H')}.png"

        atm_profile = os.path.join(atmos_dir, atm_profile)
        title = f"{station.upper()} at {date.strftime('%Y-%m-%dT%H')}"
        taup.plot_atm_profile(atm_profile, gw=False, title=title, savefile=None)
        taup.plot_ray_paths(az_deg=90, inc_degs=inc_degs_range, atm_profile=atm_profile, gw=False)
        # taup.plot_ray_paths_2d(az_degs=az_degs, inc_degs=inc_degs, atm_profile=atm_profile, lat0=lat, lon0=lon, gw=False, only_arrivals=False, savefile=None, title=title)
        plt.show()
        # taup.plot_atm_profile(atm_profile, gw=True)

    # atm_profiles = [os.path.join(profile_dir + fname) for fname in os.listdir(profile_dir)]
    # taup.plot_distributions(atm_profiles, N_voronoi=3, gw=False)
    # taup.plot_distributions(atm_profiles, N_voronoi=4, gw=True)
    # plt.show()
    
    # taup.plot_ray_paths(az_deg=10, inc_degs=np.arange(45, 90, 0.5), atm_profile=atm_profile, gw=False)
    # taup.plot_ray_paths(az_deg=120, inc_degs=np.arange(45, 90, 0.5), atm_profile=atm_profile, gw=True)
    # plt.show()    

    # inc_degs = np.array([65])
    # az_degs = np.arange(-180, 180, 1)
    # az_degs, inc_degs = np.meshgrid(az_degs, inc_degs)
    # az_degs, inc_degs = az_degs.flatten(), inc_degs.flatten()
    #
    # atm_profiles = atmos_files.list_g2s_files(atmos_dir, lat, lon, start_date, start_date, h_resol=h_resol)
    # atm_profile = os.path.join(atmos_dir, atm_profiles[0][0]) 
    #
    # atm_profile = ( "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids/grid_2022-02-01T12/profiles/g2stxt_2022020112_50.0000_29.0000.dat")
    # taup.plot_ray_paths_2d(az_degs=az_degs, inc_degs=inc_degs, atm_profile=atm_profile, gw=False, only_arrivals=False)
    # plt.show()
    # # taup.get_eigenrays(atm_profile, az_degs=az_degs, inc_degs=inc_degs, plot=True, gw=False)
    # taup.plot_arrival_map(atm_profile, az_degs, inc_degs, gw=False, multiple_bounces=False)
    # plt.show()
    #
    # # G2S = g2s_profile.G2Sprofile(lat, lon, grid_dir)
    # # G2S.plot_atm_profile(UTCDateTime("2022-03-07T16:00:00"), grid_mode=True)
    # # G2S.get_grid(date=grid_date, grid_dir=grid_dir)
    # # G2S.plot_grid(grid_date, grid_dir)
    
