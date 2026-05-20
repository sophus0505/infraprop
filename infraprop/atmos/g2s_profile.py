import os
import re
import subprocess
import warnings
from datetime import timedelta

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime

from infraprop.utils import cmaps, atmos_files


class G2Sprofile:
    # Wrapper class to retrieve G2S data for a single vertical profile above (lat, lon)
    # NB: You need to make sure ncpag2s.py is in the path --> add ncpag2s-clc dir to path and chmod ncpag2s.py

    def __init__(
        self,
        lat: float | None = None,
        lon: float | None = None,
        atmos_dir: str | None = None,
        format: str = "zTuvdp",
    ):

        self.lat = lat
        self.lon = lon
        self.atmos_dir = atmos_dir
        self.format = format

    def profile_exist(self, date, atmos_dir):
        return os.path.exists(os.path.join(atmos_dir, atmos_files.g2s_fname(date, self.lat, self.lon)))

    def get_single_profile_path(self, date, lat=None, lon=None):
        lat = lat if lat is not None else self.lat
        lon = lon if lon is not None else self.lon
        if self.profile_exist(date, self.atmos_dir):
            return os.path.join(self.atmos_dir, atmos_files.g2s_fname(date, lat, lon))

    def get_single_profile(self, date, output_dir=None):
        # call G2S to get a single profile at (self.lat, self.lon) at time: date
        if output_dir is None:
            raise Warning("Output dir is missing! Cannot call G2S...")
        if (self.lat is None) or (self.lon is None) or (self.atmos_dir is None):
            raise Warning("One of lat, lon or atmos_dir is not set!")

        output_str = os.path.join(output_dir, atmos_files.g2s_fname(date, self.lat, self.lon))

        if not self.profile_exist(date, output_dir):
            cmd = [
                "ncpag2s.py",
                "point",
                "--date",
                date.strftime("%Y-%m-%d"),
                "--hour",
                str(date.hour),
                "--lat",
                str(self.lat),
                "--lon",
                str(self.lon),
                "--outputformat",
                "ncpaprop",
                "--output",
                output_str,
            ]
            subprocess.run(cmd, check=True)
            print(f"Created profile: {output_str}")
        else:
            print("Profile already exists at:")
            print(f"{output_str}")
        return output_str

    def get_multiple_profiles(self, start_date, end_date, h_resol):
        # call G2S to get multiple profiles from (self.lat, self.lon) between start and end date using hourly resolution h_resol
        # The results are stored in the atmos dir using the custom format (g2s_fname()) that I think is easier to read on a glance

        existing_files, existing_dates = atmos_files.list_g2s_files(
            self.atmos_dir,
            self.lat,
            self.lon,
            start_date,
            end_date,
            ignore_warnings=True,
        )
        if (self.lat is None) or (self.lon is None) or (self.atmos_dir is None):
            raise Warning("One of lat, lon or atmos_dir is not set!")

        if len(existing_dates) > 0:
            print(f"Existing dates: {existing_dates}")
            raise Warning("Some profiles already exist for this location in the requested time window.")

        cmd = [
            "ncpag2s.py",
            "point",
            "--lat",
            str(self.lat),
            "--lon",
            str(self.lon),
            "--startdate",
            start_date.strftime("%Y-%m-%d"),
            "--starthour",
            str(start_date.hour),
            "--enddate",
            end_date.strftime("%Y-%m-%d"),
            "--endhour",
            str(end_date.hour),
            "--every",
            str(h_resol),
            "--outputformat",
            "ncpaprop",
            "--output",
            self.atmos_dir,
        ]
        subprocess.run(cmd, check=True)

        for fname in os.listdir(self.atmos_dir):
            if not fname.endswith(".met"):
                continue
            date, lat, lon = atmos_files.parse_ncpag2s_fname(fname)
            if date and lat and lon:
                if (lat == self.lat) and (lon == self.lon) and (start_date <= date <= end_date):
                    os.rename(
                        os.path.join(self.atmos_dir, fname),
                        os.path.join(self.atmos_dir, atmos_files.g2s_fname(date, self.lat, self.lon)),
                    )

    def get_grid(self, date: UTCDateTime, grid_dir: str, extent: list[int] | None = None):
        # extent should be same as used for cartopy: [lon1, lon2, lat1, lat2]
        # The resolution is set to 0.5 degrees (roughly matching Merra2 which is 0.5x0.625)
        dirname = f"grid_{date.strftime('%Y-%m-%dT%H')}/"
        new_dir = os.path.join(grid_dir, dirname)

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        elif len(os.listdir(new_dir)) > 0:
            # raise Warning(f"Files already exist in: {new_dir}")
            print(f"Found grid in {new_dir}, returning path...")
            return new_dir

        if extent is None:
            # set a default Ukraine extent that also cover Romania
            extent = [20, 42, 43, 54]

        start_lon, end_lon = int(extent[0]), int(extent[1])
        start_lat, end_lat = int(extent[2]), int(extent[3])

        n_lons = int(np.abs(end_lon - start_lon)) * 2 + 1
        n_lats = int(np.abs(end_lat - start_lat)) * 2 + 1

        # breakpoint()
        cmd = [
            "ncpag2s.py",
            "grid",
            "--date",
            date.strftime("%Y-%m-%d"),
            "--hour",
            str(date.hour),
            "--startlongitude",
            str(start_lon),
            "--endlongitude",
            str(end_lon),
            "--startlatitude",
            str(start_lat),
            "--endlatitude",
            str(end_lat),
            "--lonpoints",
            str(n_lons),
            "--latpoints",
            str(n_lats),
            "--output",
            new_dir,
            "--outputformat",
            "ncpaprop",
        ]

        subprocess.run(cmd, check=True)

    def check_time(self, date):
        # Check that the time-location is available (not stable?)
        check_cmd = [
            "python",
            "/staff/sophus/packages/ncpag2s-clc/ncpag2s.py",
            "checktime",
            "--date",
            date.strftime("%Y-%m-%d"),
            "--hour",
            str(date.hour),
        ]
        subprocess.run(check_cmd, check=True)

    def _get_z0(self, atm_profile):
        # get z0 in km from file
        if not os.path.exists(atm_profile):
            raise Warning("Atm profile does not exist!")
        z0 = 0
        with open(atm_profile, "r") as infile:
            for line in infile:
                if "Ground Height" in line:
                    z0 = float(re.findall(r"\d+\.\d+", line)[0])
                    break
                elif line[0] != "#":
                    warnings.warn(f"No ground height found in file: {atm_profile}")
                    break
        return z0

    def plot_grid(self, date, 
                  atmos_dir, 
                  extent=None, 
                  altitude=10, 
                  station_type="is",
                  vmin=None, 
                  vmax=None, 
                  savename=None, 
                  block=True):
        import cartopy.crs as ccrs

        if extent is None:
            # set a default Ukraine extent that also cover Romania
            extent = [20, 42, 43, 54]

        grid_dir = atmos_files.find_g2s_grid(date, atmos_dir)

        summary_file = os.path.join(grid_dir, "summary.dat")

        coords = np.loadtxt(summary_file, usecols=(0, 1))
        fnames = np.loadtxt(summary_file, usecols=2, dtype=str)

        lats, lons = coords[:, 0], coords[:, 1]

        unique_lats = np.unique(lats)
        unique_lons = np.unique(lons)

        lat_idx = np.searchsorted(unique_lats, lats)
        lon_idx = np.searchsorted(unique_lons, lons)

        LONS, LATS = np.meshgrid(unique_lons, unique_lats)

        U = np.zeros((unique_lats.size, unique_lons.size))
        V = np.zeros((unique_lats.size, unique_lons.size))

        for k, fname in enumerate(fnames):
            path = os.path.join(grid_dir, fname)
            data = np.loadtxt(path)

            z, t, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

            idx = np.argmin(np.abs(z - altitude))

            i = lat_idx[k]
            j = lon_idx[k]

            U[i, j] = u[idx]
            V[i, j] = v[idx]

            # print(unique_lats[i], unique_lons[j], u[idx], v[idx])

        speed = np.sqrt(U**2 + V**2)

        fig, ax = add_map_ukraine(extent=extent, station_type=station_type, add_fill=False)
        pc = ax.pcolormesh(
            LONS,
            LATS,
            speed,
            shading="auto",
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            cmap=cmaps.divergent_cmap(),
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        cbar = fig.colorbar(pc, label="Wind speed [m/s]", shrink=0.7, pad=0.015)
        cbar.ax.tick_params(labelsize=12) 

        M = 3
        ax.quiver(
            LONS,
            LATS,
            U,
            V,
            transform=ccrs.PlateCarree(),
            scale=vmax*5,
            angles="xy",
            pivot="mid",
            alpha=0.75,
            regrid_shape=8,
            rasterized=True,

        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{date.strftime('%Y-%m-%dT%H')}: Winds at {altitude} km altitude")
        if savename is not None: 
            plt.savefig(savename)
        plt.show(block=block)

    def get_atmo_vars(self, atm_profile, include_z0=False):
        atmo = np.loadtxt(atm_profile)
        z = atmo[:, self.format.find("z")]
        u = atmo[:, self.format.find("u")]
        v = atmo[:, self.format.find("v")]
        c = np.sqrt(0.14 * atmo[:, self.format.find("p")] / atmo[:, self.format.find("d")])

        if include_z0:
            z0 = self._get_z0(atm_profile)
            z = z + z0

        return z, u, v, c


    def plot_atm_profile(self, date, grid_mode=False):
        # Plot the atm profile
        # Stolen from infraGA :)
        if (self.lat is None) or (self.lon is None) or (self.atmos_dir is None):
            raise Warning("One of lat, lon or atmos_dir is not set!")

        if grid_mode:
            grid_dir = atmos_files.find_g2s_grid(date, self.atmos_dir)

            summary_file = os.path.join(grid_dir, "summary.dat")

            coords = np.loadtxt(summary_file, usecols=(0, 1))
            fnames = np.loadtxt(summary_file, usecols=2, dtype=str)
            atm_idx = np.argwhere((coords[:, 0] == self.lat) & (coords[:, 1] == self.lon))[0][0]
            output_str = os.path.join(grid_dir, fnames[atm_idx])
        else:
            output_str = os.path.join(self.atmos_dir, atmos_files.g2s_fname(date, self.lat, self.lon))
        if not os.path.exists(output_str):
            raise Warning(f"Could not find atm: {output_str}")

        atmo = np.loadtxt(output_str)
        z = atmo[:, self.format.find("z")]
        u = atmo[:, self.format.find("u")]
        v = atmo[:, self.format.find("v")]
        c = np.sqrt(0.14 * atmo[:, self.format.find("p")] / atmo[:, self.format.find("d")])

        z0 = self._get_z0(output_str)
        z = z + z0

        f, ax = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 4]}, figsize=(12, 5))
        f.suptitle(f"Lat: {self.lat:.3f}, lon: {self.lon:.3f}, date: {date.strftime('%Y-%m-%dT%H:%M:%S')}")
        ax[0].grid(color="k", linestyle="--", linewidth=0.5)
        ax[1].grid(color="k", linestyle="--", linewidth=0.5)

        ax[0].set_ylim(z[0], z[-1])
        ax[0].set_ylabel("Altitude [km]")
        ax[0].set_xlabel("Sound Speed [m/s]")

        ax[1].set_ylim(z[0], z[-1])
        ax[1].set_xlabel("Wind Speed [m/s]")

        ax[1].yaxis.set_ticklabels([])

        ax[2].yaxis.set_label_position("right")
        ax[2].yaxis.tick_right()

        ax[2].set_xlim(-180.0, 180.0)
        ax[2].set_ylim(0.0, 50.0)
        ax[2].set_xticks((-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0))
        ax[2].set_xticklabels(["S", "SW", "W", "NW", "N", "NE", "E", "SE", "S"])
        ax[2].set_xlabel("Propagation Direction")
        ax[2].set_ylabel("Inclination [deg]")

        ax[0].plot(c, z, "-k", linewidth=3.0)
        ax[1].plot(u, z, "indigo", linewidth=3.0, label="Zonal")
        ax[1].plot(v, z, "orangered", linewidth=3.0, label="Merid.")
        ax[1].legend(fontsize="small")

        incl_vals = np.arange(0.0, 50.0, 0.2)
        for az in np.arange(-180.0, 180.0, 1.0):
            ceff = c + u * np.sin(np.radians(az)) + v * np.cos(np.radians(az))
            refract_ht = [
                (
                    z[np.min(np.where((ceff / ceff[0]) * np.cos(np.radians(incl)) > 1.0)[0])]
                    if len(np.where((ceff / ceff[0]) * np.cos(np.radians(incl)) > 1.0)[0]) > 0
                    else z[-1]
                )
                for incl in incl_vals
            ]
            sc = ax[2].scatter(
                [az] * len(refract_ht),
                incl_vals,
                c=refract_ht,
                marker="s",
                s=5.0,
                cmap=cmc.batlow,
                alpha=0.75,
                edgecolor="none",
                vmin=z[0],
                vmax=120.0,
            )

        f.colorbar(sc, ax=[ax[2]], location="top", label="Estimated Refraction Altitude [km]")

        plt.show()


if __name__ == "__main__":
    start_date = UTCDateTime("2022-04-01T06:00:00")
    # end_date = start_date + timedelta(weeks=4)
    end_date = UTCDateTime("2022-04-08T06:00:00") - timedelta(hours=6)

    h_resol = 6 
    # start_date = UTCDateTime("2022-03-02T00:00:00") + timedelta(hours=6)
    # end_date = start_date + timedelta(days=5, hours=18)
    
    h_resol = 6
    latlons = {
        "MAAG": (50.7014, 29.2301),
        "GRDI": (50.5993, 29.4471),
    }
    station = "MAAG"
    # station = "GRDI"
    lat, lon = latlons[station]

    atmos_dir = f"/staff/sophus/Documents/ismonpy/test_scripts/data/{station.upper()}"
    G2S = G2Sprofile(lat, lon, atmos_dir)
    G2S.plot_atm_profile(start_date)
    # G2S.get_multiple_profiles(start_date, end_date, h_resol)
    # grid_dir = "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids"

    # grid_dir = "../../test_scripts/data/ukraine_grids/"
    # save_dir = "/staff/sophus/Documents/ismonpy/test_scripts/figures/ukraine_wind_maps"
    # for month in ["02", "03", "04"]:
    #     for day in ["01", "15"]:
    #         grid_date = UTCDateTime(f"2022-{month}-{day}T12:00:00")
    #         G2S = G2Sprofile()
    #         # G2S.get_single_profile(start_date, atmos_dir)
    #         # G2S.get_multiple_profiles(start_date, end_date, h_resol)
    #         G2S.get_grid(date=grid_date, grid_dir=grid_dir)
    #         for altitude in [10, 25]:
    #             savename = os.path.join(save_dir, f"2022-{month}-{day}T12_alt{altitude}km.pdf")
    #             G2S.plot_grid(grid_date, grid_dir, altitude=altitude, station_type="seismic", savename=savename, block=False, vmin=0, vmax=60)
    # plt.show()

    # G2S.check_time()
    # G2S.get_profile(atmos_dir="/staff/sophus/Documents/ismonpy/test_scripts/data/MAAG/")

    # G2S.plot_atm_profile()
