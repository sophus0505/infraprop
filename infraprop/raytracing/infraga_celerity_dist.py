import os
import subprocess
import time
from datetime import timedelta

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from infraview import atmos
from pyproj import Geod

from infraprop.utils import plot_maps, atmos_files


class CelerityDistribution:
    # Class for calculating a celerity distribution from a single station over a given time window.
    # Before running you need to install infraGA to the current environment (pip install -e ../infraGA/.)
    # Before running it is necessary to have atmospheric profiles available in the ncpaprop or infraGA
    # expected format --> See the atmos module for wrapper around G2S (and future ERA5)

    def __init__(self, st_lat: float, st_lon: float):

        self.st_lat = st_lat
        self.st_lon = st_lon

    import os

    def get_file_name(self, date):
        return atmos_files.g2s_fname(date, self.st_lat, self.st_lon)

    def launch_rays_3d_multiple(
        self, atmos_dir, start_time, end_time, h_resol, overwrite=False
    ):
        files, times = atmos_files.list_g2s_files(
            atmos_dir, self.st_lat, self.st_lon, start_time, end_time, h_resol
        )
        for date in times:
            self.launch_rays_3d(atmos_dir, date, overwrite)

    def launch_rays_3d(self, atmos_dir, date, overwrite=False):
        fname = self.get_file_name(date)
        arrivals = os.path.join(atmos_dir, fname.removesuffix(".met") + ".arrivals.dat")
        if os.path.exists(arrivals) and not overwrite:
            print(f"Arrivals already computed for: {fname}")
            return

        nproc = 8
        cmd = [
            "infraga",
            "sph",
            "prop",
            "--atmo-file",
            os.path.join(atmos_dir, fname),
            "--incl-min",
            "1",
            "--incl-max",
            "50",
            "--incl-step",
            "1.5",
            "--az-min",
            "-180",
            "--az-max",
            "180",
            "--az-step",
            "5",
            "--max-rng",
            "500",
            "--src-lat",
            str(self.st_lat),
            "--src-lon",
            str(self.st_lon),
            # "--bounces", "1",# I am not sure it makes sense to restric this? Would it not bias towards lower celerities?
            "--bounces",
            "10",
            "--freq",
            "3",
            "--cpu-cnt",
            str(nproc),
        ]
        subprocess.run(cmd, check=False)

    def launch_rays_1d(self, atm_profile, az, overwrite=False, plot=True):
        atm_name, atm_suffix = os.path.splitext(atm_profile)
        arrivals = atm_name + ".arrivals.dat"
        raypaths = atm_name + ".raypaths.dat"

        if os.path.exists(arrivals) and not overwrite:
            print(f"Arrivals already computed for: {atm_profile}")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)

            src_loc = [50, 29] 
            nproc = 1 
            cmd = [ "infraga",
                "sph",
                "prop",
                "--atmo-file", atm_profile,
                "--incl-min",
                "0.5",
                "--incl-max",
                "45",
                "--incl-step",
                "0.5",
                "--azimuth", str(az), 
                "--max-rng",
                "600",
                "--src-lat",
                str(src_loc[0]),
                "--src-lon",
                str(src_loc[1]),
                # "--bounces", "1",# I am not sure it makes sense to restric this? Would it not bias towards lower celerities?
                "--bounces",
                "0",
                "--freq",
                "3",
                "--cpu-cnt",
                str(nproc),
                "--write-rays", "true",
            ]
            start_time = time.perf_counter()
            subprocess.run(cmd, check=False)
            end_time = time.perf_counter()
            tot_time = end_time - start_time 
        
            # lat [deg]	lon [deg]	z [km]	trans. coeff. [dB]	absorption [dB]	time [s]
            sph_proj = Geod(ellps='sphere')
            ray_data = np.loadtxt(raypaths)

            ray_rngs = sph_proj.inv([src_loc[1]] * len(ray_data), [src_loc[0]] * len(ray_data), ray_data[:, 1], ray_data[:, 0])[2] * 1.0e-3
            ray_alts = ray_data[:, 2]
            
            ax.scatter(
                ray_rngs, 
                ray_alts, 
                s=0.025,
                alpha=0.9,
            )

            cmd = [
                "infraga",
                "sph",
                "prop",
                "--atmo-file", atm_profile,
                "--incl-min",
                "0.5",
                "--incl-max",
                "45",
                "--incl-step",
                "0.5",
                "--azimuth", str((az + 360) % 360 - 180), 
                "--max-rng",
                "600",
                "--src-lat",
                str(src_loc[0]),
                "--src-lon",
                str(src_loc[1]),
                # "--bounces", "1",# I am not sure it makes sense to restric this? Would it not bias towards lower celerities?
                "--bounces",
                "0",
                "--freq",
                "3",
                "--cpu-cnt",
                str(nproc),
                "--write-rays", "true",
            ]
            start_time = time.perf_counter()
            subprocess.run(cmd, check=False)
            end_time = time.perf_counter()
            tot_time += end_time - start_time

            print(f"Used a total of {tot_time:.3f} s. to compute rays")
            print(f"Resolution: {ray_rngs.size / 90})")

            ray_data = np.loadtxt(raypaths)

            ray_rngs = sph_proj.inv([src_loc[1]] * len(ray_data), [src_loc[0]] * len(ray_data), ray_data[:, 1], ray_data[:, 0])[2] * 1.0e-3
            ray_alts = ray_data[:, 2]
            ax.scatter(
                -ray_rngs, 
                ray_alts, 
                s=0.025,
                alpha=0.9,
            )

            ax.set_xlabel("Range [km]")
            ax.set_ylabel("Altitude [km]")
            ax.set_title(f"InfraGA ray fan: az=+-{az}°")
            ax.set_ylim(0, 150)
            ax.set_xlim(-450, 750)

            plt.show()


    

    def load_arrivals(self, atmos_dir, start_time, end_time, h_resol):
        # Load the arrivals in a given time-period
        # Format from infraGA: name [unit][idx]
        # incl [deg][0]	 az [deg][1]  n_b [][2]  lat_0 [deg][3]	 lon_0 [deg][4]  time [s][5]  cel [m/s][6]  turning_ht [km][7]  inclination [deg][8]  back_azimuth [deg][9]  trans. coeff. [dB][10]  absorption [dB][11]
        files, times = atmos_files.list_g2s_files(
            atmos_dir, self.st_lat, self.st_lon, start_time, end_time, h_resol
        )
        arrivals = []
        for f, t in zip(files, times):
            arrival_path = os.path.join(
                atmos_dir, f.removesuffix(".met") + ".arrivals.dat"
            )
            if not os.path.exists(arrival_path):
                raise Warning(f"Did not find: {arrival_path}")

            print(arrival_path)
            arrival_data = np.loadtxt(arrival_path, usecols=range(12))
            arrival_data[:, 6] = arrival_data[:, 6] * 1000  # convert to m/s
            # th_mask = arrival_data[:, 7] < 100
            # cel_mask = arrival_data[:, 6] > 200
            bounce_mask = arrival_data[:, 2] == 0
            TL_mask = arrival_data[:, 10] > -40
            mask = bounce_mask & TL_mask
            # mask = arrival_data[:, 2] == 0 # only include the first bounce
            arrival_data = arrival_data[mask]
            arrivals.append(arrival_data)

        return times, arrivals

    def get_total_celerity_dist(self, atmos_dir, start_time, end_time, h_resol):
        import cartopy.crs as ccrs

        times, arrivals = self.load_arrivals(atmos_dir, start_time, end_time, h_resol)
        celerities = []
        lats = []
        lons = []
        TLs = []
        turning_heights = []
        azs = []
        bazs = []

        for arr in arrivals:
            celerities += list(arr[:, 6])
            azs += list(arr[:, 1])
            bazs += list(arr[:, 9])
            lats += list(arr[:, 3])
            lons += list(arr[:, 4])
            TLs += list(arr[:, 10])
            turning_heights += list(arr[:, 7])

        # extent = [self.st_lon - 9,
        #           self.st_lon + 9,
        #           self.st_lat - 5,
        #           self.st_lat + 5]
        extent = [20, 42, 43, 54]
        fig, ax = plot_maps.add_map_ukraine(extent=extent)
        ax.set_title("Bounce locations")
        sc = ax.scatter(
            lons,
            lats,
            c=bazs,
            cmap=cmc.batlow_r,
            # vmin=200,
            # vmax=350,
            s=1,
            alpha=0.6,
            zorder=1000,
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(sc, shrink=0.7, pad=0.015, label="Celerity [m/s]")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.show(block=False)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axs = axs.flatten()

        axs[0].set_title("Transmission Losses (3 Hz)")
        tl_bins = np.linspace(-100, 0, 50)
        axs[0].hist(TLs, bins=tl_bins)
        axs[0].set_xlabel("TL [dB]")
        axs[0].set_ylabel("Count")

        axs[1].set_title("Turning heights")
        th_bins = np.linspace(0, 150, 50)
        axs[1].hist(turning_heights, bins=th_bins)
        axs[1].set_xlabel("Turning height [km]")
        axs[1].set_ylabel("Count")

        axs[2].set_title("Celerities")
        cel_bins = np.linspace(150, 400, 50)
        axs[2].hist(celerities, bins=cel_bins)
        axs[2].set_xlabel("Celerity [m/s]")
        axs[2].set_ylabel("Count")
        plt.show()


if __name__ == "__main__":
    station = "MAAG"
    start_time = UTCDateTime("2022-03-01T00:00:00")
    end_time = start_time + timedelta(days=7)
    latlons = {
        "MAAG": (50.7014, 29.2301),
        "GRDI": (50.5993, 29.4471),
    }
    station = "MAAG"
    lat, lon = latlons[station]
    atmos_dir = f"/staff/sophus/Documents/ismonpy/test_scripts/data/{station.upper()}/"
    h_resol = 6

    CelDist = CelerityDistribution(lat, lon)

    
    atm_profile = ( "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids/grid_2022-02-01T12/profiles/g2stxt_2022020112_50.0000_29.0000.dat")

    # CelDist.launch_rays_1d(atm_profile, az=10, overwrite=True, plot=True)

    
    # CelDist.launch_rays_3d_multiple(atmos_dir, start_time, end_time, h_resol, overwrite=False)
    # CelDist.get_total_celerity_dist(atmos_dir, start_time, end_time, h_resol)














    
