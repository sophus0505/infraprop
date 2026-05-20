import os
import subprocess
import time
from datetime import timedelta

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from pyproj import Geod
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture as GMM

from infraprop.raytracing import tau_p
from infraprop.utils import atmos_files, plot_maps


class PropagationPriors:
    # Class for calculating propagation priors for infrasound association and location
    # Made to overwrite the methods in infraGA such that we can plug and play the tau-p methods for use in infraGA without writing to files. 
    # core: generate distributions over celerity and azimuth for a given station
    # This class uses the tau-p to calculate rays (see ./tau_p.py) for implementation
    # expected format --> See the atmos module for wrapper around G2S (and future ERA5)

    def __init__(self, st_lat: float, st_lon: float):
        self.st_lat = st_lat
        self.st_lon = st_lon

        self.has_distributions = False
        self.cel_vals = None 
        self.az_vals = None
        self.az_degs = None



    def get_file_name(self, date):
        return atmos_file_handling.g2s_fname(date, self.st_lat, self.st_lon)

    def get_az_mask(self, az, az_rng=60): 
        # get the mask towards an azimuth az, such that azimuths in (az - az_rng, az + az_rng) 
        if self.az_degs is None: 
            raise Warning("Cannot calculate azimuthal mask without first having a distribution")

        az_min = (az + 360 - az_rng) % 360
        az_max = (az + 360 + az_rng) % 360
        azimuths = (self.az_degs + 360) % 360
        if az_min <= az_max:
            mask = (azimuths >= az_min) & (azimuths <= az_max) 
        else: 
            mask = (azimuths >= az_min) | (azimuths <= az_max)
        return mask

    def get_rcel_gmm(self, az=None, app_vel=None, slowness_domain=True, plot=False): 
        # Get a three component Gaussian mixture model 
        # If slowness_domain = True --> convert everything to the slowness domain as is done in infraPy 
        
        if self.cel_vals is not None:
            cel_vals = self.cel_vals 
        else: 
            raise Warning("No distribution calculated/read for priors... Calculate first!")
    
        if az is not None: 
            mask = self.get_az_mask(az)
            if np.any(mask):
                cel_vals = cel_vals[mask]


        # canon_rcel_mns = np.array([1.0 / 0.335, 1.0 / 0.315, 1.0 / 0.26])
        means_init = np.array([260, 315, 335]).reshape(-1, 1)
        if slowness_domain: 
            cel_vals = 1000 / cel_vals 
            means_init = 1000 / means_init

        reg_covar_factor = 0.5
        reg_covar = (reg_covar_factor * np.std(cel_vals))**2
        print(f"{az = }, {reg_covar = }")

        gauss_mix = GMM(n_components=3, covariance_type="full", reg_covar=reg_covar , means_init=means_init).fit(cel_vals.reshape(-1, 1))
        weights = gauss_mix.weights_.flatten() 
        means = gauss_mix.means_.flatten() 
        vars = np.sqrt(gauss_mix.covariances_.flatten())
        
        if plot:
            plt.figure()
            plt.title(f"{az = }")
            
            if slowness_domain:
                x_vals = np.linspace(1000/360, 1000/200, 1000).reshape(-1, 1)
                plt.xlabel('Slowness [s/km]')
            else:
                x_vals = np.linspace(200, 360, 1000).reshape(-1, 1)
                plt.xlabel('Celerity [m/s]')

            gmm_y = np.exp(gauss_mix.score_samples(x_vals))
            plt.plot(x_vals, gmm_y, linewidth=4.0, color='Blue')
            
            # Individual components
            for i in range(3):
                comp_y = weights[i] * norm.pdf(x_vals.flatten(), loc=means[i], scale=vars[i])
                plt.plot(x_vals, comp_y, linewidth=2.0, color='Green')
            
            plt.show()

        return weights, means, vars


    def get_az_deviation_mean_var(self, az=None):
        # get the mean and variance of the azimuth deviations
        if self.az_dev_vals is not None:
            az_dev_vals = self.az_dev_vals
        else: 
            raise Warning("No distribution calculated/read for priors... Calculate first!")

        if az is not None: 
            mask = self.get_az_mask(az)
            if np.any(mask):
                az_dev_vals = az_dev_vals[mask]

        az_dev_mean = np.mean(az_dev_vals)
        az_dev_var = np.var(az_dev_vals)

        return az_dev_mean, az_dev_var





    def calculate_distributions(self, atm_dir, start_date, end_date, h_resol=None, N_voronoi=3, filter=True, gw=False):
        #It should be filtered after based on the current azimuth and/or inclination, app. vel. or similar... to be decided 
        # Calculate the distributions in azimuth and celerity based on Tau-p rays. 
        # NOTE: This method calculates for an even distribution of rays over azimuth and inclination, based on a tesselated sphere (N_voronoi controls resolution)
        atm_profiles, _ = atmos_file_handling.list_g2s_files(atm_dir, self.st_lat, self.st_lon, start_date, end_date, h_resol, ignore_warnings=True)
            
        cel_vals = []
        az_dev_vals = []
        az_degs = []
        TauP = tau_p.TauP()
        start_date = time.perf_counter()
        for atm_profile in atm_profiles:
            atm_profile = os.path.join(atm_dir, atm_profile)
            if "arrivals" in atm_profile:
                continue 
            elif "raypaths" in atm_profile:
                continue
            elif ".met" not in atm_profile: 
                continue
            print(f"processing {os.path.basename(atm_profile)}")
            n_reals = 10 if gw else 1
            
            for i in range(n_reals):
                taup_results = TauP.evaluate_tau_p_integrals_profile(atm_profile, N_voronoi=N_voronoi, gw=gw)
                mask = ~np.isnan(taup_results["ducting_heights"])
                if filter: # filter out unwanted rays 
                    mask = np.logical_and(mask, taup_results["ducting_heights"] < 100)

                cel_vals += list(taup_results["celerities"][mask].flatten())
                az_dev_vals += list(taup_results["az_deviations"][mask].flatten())
                az_degs += list(taup_results["az_degs"][mask].flatten())

        end_date = time.perf_counter()
        print(f"Finished calculating distributions after {end_date - start_date:.3f} s.")
        
        self.cel_vals = np.array(cel_vals)
        self.az_dev_vals = np.array(az_dev_vals)    
        self.az_degs = np.array(az_degs)
        self.has_distributions = True
        
    
    def plot_1d_dists(self, az=None, block=True, gmm=True, title=None): 
        # Calculate celerity and azimuth distributions based on the Tau-p rays from the location of the station (self.st_lat, self.st_lon)
        # TODO: Add the possibility of filtering the values based on priors such as az, app cel, and turning height
        
        if not self.has_distributions:
            raise Warning("Need to calculate/read distributions before plotting!")

        cel_vals, az_dev_vals = self.cel_vals, self.az_dev_vals
        if az is not None: 
            mask = self.get_az_mask(az)
            if ~np.any(mask): 
                mask = self.get_az_mask(az, az_rng=180)

            if np.any(mask):
                cel_vals, az_dev_vals = cel_vals[mask], az_dev_vals[mask]
            else:
                print(f"No celerities within the az-mask for {az = }")

        cel_bins = np.arange(200, 360, 5)
        az_dev_bins = np.arange(-10, 10, 0.5)

        cel_kde = gaussian_kde(cel_vals)
        cel_kde_x = np.linspace(200, 360, 100)

        az_dev_kde_x = np.linspace(-10, 10, 50)
        az_dev_kde = gaussian_kde(az_dev_vals)

        if gmm:
            means_init = np.array([260, 315, 335]).reshape(-1, 1)
            gauss_mix = GMM(n_components=3, covariance_type="full", reg_covar=30, means_init = means_init).fit(cel_vals.reshape(-1, 1))
            # gmm_x = np.linspace(cel_vals.min(), cel_vals.max(), 200)
            gmm_x = cel_kde_x
            gmm_y = np.exp(gauss_mix.score_samples(gmm_x.reshape(-1, 1)))
        
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs = axs.flatten()
        if title is not None: 
            title = title + ", # vals = " + str(len(cel_vals))
            fig.suptitle(title)
        # date, lat, lon = atmos_file_handling.parse_g2s_fname(os.path.basename(atm_profile))
        # fig.suptitle(f"Distributions for ({lat}, {lon}) at {date.strftime('%Y-%m-%dT%H')}")

        axs[0].hist(cel_vals, bins=cel_bins, density=True, alpha=0.5)
        axs[0].plot(cel_kde_x, cel_kde(cel_kde_x), label="kde", c="r", linewidth=2)
        if gmm: 
            axs[0].plot(gmm_x, gmm_y, c="g", label="gmm", linewidth=2)
        axs[0].set_title("Celerity")
        axs[0].set_xlabel("Celerity [m/s]")
        axs[0].set_ylabel("Count")
        axs[0].legend()

        axs[1].hist(az_dev_vals, bins=az_dev_bins, density=True, alpha=0.5)
        axs[1].plot(az_dev_kde_x, az_dev_kde(az_dev_kde_x), label="kde", c="r", linewidth=2)
        axs[1].set_title("Azimuth deviation")
        axs[1].set_xlabel("Az. dev [Deg.]")
        axs[1].set_ylabel("Count")
        axs[1].legend()

        plt.show(block=block)


    def get_2d_dists(self, grid_dir, date, N_voronoi=3, filter_out_thermo=True, gw=False, block=True):
        # Calculate celerity and azimuth distributions based on the Tau-p rays over a grid 
        # The grid dir should be a directory with grids created by atmos/g2s_profile.py 
        # TODO: Add the possibility of filtering the values based on priors such as az, app cel, and turning height
        
        dir = atmos_file_handling.find_g2s_grid(date, grid_dir)
        summary_file = os.path.join(dir, "summary.dat")
        
        coords = np.loadtxt(summary_file, usecols=(0, 1))
        atm_profiles = np.loadtxt(summary_file, usecols=2, dtype=str)

          
        cel_vals = []
        az_dev_vals = []
        TauP = tau_p.TauP()
        start_time = time.perf_counter()
        for atm_profile in atm_profiles:
            atm_profile = os.path.join(dir, atm_profile)
            if "arrivals" in atm_profile:
                continue 
            elif "raypaths" in atm_profile:
                continue
            print(f"processing {os.path.basename(atm_profile)}")
            n_reals = 10 if gw else 1
            
            for i in range(n_reals):
                taup_results = TauP.evaluate_tau_p_integrals_profile(atm_profile, N_voronoi=N_voronoi, gw=gw)

                # mask = (taup_results["ducting_heights"] < 100) & (~np.isnan(taup_results["ducting_heights"]))
                mask = ~np.isnan(taup_results["ducting_heights"])
                # if filter_out_thermo: 
                #     mask = np.logical_and(mask, taup_results["ducting_heights"] < 100)
                # mask = np.logical_and(mask, taup_results["inc_degs"] > 80)
                mask = np.logical_and(mask, taup_results["ducting_heights"]<110)

                cel_vals += list(taup_results["celerities"][mask].flatten())
                az_dev_vals += list(taup_results["az_deviations"][mask].flatten())
        
        end_time = time.perf_counter()
        print(f"Finished calculating distributions after {end_time - start_time:.3f} s.")
        cel_vals = np.array(cel_vals)
        az_dev_vals = np.array(az_dev_vals)
        cel_bins = np.arange(200, 360, 5)
        az_dev_maxval = max(np.nanmax(az_dev_vals), np.abs(np.nanmin(az_dev_vals)))
        az_dev_bins = np.arange(-az_dev_maxval, az_dev_maxval, 0.5)

        cel_kde = gaussian_kde(cel_vals)
        cel_kde_x = np.linspace(200, 360, 100)

        az_dev_kde_x = np.linspace(-az_dev_maxval, az_dev_maxval, 50)
        az_dev_kde = gaussian_kde(az_dev_vals)
        
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs = axs.flatten()
        # date, lat, lon = atmos_file_handling.parse_g2s_fname(os.path.basename(atm_profile))
        # fig.suptitle(f"Distributions for ({lat}, {lon}) at {date.strftime('%Y-%m-%dT%H')}")

        axs[0].hist(cel_vals, bins=cel_bins, density=True)
        axs[0].plot(cel_kde_x, cel_kde(cel_kde_x), label="kde", c="r")
        axs[0].set_title("Celerity")
        axs[0].set_xlabel("Celerity [m/s]")
        axs[0].set_ylabel("Count")
        axs[0].legend()

        axs[1].hist(az_dev_vals, bins=az_dev_bins, density=True)
        axs[1].plot(az_dev_kde_x, az_dev_kde(az_dev_kde_x), label="kde", c="r")
        axs[1].set_title("Azimuth deviation")
        axs[1].set_xlabel("Az. dev [Deg.]")
        axs[1].set_ylabel("Count")
        axs[1].legend()

        plt.show(block=block)
   

if __name__ == "__main__":
    station = "MAAG"
    start_date = UTCDateTime("2022-04-01T00:00:00")
    end_date = start_date + timedelta(hours=8)
    latlons = {
        "MAAG": (50.7014, 29.2301),
        "GRDI": (50.5993, 29.4471),
    }
    # atm_profile = ( "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids/grid_2022-02-01T12/profiles/g2stxt_2022020112_50.0000_29.0000.dat")

    grid_dir = "/staff/sophus/Documents/ismonpy/test_scripts/data/ukraine_grids"
    grid_date = UTCDateTime("2022-02-01T12:00:00")

    # station = "MAAG"
    stations = ["MAAG", "GRDI"]   
    # stations = ["MAAG"]
    lat, lon = latlons[station]
    atmos_dir = f"/staff/sophus/Documents/ismonpy/test_scripts/data/{station.upper()}/"
    h_resol = 6

    CelDist = PropagationPriors(lat, lon)
    # CelDist.get_2d_dists(grid_dir, grid_date, filter_out_thermo=True, gw=False, N_voronoi=2)


    CelDist.calculate_distributions(atmos_dir, start_date, end_date, h_resol, N_voronoi=4, gw=True, filter=True)
    
    for az in range(-60, 180, 30):
        CelDist.plot_1d_dists(az=az, block=True, title=str(az))
        # CelDist.get_rcel_gmm(az=az, plot=True, slowness_domain=True)



