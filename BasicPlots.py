from cmath import tau
from itertools import chain
import os          
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2
import ast
import lmfit
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
import matplotlib.colors as mcolors


class AmBeNeutronAnalyzer:
    """
    Modular analyzer for AmBe neutron data with separate functions for different analysis tasks.
    """
    
    def __init__(self, data_directory: str = './EventAmBeNeutronCandidatesData/', 
                 output_pdf: str = 'AllAmBePositionsPlottest.pdf'):
        self.data_directory = data_directory
        self.output_pdf = output_pdf
        self.source_groups = {}
        self.time_fit_values = []
        self.pyMC_summary = []
        self.lmfit_summary = []
        
        # Port information mapping
        self.port_info = {
            (0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', 
            (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', (0, 55, 0): 'Port 5',
            (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', 
            (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1',
            (75, 100, 0): 'Port 4', (75, 50, 0): 'Port 4', (75, 0, 0): 'Port 4', 
            (75, -50, 0): 'Port 4', (75, -100, 0): 'Port 4',
            (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3', 
            (0, -50, 102): 'Port 3', (0, -100, 102): 'Port 3', (0, -105, 102): 'Port 3',
            (0, 100, 75): 'Port 2', (0, 50, 75): 'Port 2', (0, 0, 75): 'Port 2', 
            (0, -50, 75): 'Port 2', (0, -100, 75): 'Port 2'
        }
        self.port_order = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

    def load_and_group_data(self, file_pattern: str = 'EventAmBeNeutronCandidates_FullPECBWindow_4640.csv'):
        """Load CSV files and group them by source position."""
        files = self.data_directory
        csvs = glob.glob(os.path.join(files, file_pattern))

        
        for file in csvs:
            filename = os.path.basename(file)
            print(f"Processing file: {filename}")
            
            # Determine type (Event or Background)
            if filename.startswith('EventAmBeNeutronCandidates'):
                data_type = 'Event'
            elif filename.startswith('BackgroundAmBeNeutronCandidates'):
                data_type = 'Background'
            else:
                print(f"Skipping unrecognized file: {filename}")
                continue

            # Extract run number using regex
            match = re.search(r'_(\d+)\.csv', filename)
            if match:
                run_number = int(match.group(1))
            else:
                print(f"Could not extract run number from: {filename}")
                continue

            # Load CSV
            df = pd.read_csv(file)
            if df.empty:
                print(f"No data in file: {filename}")
                continue

            ###### Correct hitID swap as per johannes' suggestion ########
            id1, id2 = 453, 455
            if df['hitID'].astype(str).str.contains(f'{id1}|{id2}').any():
                original_hitids = df['hitID'].copy()
                df['hitID'] = (df['hitID'].astype(str)
                              .str.replace(f'{id1}', 'TEMP', regex=False)
                              .str.replace(f'{id2}', f'{id1}', regex=False)
                              .str.replace('TEMP', f'{id2}', regex=False))
                
                if (df['hitID'] != original_hitids).any():
                    print(f"Swapped hitID {id1} and {id2} in file: {filename}")

            x, y, z = df["sourceX"].iloc[0], df["sourceY"].iloc[0], df["sourceZ"].iloc[0]
            source_key = (round(x, 3), round(y, 3), round(z, 3))
            if source_key == (0, -105.5, 102):
                source_key = (0, -100, 102)
            if source_key not in self.source_groups:
                self.source_groups[source_key] = []
            self.source_groups[source_key].append(df)

    def prepare_data(self, combined_df: pd.DataFrame) -> Dict:
        """Prepare and process data for analysis."""
        # Convert cluster time to microseconds
        combined_df['clusterTime'] = combined_df['clusterTime'] / 1000
        
        # Extract relevant columns
        EventTime = combined_df['eventTankTime'].value_counts()
        PE = combined_df['clusterPE']
        CCB = combined_df['clusterChargeBalance']
        CT = combined_df['clusterTime']
        hit_delta_t = combined_df['hit_delta_t']
        CvX = combined_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[0]))
        CvY = combined_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[1]))
        CvZ = combined_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[2]))


        # Multi-cluster event analysis

        '''
        hit_delta_t represents the time spread of individual hits within a cluster,
        while delta_t_values represents the time difference between the first cluster and subsequent clusters in multi-cluster events.
        '''

        event_counts = combined_df.groupby('eventTankTime')['clusterTime'].transform('count')
        multi_cluster_df = combined_df[event_counts > 1].copy()
        multi_cluster_df['first_cluster_time'] = multi_cluster_df.groupby('eventTankTime')['clusterTime'].transform('min')
        multi_cluster_df['delta_t'] = multi_cluster_df['clusterTime'] - multi_cluster_df['first_cluster_time']
        delta_t_values = multi_cluster_df[multi_cluster_df['delta_t'] > 0]['delta_t']

        single_hit_delta_t = combined_df[event_counts == 1]['hit_delta_t']
        multi_hit_delta_t = combined_df[event_counts > 1]['hit_delta_t']
        filtered_EventTime = combined_df[combined_df['hit_delta_t'] > 20]['eventTankTime'].value_counts()


        return {
            'EventTime': EventTime,
            'PE': PE,
            'CCB': CCB,
            'CT': CT,
            'CvX': CvX,
            'CvY': CvY,
            'CvZ': CvZ,
            'delta_t_values': delta_t_values,
            'combined_df': combined_df,
            'hit_delta_t': hit_delta_t,
            'single_hit_delta_t': single_hit_delta_t,
            'multi_hit_delta_t': multi_hit_delta_t,
            'filtered_EventTime': filtered_EventTime
        }

    def plot_2d_histograms(self, data_dict: Dict, source_key: Tuple, pdf):
        """Generate all 2D histogram plots."""
        PE = data_dict['PE']
        CCB = data_dict['CCB']
        CT = data_dict['CT']
        CvX = data_dict['CvX']
        CvY = data_dict['CvY']
        CvZ = data_dict['CvZ']

        sx, sy, sz = (int(v) for v in source_key)


        # Compute 2D histograms first to find global max for normalization
        Hxy, _, _ = np.histogram2d(CvX, CvY, bins=100, range=[[-1, 1], [-1, 1]])
        Hxz, _, _ = np.histogram2d(CvX, CvZ, bins=100, range=[[-1, 1], [-1, 1]])
        Hyz, _, _ = np.histogram2d(CvY, CvZ, bins=100, range=[[-1, 1], [-1, 1]])

        # Find maximum bin count across all three histograms
        vmax = max(Hxy.max(), Hxz.max(), Hyz.max())
       

        # Create a Normalize object to share across subplots
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        # Create figure with 3 subplots for CvX, CvY, CvZ
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot XY
        im0 = axes[0].hist2d(CvX, CvY, bins=100, cmap='viridis', range=[[-1, 1], [-1, 1]], cmin=1, norm=norm)
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_title('XY')
        fig.colorbar(im0[3], ax=axes[0], label='Counts')

        # Plot XZ
        im1 = axes[1].hist2d(CvX, CvZ, bins=100, cmap='viridis', range=[[-1, 1], [-1, 1]], cmin=1, norm=norm)
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Z'); axes[1].set_title('XZ')
        fig.colorbar(im1[3], ax=axes[1], label='Counts')

        # Plot YZ
        im2 = axes[2].hist2d(CvY, CvZ, bins=100, cmap='viridis', range=[[-1, 1], [-1, 1]], cmin=1, norm=norm)
        axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z'); axes[2].set_title('YZ')
        fig.colorbar(im2[3], ax=axes[2], label='Counts')

        plt.suptitle(f'Cluster vector distributions for AmBe 2.0v1 (PE < 80, CCB < 0.40), run positions:({sx}, {sy}, {sz})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


        # PE vs Charge Balance
        plt.figure(figsize=(10, 6))
        plt.hist2d(PE, CCB, bins=100, cmap='viridis', 
                range=[[-10, 120], [0.1, 0.5]], cmin=1)
        plt.colorbar(label='Counts')
        plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0v1 (PE < 80, CCB < 0.40), run positions:({sx}, {sy}, {sz})")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # PE vs Cluster Time
        plt.figure(figsize=(10, 6))
        plt.hist2d(CT, PE, bins=100, cmap='viridis', range=[[0.1, 70], [-10, 100]], cmin=1)
        plt.colorbar(label='Counts')
        plt.xlabel('Cluster Time (μs)')
        plt.ylabel('Cluster PE')
        plt.title(f'AmBe Neutron Capture Time vs PE (run positions:({sx}, {sy}, {sz}))')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Cluster Time vs Charge Balance
        plt.figure(figsize=(10, 6))
        plt.hist2d(CT, CCB, bins=100, cmap='viridis', 
                range=[[0, 70], [0.1, 0.5]], cmin=1)
        plt.colorbar(label='Counts')
        plt.title(f"Cluster Time vs Charge Balance for AmBe 2.0v1 (PE < 80, CCB < 0.40), run positions:({sx}, {sy}, {sz})")
        plt.xlabel("Cluster Time (μs)")
        plt.ylabel("Cluster Charge Balance")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    def plot_1d_histograms(self, data_dict: Dict, source_key: Tuple, pdf):
        """Generate 1D histogram plots."""
        EventTime = data_dict['EventTime']
        filtered_EventTime = data_dict['filtered_EventTime']
        PE = data_dict['PE']
        delta_t_values = data_dict['delta_t_values']
        hit_delta_t = data_dict['hit_delta_t']
        single_hit_delta_t = data_dict['single_hit_delta_t']
        multi_hit_delta_t = data_dict['multi_hit_delta_t']
        sx, sy, sz = (int(v) for v in source_key)

        plt.figure(figsize=(10, 6))
        plt.hist(delta_t_values, bins=50, color='coral', edgecolor='black')
        plt.title(f'Time Difference (Δt) Between First and Subsequent Clusters for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (μs)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Neutron multiplicity
        plt.figure(figsize=(10, 6))
        plt.hist(EventTime, bins=range(1, 10, 1), log=True, edgecolor='blue', 
                color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0v1 (PE < 80, CCB < 0.40), run positions:({sx}, {sy}, {sz})')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(filtered_EventTime, bins=range(1, 10, 1), log=True, edgecolor='blue', 
                color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0v1 (PE < 80, CCB < 0.40), run positions:({sx}, {sy}, {sz})')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # PE spectrum
        plt.figure(figsize=(10, 6))
        plt.hist(PE, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
        plt.xlabel("Cluster PE")
        plt.ylabel("Counts")
        plt.title(f"PE Spectrum for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(hit_delta_t, bins=200, color='coral', edgecolor='black')
        plt.xlabel("hit Δt (ns)")
        plt.ylabel("Counts")
        plt.title(f"Δt Distribution for cluster collection for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(single_hit_delta_t, bins=50, color='coral', edgecolor='black')
        plt.title(f'SINGLE CLUSTER - Time Difference (Δt) Between First and Subsequent Clusters for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (ns)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(multi_hit_delta_t, bins=50, color='coral', edgecolor='black')
        plt.title(f'MULTI CLUSTER - Time Difference (Δt) Between First and Subsequent Clusters for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (ns)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    @staticmethod
    def NeutCapture(t, A, therm, tau, B):
        """Neutron capture time model."""
        return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B

    def scipy_curve_fit(self, data_dict: Dict, source_key: Tuple, pdf):
        """Perform curve fitting using scipy.optimize.curve_fit."""
        CT = data_dict['CT']
        EventTime = data_dict['EventTime']
        sx, sy, sz = (int(v) for v in source_key)

        # Prepare data for fitting
        counts, bin_edges = np.histogram(CT, bins=70, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fit_mask = (bin_centers > 2) & (bin_centers < 67)
        xdata = bin_centers[fit_mask]
        ydata = counts[fit_mask]
        ydata_errors = np.sqrt(ydata)
        ydata_errors[ydata_errors == 0] = 1e-10

        # Initial parameters
        init = [np.max(counts), 5, 25, np.min(counts)]

        # Perform fit
        try:
            popt, pcov = curve_fit(self.NeutCapture, xdata, ydata, p0=init, 
                                 sigma=ydata_errors, absolute_sigma=True,
                                 bounds=([0, 0.1, 10, 0], [np.inf, 10, 70, 15]))
            perr = np.sqrt(np.diag(pcov))
            ydata_expected = self.NeutCapture(xdata, *popt)
            
            # Calculate chi-square statistics
            chi2_value = np.sum(((ydata - ydata_expected) ** 2) / (ydata_errors ** 2))
            ndof = len(ydata) - len(popt)
            chi2_ndof = chi2_value / ndof
            p_value = chi2.sf(chi2_value, ndof)

            # Store results
            self.time_fit_values.append((f"{popt[1]:.2f}", f"{perr[1]:.2f}", 
                                       f"{popt[2]:.2f}", f"{perr[2]:.2f}", 
                                       (sx, sy, sz), len(EventTime), 
                                       f"{chi2_ndof:.2f}", f"{p_value:.3f}", f"{popt[3]:.2f}"))

            print(f"Scipy fit results for position {source_key}:")
            print(f"Thermal time:{popt[1]:.2f}±{perr[1]:.2f}, Capture time:{popt[2]:.2f}±{perr[2]:.2f}")

            # Plot results
            plt.figure(figsize=(10, 6))
            plt.hist(CT, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
            plt.errorbar(xdata, ydata, yerr=ydata_errors, color='blue', linestyle='None', alpha=0.7)

            label = (
                fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
                fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
                fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof}$, " + "\n"
                fr"$\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$"
            )

            plt.plot(xdata, self.NeutCapture(xdata, *popt), 'r-', linewidth=2, label=label)
            plt.xlabel(fr"Cluster Time [$\mu s$]")
            plt.ylabel("Counts")
            plt.legend()
            plt.title(f"Neutron Capture Time (scipy) for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Plot residuals
            residuals = (ydata - ydata_expected) / ydata_errors
            plt.figure(figsize=(10, 6))
            plt.plot(xdata, residuals, 'o-')
            plt.axhline(0, color='gray', linestyle='--')
            plt.xlabel("Time [μs]")
            plt.ylabel("Normalized Residual")
            plt.title(f"Fit Residuals (scipy) for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Scipy curve fit failed for position {source_key}: {e}")

    def lmfit_analysis(self, data_dict: Dict, source_key: Tuple, pdf):
        """Perform curve fitting using lmfit."""
        CT = data_dict['CT']
        sx, sy, sz = (int(v) for v in source_key)

        # Prepare data for fitting
        counts, bin_edges = np.histogram(CT, bins=70, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fit_mask = (bin_centers > 2) & (bin_centers < 67)
        xdata = bin_centers[fit_mask]
        ydata = counts[fit_mask]
        ydata_errors = np.sqrt(ydata)
        ydata_errors[ydata_errors == 0] = 1e-10

        try:
            model = lmfit.Model(self.NeutCapture)
            params = model.make_params(A=200, therm=5, tau=25, B=0)
            params['A'].min = 0
            params['therm'].min = 1e-9
            params['tau'].min = 1e-9
            params['B'].vary = False
            
            result = model.fit(ydata, params, t=xdata, weights=1/ydata_errors, method="basinhopping")
            print(f"LMFIT results for position {source_key}:")
            print(result.fit_report())
            
            best_fit_curve = result.best_fit
            
            lm_run_summary = {
                "Thermal": result.params['therm'].value,
                "Thermal_err": result.params['therm'].stderr,
                "Tau": result.params['tau'].value,
                "Tau_err": result.params['tau'].stderr,
                "Coordination": (sx, sy, sz),
                "chi2": result.chisqr,
                "reduced_chi2": result.redchi,
                "p_value": chi2.sf(result.chisqr, result.nfree)    
            }
            self.lmfit_summary.append(lm_run_summary)

            # Plot results
            plt.figure(figsize=(10, 6))
            plt.hist(CT, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
            plt.errorbar(xdata, ydata, yerr=ydata_errors, color='blue', linestyle='None', alpha=0.7)
            plt.plot(xdata, best_fit_curve, 'g-', linewidth=2, 
                    label=f'LMFIT: τ={result.params["tau"].value:.2f}±{result.params["tau"].stderr:.2f}μs')
            plt.xlabel("Cluster Time [μs]")
            plt.ylabel("Counts")
            plt.legend()
            plt.title(f"Neutron Capture Time (LMFIT) for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"LMFIT analysis failed for position {source_key}: {e}")

    def pymc_analysis(self, data_dict: Dict, source_key: Tuple, pdf):
        """Perform Bayesian analysis using PyMC."""
        CT = data_dict['CT']
        sx, sy, sz = (int(v) for v in source_key)

        # Prepare data for fitting
        counts, bin_edges = np.histogram(CT, bins=70, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fit_mask = (bin_centers > 2) & (bin_centers < 67)
        xdata = bin_centers[fit_mask]
        ydata = counts[fit_mask]
        ydata_errors = np.sqrt(ydata)
        ydata_errors[ydata_errors == 0] = 1e-10

        try:
            print(f"PyMC model running for position {source_key}")
            
            def NeutCapture_safe(t, A, therm, tau):
                epsilon = 1e-9
                rise_term = (1 - np.exp(-t / (therm + epsilon)))
                decay_term = np.exp(-t / (tau + epsilon))
                return (A * rise_term * decay_term)
        
            with pm.Model() as neutron_model:
                A = pm.HalfNormal("A", sigma=500)
                therm = pm.Uniform("therm", lower=1.0, upper=15.0)
                tau = pm.Uniform("tau", lower=15.0, upper=50.0)

                mu = NeutCapture_safe(xdata, A, therm, tau)
                Y_obs = pm.Normal("Y_obs", mu=mu, sigma=ydata_errors, observed=ydata)
                starting_values = {"A": 200, "therm": 6.00, "tau": 27.00}

                with neutron_model:
                    idata = pm.sample(2000, tune=1000, initvals=starting_values, chains=4, cores=4)
                
                print("PyMC sampling completed successfully!")
                summary_df = az.summary(idata)
                var_names = ["therm", "tau"]

                # Plot trace
                az.plot_trace(idata, var_names=var_names)
                plt.tight_layout()
                pdf.savefig(bbox_inches='tight')
                plt.close()

                run_summary = {
                    "Therm": summary_df.loc['therm']['mean'],
                    "Thermerr": summary_df.loc['therm']['sd'],
                    "Tau": summary_df.loc['tau']['mean'],
                    "Tauerr": summary_df.loc['tau']['sd'],
                    "Coordination": (sx, sy, sz),
                    "ThermHDI": (summary_df.loc['therm']['hdi_3%'], summary_df.loc['therm']['hdi_97%']),
                    "TauHDI": (summary_df.loc['tau']['hdi_3%'], summary_df.loc['tau']['hdi_97%'])
                }
                self.pyMC_summary.append(run_summary)

        except Exception as e:
            print(f"PyMC analysis failed for position {source_key}: {e}")

    def generate_summary_plots(self):
        """Generate summary heatmaps and statistics."""
        if not self.time_fit_values:
            print("No fit results to summarize")
            return

        # Create summary DataFrame
        Info = pd.DataFrame(self.time_fit_values, columns=['ThermalTime','ThermalTimeErr', 'CaptureTime',
                                                          'CaptureTimeErr', 'SourcePosition', 'LenEvents', 
                                                          "Chi2Ndof", "p-value", "B"])
        Info['ThermalTime'] = Info['ThermalTime'].astype(float)
        Info['ThermalTimeErr'] = Info['ThermalTimeErr'].astype(float)
        Info['CaptureTime'] = Info['CaptureTime'].astype(float)
        Info['CaptureTimeErr'] = Info['CaptureTimeErr'].astype(float)
        print(Info)

        # Add port information
        Info["Port"] = Info["SourcePosition"].map(self.port_info)
        Info["Y"] = Info["SourcePosition"].apply(lambda pos: pos[1])

        # Create pivot tables
        pivot_capturetime = Info.pivot(index="Y", columns="Port", values="CaptureTime")
        pivot_capturetimeerr = Info.pivot(index="Y", columns="Port", values="CaptureTimeErr")
        pivot_capturetime = pivot_capturetime.reindex(columns=self.port_order)
        pivot_capturetimeerr = pivot_capturetimeerr.reindex(columns=self.port_order)

        pivot_thermal_time = Info.pivot(index="Y", columns="Port", values="ThermalTime")
        pivot_thermal_time_err = Info.pivot(index="Y", columns="Port", values="ThermalTimeErr")
        pivot_thermal_time = pivot_thermal_time.reindex(columns=self.port_order)
        pivot_thermal_time_err = pivot_thermal_time_err.reindex(columns=self.port_order)

        # Create labels with errors
        def make_label_se(eff, err):
            return f"{eff}${{\\pm{err}}}$"

        vectorized_label = np.vectorize(make_label_se)
        labels_SE_capture = vectorized_label(pivot_capturetime.values, pivot_capturetimeerr.values)
        labels_SE_thermal = vectorized_label(pivot_thermal_time.values, pivot_thermal_time_err.values)

        # Statistics normalization
        Info['LenEvents'] = Info['LenEvents'].astype(int)
        Info['LenEvents'] = Info['LenEvents'] * (100/Info['LenEvents'].max())
        pivot_lenEvents = Info.pivot(index="Y", columns="Port", values="LenEvents").round(2)
        pivot_lenEvents = pivot_lenEvents.reindex(columns=self.port_order)

        # Create output directory
        os.makedirs("OutputPlots", exist_ok=True)

        # Capture time heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_capturetime, annot=labels_SE_capture, fmt="", cmap="YlOrBr", 
                   cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, 
                   cbar_kws={"label": "Capture Time (μs)"})
        plt.title("Capture Time of AmBe 2.0v1")
        plt.xlabel("Port")
        plt.ylabel("Y Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("OutputPlots/CaptureTime_AmBeNeutrons_AmBe2.0v1.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Thermal time heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_thermal_time, annot=labels_SE_thermal, fmt="", cmap="YlOrBr", 
                   cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, 
                   cbar_kws={"label": "Thermal Time (μs)"})
        plt.title("Thermal Time of AmBe 2.0v1")
        plt.xlabel("Port")
        plt.ylabel("Y Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("OutputPlots/ThermalTime_AmBeNeutrons_AmBe2.0v1.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Statistics heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_lenEvents, annot=True, fmt="", cmap="YlOrBr", cbar=True, 
                   annot_kws={"size": 12}, linecolor='black', linewidths=0.2, 
                   cbar_kws={"label": "%"})
        plt.title("Normalized Statistics of all AmBe Neutron-like Events for AmBe 2.0v1")
        plt.xlabel("Port")
        plt.ylabel("Y Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("OutputPlots/Statistics_AmBeNeutronEvents_AmBe2.0v1.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Calculate weighted averages
        def weighted_average(df, value, error):
            weights = 1.0 / (df[error] ** 2)
            weighted_avg = ((df[value] * weights).sum()) / (weights.sum())
            weighted_err = np.sqrt(1 / weights.sum())
            return weighted_avg, weighted_err

        NeutCaptureTime, NeutCaptureTimeErr = weighted_average(Info, 'CaptureTime', 'CaptureTimeErr')
        print(f"Weighted Average Capture Time: {NeutCaptureTime:.2f} ± {NeutCaptureTimeErr:.2f} μs")

        NeutThermalTime, NeutThermalTimeErr = weighted_average(Info, 'ThermalTime', 'ThermalTimeErr')
        print(f"Weighted Average Thermal Time: {NeutThermalTime:.2f} ± {NeutThermalTimeErr:.2f} μs")

        # Print PyMC and LMFIT summaries if available
        if self.pyMC_summary:
            pyMC_df = pd.DataFrame(self.pyMC_summary)
            print("PyMC Summary:")
            print(pyMC_df)

        if self.lmfit_summary:
            lmfit_df = pd.DataFrame(self.lmfit_summary)
            print("LMFIT Summary:")
            print(lmfit_df)

        return Info

    def run_analysis(self, file_pattern: str = 'EventAmBeNeutronCandidates_FullPECBWindow_4640.csv',
                    tasks: List[str] = None):
        """
        Run the complete analysis with specified tasks.
        
        Available tasks:
        - '2d_histograms': Generate 2D histogram plots
        - '1d_histograms': Generate 1D histogram plots  
        - 'scipy_fit': Perform scipy curve fitting
        - 'lmfit_fit': Perform lmfit analysis
        - 'pymc_fit': Perform PyMC Bayesian analysis
        - 'summary': Generate summary plots and statistics
        """
        if tasks is None:
            tasks = ['2d_histograms', '1d_histograms', 'scipy_fit', 'summary']

        # Load and group data
        self.load_and_group_data(file_pattern)

        # Create output directory
        os.makedirs("OutputPlots", exist_ok=True)

        with PdfPages(self.output_pdf) as pdf:
            for source_key, df_list in self.source_groups.items():
                combined_df = pd.concat(df_list, ignore_index=True)
                data_dict = self.prepare_data(combined_df)

                print(f"Analyzing source position: ({source_key[0]}, {source_key[1]}, {source_key[2]})")

                # Run selected tasks
                if '2d_histograms' in tasks:
                    self.plot_2d_histograms(data_dict, source_key, pdf)
                    
                if '1d_histograms' in tasks:
                    self.plot_1d_histograms(data_dict, source_key, pdf)
                    
                if 'scipy_fit' in tasks:
                    self.scipy_curve_fit(data_dict, source_key, pdf)
                    
                if 'lmfit_fit' in tasks:
                    self.lmfit_analysis(data_dict, source_key, pdf)
                    
                if 'pymc_fit' in tasks:
                    self.pymc_analysis(data_dict, source_key, pdf)

        # Generate summary plots if requested
        if 'summary' in tasks and self.time_fit_values:
            summary_info = self.generate_summary_plots()
            return summary_info

        return None


def main():
    """Example usage of the AmBeNeutronAnalyzer class."""
    
    # Initialize analyzer
    analyzer = AmBeNeutronAnalyzer()
    
    # Example 1: Run only 2D histograms for quick testing
    print("=== Running only 2D histograms ===")
    analyzer.run_analysis(tasks=['2d_histograms', '1d_histograms'])

    # Example 2: Run full analysis with all tasks
    # print("=== Running full analysis ===")
    # analyzer.run_analysis(tasks=['2d_histograms', '1d_histograms', 'scipy_fit', 'lmfit_fit', 'pymc_fit', 'summary'])
    
    # Example 3: Run only curve fitting tasks
    # print("=== Running only curve fitting ===")
    # analyzer.run_analysis(tasks=['scipy_fit', 'lmfit_fit', 'summary'])


if __name__ == "__main__":
    main()