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
from lmfit.models import ExponentialGaussianModel, ConstantModel
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
import matplotlib.colors as mcolors
from scipy.special import erfc
from scipy.optimize import minimize


class AmBeNeutronAnalyzer:
    """
    Modular analyzer for AmBe neutron data with separate functions for different analysis tasks.
    """
    
    def __init__(self, data_directory: str = './EventAmBeNeutronCandidatesData/', 
                 output_pdf: str = 'CB0.4PE120.pdf'):
        self.data_directory = data_directory
        self.output_pdf = output_pdf
        self.source_groups = {}
        self.time_fit_values = []
        self.pyMC_summary = []
        self.lmfit_summary = []
        
        # ===== FITTING CONFIGURATION - SINGLE SOURCE OF TRUTH =====
        # Change parameters here and all fitting methods will use them automatically
        self.fitting_config = {
            'time_bins': 70,
            'time_range': (0, 70),
            'fit_min_time': 2.0,
            'fit_max_time': 67.0,
            'initial_amplitude': 200.0,
            'initial_thermal_time': 5.0,
            'initial_capture_time': 25.0,
            'initial_background': 0.0,
            'amplitude_bounds': (0, np.inf),
            'thermal_bounds': (0.1, 10.0),
            'capture_bounds': (10.0, 70.0),
            'background_bounds': (0, 15.0)
        }

        
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

    def update_fitting_config(self, **kwargs):
        """
        Update fitting configuration parameters easily.
        Example: analyzer.update_fitting_config(fit_min_time=1.5, initial_capture_time=30)
        """
        for key, value in kwargs.items():
            if key in self.fitting_config:
                self.fitting_config[key] = value
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: {key} is not a valid configuration parameter")
        print("Updated fitting configuration:", self.fitting_config)

    def load_and_group_data(self, file_pattern: str = 'EventAmBeNeutronCandidates_test_4499.csv'):
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
            print(df.head())
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
        event_counts = combined_df.groupby('eventTankTime')['clusterTime'].transform('count')
        PE = combined_df['clusterPE']
        CCB = combined_df['clusterChargeBalance']
        CT = combined_df['clusterTime']
        hit_delta_t = combined_df['hit_delta_t']
        CvX = combined_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[0]))
        CvY = combined_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[1]))
        CvZ = combined_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[2]))
        
        # Keep one entry per eventTankTime (single or multi-cluster)
        unique_events_df = combined_df.drop_duplicates(subset='eventTankTime').copy()

        # Parse neutronTofCorrection into arrays
        unique_events_df['neutronTofCorrection'] = unique_events_df['neutronTofCorrection'].apply(
            lambda x: np.array([float(val) for val in str(x).split()])
            if pd.notna(x) and str(x).strip() else np.array([])
        )
        valid_arrays = [arr for arr in unique_events_df['neutronTofCorrection'].values if len(arr) > 0]
        all_neutron_tof = np.concatenate(valid_arrays) if valid_arrays else np.array([])

        # ToF corrected delta_t for all hits in all events
        unique_events_df['allHitsDeltaT_TofCorrected'] = unique_events_df['allHitsDeltaT_TofCorrected'].apply(
            lambda x: np.array([float(val) for val in str(x).split()])
            if pd.notna(x) and str(x).strip() else np.array([])
        )
        valid_delta_t_arrays = [arr for arr in unique_events_df['allHitsDeltaT_TofCorrected'].values if len(arr) > 0]
        all_delta_t_tof_corrected = np.concatenate(valid_delta_t_arrays) if valid_delta_t_arrays else np.array([])

        # Parse hitPE values (simple one-liner like neutronTofCorrection)
        all_hits_pe = np.concatenate([np.array([float(v) for v in str(x).strip('[]').replace(',', ' ').split() if v.strip() and v != '...']) for x in combined_df['hitPE'] if not (pd.isna(x) or str(x).strip() in ('', '[]'))]) if any(not (pd.isna(x) or str(x).strip() in ('', '[]')) for x in combined_df['hitPE']) else np.array([])


        # Multi-cluster event analysis

        '''
        hit_delta_t represents the time spread of individual hits within a cluster,
        while delta_t_values represents the time difference between the first cluster and subsequent clusters in multi-cluster events.
        '''
        single_cluster_events = combined_df[event_counts == 1]['eventTankTime'].unique()
        multi_cluster_df = combined_df[event_counts > 1].copy()
        '''PE = multi_cluster_df['clusterPE']
        CCB = multi_cluster_df['clusterChargeBalance']
        CT = multi_cluster_df['clusterTime']
        hit_delta_t = multi_cluster_df['hit_delta_t']
        CvX = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[0]))
        CvY = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[1]))
        CvZ = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[2]))
        
        multi_cluster_df["neutronTofCorrection"] = multi_cluster_df["neutronTofCorrection"].apply(
            lambda x: np.array([float(val) for val in str(x).split()]) if pd.notna(x) and str(x).strip() else np.array([])
        )
        valid_arrays = [arr for arr in multi_cluster_df["neutronTofCorrection"].values if len(arr) > 0]
        all_neutron_tof = np.concatenate(valid_arrays) if valid_arrays else np.array([])'''
        

        multi_cluster_df['first_cluster_time'] = multi_cluster_df.groupby('eventTankTime')['clusterTime'].transform('min')
        multi_cluster_df['delta_t'] = multi_cluster_df['clusterTime'] - multi_cluster_df['first_cluster_time']

       # events_with_large_delta = multi_cluster_df[multi_cluster_df['delta_t'] > 10]['eventTankTime'].unique()
      #  valid_events = np.concatenate((single_cluster_events, events_with_large_delta))
       # EventTime = combined_df['eventTankTime'].value_counts()
       # FilteredEventTime = EventTime[EventTime.index.isin(valid_events)]
    

        delta_t_values = multi_cluster_df[multi_cluster_df['delta_t'] > 0]['delta_t']
  
        single_hit_delta_t = combined_df[event_counts == 1]['hit_delta_t']
        multi_hit_delta_t = combined_df[event_counts > 1]['hit_delta_t']
        hit_delta_t = combined_df['hit_delta_t']
        


        return {
            'EventTime': EventTime,
           # 'FilteredEventTime': FilteredEventTime,
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
            'Neutron_vertex_tof': all_neutron_tof,
            'all_delta_t_tof_corrected': all_delta_t_tof_corrected,
            'all_hits_pe': all_hits_pe
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

        plt.suptitle(f'Cluster vector distributions for AmBe 2.0v1 (PE < 120, CCB < 0.40), run positions:({sx}, {sy}, {sz})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        Hpeccb, _, _ = np.histogram2d(PE, CCB, bins=100, range=[[-10, 120], [0.1, 0.5]])
        Hctpe, _, _  = np.histogram2d(CT, PE,  bins=100, range=[[0.1, 70], [-10, 120]])
        Hctccb, _, _ = np.histogram2d(CT, CCB, bins=100, range=[[0, 70], [0.1, 0.5]])

        vmax2 = max(Hpeccb.max(), Hctpe.max(), Hctccb.max())
        norm2 = mcolors.Normalize(vmin=0, vmax=vmax2)

        # PE vs Charge Balance
        fig2, ax2 = plt.subplots(1,3, figsize=(18, 6))
        im3= ax2[0].hist2d(PE, CCB, bins=100, cmap='viridis', 
                range=[[-10, 120], [0.1, 0.5]], cmin=1, norm=norm2)
        fig2.colorbar(im3[3], ax=ax2[0], label='Counts')
        ax2[0].set_title(f"Cluster PE vs Charge Balance")
        ax2[0].set_xlabel("Cluster PE")
        ax2[0].set_ylabel("Cluster Charge Balance")


        # PE vs Cluster Time
        im4 = ax2[1].hist2d(CT, PE, bins=100, cmap='viridis', range=[[0.1, 70], [-10, 100]], cmin=1, norm=norm2)
        fig2.colorbar(im4[3], ax=ax2[1], label='Counts')
        ax2[1].set_title(f"Cluster Time vs PE")
        ax2[1].set_xlabel("Cluster Time (μs)")
        ax2[1].set_ylabel("Cluster PE")

        # Cluster Time vs Charge Balance
        im5 = ax2[2].hist2d(CT, CCB, bins=100, cmap='viridis', 
                      range=[[0, 70], [0.1, 0.5]], cmin=1, norm=norm2)
        fig2.colorbar(im5[3], ax=ax2[2], label='Counts')
        ax2[2].set_title(f"Cluster Time vs Charge Balance")
        ax2[2].set_xlabel("Cluster Time (μs)")
        ax2[2].set_ylabel("Cluster Charge Balance")

        plt.suptitle(f'Cluster PE, Charge Balance and Time distributions for AmBe 2.0v1 (PE < 120, CCB < 0.40), run positions:({sx}, {sy}, {sz})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig2, bbox_inches='tight')
        plt.show()
        plt.close(fig2)


    def plot_1d_histograms(self, data_dict: Dict, source_key: Tuple, pdf):
        """Generate 1D histogram plots."""
        EventTime = data_dict['EventTime']
        #FilteredEventTime = data_dict['FilteredEventTime']
        PE = data_dict['PE']
        delta_t_values = data_dict['delta_t_values']
        hit_delta_t = data_dict['hit_delta_t']
        single_hit_delta_t = data_dict['single_hit_delta_t']
        multi_hit_delta_t = data_dict['multi_hit_delta_t']
        Neutron_vertex_tof = data_dict['Neutron_vertex_tof']
        all_hits_pe = data_dict['all_hits_pe']

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
        plt.hist(EventTime, bins=range(1, 10, 1), edgecolor='blue', 
                color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0v1 for all CH < 21 ns, run positions:({sx}, {sy}, {sz})')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()



        # PE spectrum
        plt.figure(figsize=(10, 6))
        plt.hist(PE, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
        plt.xlabel("Cluster PE")
        plt.ylabel("Counts")
        plt.title(f"PE Spectrum for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(hit_delta_t, bins=200, color='coral', edgecolor='black')
        plt.xlabel("hit Δt (ns)")
        plt.ylabel("Counts")
        plt.title(f"Δt Distribution for cluster collection for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(single_hit_delta_t, bins=50, color='coral', edgecolor='black')
        plt.title(f'SINGLE CLUSTER - Time Difference (Δt) Between First and Subsequent Clusters for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (ns)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(multi_hit_delta_t, bins=50, color='coral', edgecolor='black')
        plt.title(f'MULTI CLUSTER - Time Difference (Δt) Between First and Subsequent Clusters for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (ns)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(Neutron_vertex_tof, bins=300, range=(-20, 50), color='coral', edgecolor='black')
        plt.xlabel("Neutron Vertex Distance from Source (ns)")
        plt.ylabel("Counts")
        plt.title(f"Multi - Neutron Vertex TOF for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot all hits PE values
        if len(all_hits_pe) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(all_hits_pe, bins=50, range=(0, 20), log=True, color='skyblue', edgecolor='black')
            plt.xlabel("Hit PE Values")
            plt.ylabel("Counts")
            plt.title(f"All Cluster Hits PE Distribution for AmBe 2.0v1, run positions:({sx}, {sy}, {sz})")
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.show()
            plt.close()
    
    def emg_lmfit(self, data_dict: Dict, source_key: Tuple, pdf):
        Neutron_TOF = data_dict['Neutron_vertex_tof']
        all_delta_t_tof_corrected = data_dict['all_delta_t_tof_corrected']
        mask = Neutron_TOF < 9 # Apply mask to filter out large TOF values for 1sigma, 2sigma, 3sigma calculation
        all_delta_t_tof_corrected = all_delta_t_tof_corrected[mask]

        sx, sy, sz = (int(v) for v in source_key)
        if len(Neutron_TOF) == 0:
            print(f"No Neutron TOF data available for position {source_key}. Skipping EMG fit.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_delta_t_tof_corrected, bins=300, range=(-20, 70), color='coral', edgecolor='black')
        plt.xlabel("All Hits Δt ToF Corrected (ns)")
        plt.ylabel("Counts")
        plt.title(f"All Hits Δt ToF Corrected for AmBe 2.0v1 for all ToF, run positions:({sx}, {sy}, {sz})")
        plt.tight_layout()
        plt.show()
        #pdf.savefig(bbox_inches='tight')
        plt.close()
        
        counts, bin_edges = np.histogram(Neutron_TOF, bins=300, range=(-20, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        valid_indices = np.where(counts > 0)[0]

        if len(valid_indices) == 0:
            print(f"No valid histogram data for position {source_key}. Skipping EMG fit.")
            return
        
        start_indices = valid_indices[0]
        end_indices = valid_indices[-1] - 2
        #fit_mask = (bin_centers >= -10) & (bin_centers <= 60)

        xdata = bin_centers[start_indices:end_indices]
        ydata = counts[start_indices:end_indices]
        ydata_errors = np.sqrt(ydata)
        ydata_errors[ydata_errors == 0] = 1e-10

        model = ExponentialGaussianModel(prefix='emg_', nan_policy='propagate') + ConstantModel(prefix='const_')
        center_guess = xdata[np.argmax(ydata)]
        params = model.make_params()
        params['emg_amplitude'].set(value=np.max(ydata), min=0)
        #params['emg_center'].set(value=center_guess, min=center_guess-10, max=center_guess+10)
        params['emg_center'].set(value=0, vary=False)
        params['emg_sigma'].set(value=15, min=1, max=30)
        params['emg_gamma'].set(value=0.5, min=0.01, max=10)
        params['const_c'].set(value=np.min(ydata[ydata > 0]) if np.any(ydata > 0) else 0.0, min=0, max=np.max(ydata))

        result = model.fit(ydata, params, x=xdata, weights=1/ydata_errors)
        print(f"\nEMG Fit successful for position {source_key[0]}, {source_key[1]}, {source_key[2]}")
        print(result.fit_report())

        
        fig, ax = plt.subplots(figsize=(12, 7))


        center_val = result.params['emg_center']
        sigma_val = result.params['emg_sigma']
        gamma_val = result.params['emg_gamma']
        bg_val = result.params['const_c']


        central_err = center_val.stderr if center_val.stderr is not None else 0.0
        sigma_err = sigma_val.stderr if sigma_val.stderr is not None else 0.0
        gamma_err = gamma_val.stderr if gamma_val.stderr is not None else 0.0
        bg_err = bg_val.stderr if bg_val.stderr is not None else 0.0

        fit_label = (
            rf"$\mu$ = {center_val.value:.3f} $\pm$ {central_err:.3f}" + "\n"
            rf"$\sigma$ = {sigma_val.value:.3f} $\pm$ {sigma_err:.3f}" + "\n"
            rf"$\gamma$ = {gamma_val.value:.3f} $\pm$ {gamma_err:.3f}" + "\n"
            rf"$Background$ = {bg_val.value:.2f} $\pm$ {bg_err:.2f}"
        )

        ax.axvline(center_val.value, color='purple', linestyle=':', linewidth=2, label=f'μ')
        ax.axvspan(center_val.value - sigma_val.value, center_val.value + sigma_val.value, alpha=0.2, color='purple', label=f'1σ range')
        ax.axvspan(center_val.value - 2*sigma_val.value, center_val.value + 2*sigma_val.value, alpha=0.4, color='purple', label=f'2σ range')
        ax.axvspan(center_val.value - 3*sigma_val.value, center_val.value + 3*sigma_val.value, alpha=0.6, color='purple', label=f'3σ range')

        #ax.errorbar(xdata, ydata, yerr=ydata_errors, fmt='o', label='Error', capsize=3, alpha=0.6, markersize=2)
        
        # Plot the best-fit line
        ax.bar(xdata, ydata, width=bin_width, label=fit_label, 
           color='coral', alpha=0.3, edgecolor='black')
        ax.plot(xdata, result.best_fit, 'r-', label='Best Fit', linewidth=2)

        # Plot the components of the fit (EMG and background)
        comps = result.eval_components(x=xdata)
        ax.plot(xdata, comps["emg_"], 'g--', label='EMG Component', linewidth=2)
        ax.plot(xdata, comps["const_"], 'k--', label='Background Component', linewidth=2)

        ax.set_title(f'LMFIT Exponential Gaussian Model (EMG) Fit for source position: ({sx}, {sy}, {sz})', fontsize=16)
        ax.set_xlabel('Time of Flight (ns)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.legend(fontsize=10)
        plt.show()
        plt.tight_layout()
        #pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)



    ############### Capture time fitting Methods #################

    def _prepare_fitting_data(self, CT):
        """
        Centralized data preparation for fitting to ensure consistency across all methods.
        This prevents the parameter inconsistency issue you identified.
        """
        # Use configuration parameters - single source of truth
        counts, bin_edges = np.histogram(CT, bins=self.fitting_config['time_bins'], 
                                       range=self.fitting_config['time_range'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Apply consistent fitting mask
        fit_mask = ((bin_centers > self.fitting_config['fit_min_time']) & 
                   (bin_centers < self.fitting_config['fit_max_time']))
        
        xdata = bin_centers[fit_mask]
        ydata = counts[fit_mask]
        ydata_errors = np.sqrt(ydata)
        ydata_errors[ydata_errors == 0] = 1e-10
        
        return xdata, ydata, ydata_errors, counts, bin_centers

    @staticmethod
    def NeutCapture(t, A, therm, tau, B):
        """Neutron capture time model."""
        return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B

    def scipy_curve_fit(self, data_dict: Dict, source_key: Tuple, pdf):
        """Perform curve fitting using scipy.optimize.curve_fit."""
        CT = data_dict['CT']
        EventTime = data_dict['EventTime']
        sx, sy, sz = (int(v) for v in source_key)

        # Use centralized data preparation - guarantees consistency
        xdata, ydata, ydata_errors, counts, bin_centers = self._prepare_fitting_data(CT)

        # Use configuration parameters instead of hardcoded values
        init = [self.fitting_config['initial_amplitude'], 
                self.fitting_config['initial_thermal_time'], 
                self.fitting_config['initial_capture_time'], 
                self.fitting_config['initial_background']]

        # Perform fit using configuration bounds
        try:
            bounds = ([self.fitting_config['amplitude_bounds'][0], 
                      self.fitting_config['thermal_bounds'][0],
                      self.fitting_config['capture_bounds'][0], 
                      self.fitting_config['background_bounds'][0]],
                     [self.fitting_config['amplitude_bounds'][1], 
                      self.fitting_config['thermal_bounds'][1],
                      self.fitting_config['capture_bounds'][1], 
                      self.fitting_config['background_bounds'][1]])
            
            popt, pcov = curve_fit(self.NeutCapture, xdata, ydata, p0=init, 
                                 sigma=ydata_errors, absolute_sigma=True,
                                 bounds=bounds)
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

            # Plot results using consistent parameters
            plt.figure(figsize=(10, 6))
            plt.hist(CT, bins=self.fitting_config['time_bins'], 
                    range=self.fitting_config['time_range'], 
                    histtype='step', color='blue', label="Data")
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

        # Use centralized data preparation - guarantees same setup as scipy
        xdata, ydata, ydata_errors, counts, bin_centers = self._prepare_fitting_data(CT)

        try:
            model = lmfit.Model(self.NeutCapture)
            # Use configuration parameters - same as scipy method
            params = model.make_params(
                A=self.fitting_config['initial_amplitude'], 
                therm=self.fitting_config['initial_thermal_time'], 
                tau=self.fitting_config['initial_capture_time'], 
                B=self.fitting_config['initial_background']
            )
            # Apply consistent bounds
            params['A'].min = self.fitting_config['amplitude_bounds'][0]
            params['therm'].min = self.fitting_config['thermal_bounds'][0]
            params['therm'].max = self.fitting_config['thermal_bounds'][1]
            params['tau'].min = self.fitting_config['capture_bounds'][0]
            params['tau'].max = self.fitting_config['capture_bounds'][1]
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

            # Plot results using consistent parameters
            plt.figure(figsize=(10, 6))
            plt.hist(CT, bins=self.fitting_config['time_bins'], 
                    range=self.fitting_config['time_range'], 
                    histtype='step', color='blue', label="Data")
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

        # Use centralized data preparation - same as other methods
        xdata, ydata, ydata_errors, counts, bin_centers = self._prepare_fitting_data(CT)

        try:
            print(f"PyMC model running for position {source_key}")
            
            def NeutCapture_safe(t, A, therm, tau):
                epsilon = 1e-9
                rise_term = (1 - np.exp(-t / (therm + epsilon)))
                decay_term = np.exp(-t / (tau + epsilon))
                return (A * rise_term * decay_term)
        
            with pm.Model() as neutron_model:
                A = pm.HalfNormal("A", sigma=500)
                # Use consistent bounds from configuration
                therm = pm.Uniform("therm", 
                                  lower=self.fitting_config['thermal_bounds'][0], 
                                  upper=self.fitting_config['thermal_bounds'][1])
                tau = pm.Uniform("tau", 
                                lower=self.fitting_config['capture_bounds'][0], 
                                upper=self.fitting_config['capture_bounds'][1])

                mu = NeutCapture_safe(xdata, A, therm, tau)
                Y_obs = pm.Normal("Y_obs", mu=mu, sigma=ydata_errors, observed=ydata)
                # Use consistent starting values from configuration
                starting_values = {
                    "A": self.fitting_config['initial_amplitude'], 
                    "therm": self.fitting_config['initial_thermal_time'], 
                    "tau": self.fitting_config['initial_capture_time']
                }

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

    def run_analysis(self, file_pattern: str = 'EventAmBeNeutronCandidates_test_4499.csv',
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

                if 'emg_lmfit' in tasks:
                    self.emg_lmfit(data_dict, source_key, pdf)



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
    analyzer.run_analysis(tasks=['2d_histograms', '1d_histograms', 'scipy_fit', 'lmfit_fit', 'pymc_fit', 'summary'])

    # Example 2: Run full analysis with all tasks
    # print("=== Running full analysis ===")
    # analyzer.run_analysis(tasks=['2d_histograms', '1d_histograms', 'scipy_fit', 'lmfit_fit', 'pymc_fit', 'summary'])
    
    # Example 3: Run only curve fitting tasks
    # print("=== Running only curve fitting ===")
    # analyzer.run_analysis(tasks=['scipy_fit', 'lmfit_fit', 'summary'])


if __name__ == "__main__":
    main()