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
from typing import Dict, List, Tuple, Optional
import matplotlib.colors as mcolors
from scipy.special import erfc
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D



class AmBeNeutronAnalyzer:
    """
    Modular analyzer for AmBe neutron data with separate functions for different analysis tasks.
    """
    
    def __init__(self, data_directory: str = './EventAmBeNeutronCandidatesData/',
                 output_pdf: str = 'AmBevtestv4ch10.pdf',
                 campaign_label: str = 'AmBe 2.0v4',
                 min_cluster_hits: int = 10,
                 selection_label: str = ''):
        self.data_directory = data_directory
        self.output_pdf = output_pdf
        # Plot-only hits cut applied in prepare_data. See the note there: it is not
        # part of any published selection, so it has to be stated, not assumed.
        self.min_cluster_hits = min_cluster_hits
        # How this book's clusters were selected, printed on every page. Empty
        # falls back to the box wording. Several titles used to hardcode both the
        # campaign ('AmBe 2.0v1' on one page, 'AmBe 2.0v4' on the next, for the same
        # run) and '(PE < 100 & CCB < 0.45)' -- which is simply false for the MVA
        # book, whose selection has no PE or charge-balance cut at all.
        self.selection_label = selection_label or (
            'box cuts PE ≤ 100, CB < 0.45, t ≥ 2 µs, hits ≥ 5')
        self.campaign_label = campaign_label
        self.campaign_tag = campaign_label.replace(' ', '')
        self.source_groups = {}
        self.time_fit_values = []
        self.pyMC_summary = []
        self.lmfit_summary = []
        
        # ===== FITTING CONFIGURATION - SINGLE SOURCE OF TRUTH =====
        # Change parameters here and all fitting methods will use them automatically
        self.fitting_config = {
            'time_bins': 70,
            'time_range': (0, 70),
            # WAS 10.0. The fit window is now 2-67 us campaign-wide -- see the note
            # on FIT_MIN in fit_capture_time_fprompt_compare.py, which this must stay
            # equal to. 2 us is where the data starts (the box cut is t >= 2 us and
            # the cosmic veto removes anything earlier), so fitting from 10 discarded
            # the whole thermalisation rise. Every published capture time moves
            # because of this; the tau = 30.53 +- 0.26 us anchor was a 10-67 number
            # and is superseded.
            'fit_min_time': 2.0,
            'fit_max_time': 67.0,
            # Where the fitted curve is DRAWN from. Now equal to fit_min_time, so the
            # drawn curve and the fitted range coincide and there is no extrapolated
            # segment to distinguish. Kept as its own key so the two can be separated
            # again without touching the fit.
            'plot_min_time': 2.0,
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
            (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', (0, 55.3, 0): 'Port 5',
            (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1',
            (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1',
            (75, 100, 0): 'Port 4', (75, 50, 0): 'Port 4', (75, 0, 0): 'Port 4',
            (75, -50, 0): 'Port 4', (75, -100, 0): 'Port 4',
            (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3',
            (0, -50, 102): 'Port 3', (0, -60, 102): 'Port 3', (0, -100, 102): 'Port 3', (0, -105.5, 102): 'Port 3',
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

    def load_and_group_data(self, file_pattern: str = 'EventAmBeNeutronCandidates_fullwindowtest_*.csv'):
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

        # PLOT-ONLY CUT, now explicit. This used to be a hardcoded
        # `clusterHits >= 10`, applied on top of whatever selection produced the
        # candidate CSVs. Two reasons it had to become a parameter:
        #   * It is NOT part of the published box (which requires hits >= 5), so the
        #     per-position pages did not show the selection they were labelled with.
        #   * Under the MVA neutron definition there is no hits requirement at all,
        #     so leaving it hardcoded would silently reimpose one and make the
        #     "no box" deck a hits-cut deck.
        # Default stays 10 so existing callers reproduce their old pages; the deck
        # configs set it explicitly, and identically, so the two decks stay
        # comparable. None / 0 disables it.
        if self.min_cluster_hits:
            n_before = len(combined_df)
            combined_df = combined_df[combined_df['clusterHits'] >= self.min_cluster_hits]
            print(f"  plot cut clusterHits >= {self.min_cluster_hits}: "
                  f"{n_before:,} -> {len(combined_df):,} clusters")

        EventID = combined_df['eventID'].value_counts()
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
        if 'neutronTofCorrection' not in unique_events_df.columns:
            unique_events_df['neutronTofCorrection'] = np.nan
        else:
            unique_events_df['neutronTofCorrection'] = unique_events_df['neutronTofCorrection'].apply(
                lambda x: np.array([float(val) for val in str(x).split()])
                if pd.notna(x) and str(x).strip() else np.array([])
            )
            valid_arrays = [arr for arr in unique_events_df['neutronTofCorrection'].values if len(arr) > 0]
            all_neutron_tof = np.concatenate(valid_arrays) if valid_arrays else np.array([])

        # ToF corrected delta_t for all hits in all events
        if 'allHitsDeltaT_TofCorrected' not in unique_events_df.columns:
            unique_events_df['allHitsDeltaT_TofCorrected'] = np.nan
        else:
            unique_events_df['allHitsDeltaT_TofCorrected'] = unique_events_df['allHitsDeltaT_TofCorrected'].apply(
                lambda x: np.array([float(val) for val in str(x).split()])
                if pd.notna(x) and str(x).strip() else np.array([])
            )
            valid_delta_t_arrays = [arr for arr in unique_events_df['allHitsDeltaT_TofCorrected'].values if len(arr) > 0]
            all_delta_t_tof_corrected = np.concatenate(valid_delta_t_arrays) if valid_delta_t_arrays else np.array([])

        # Parse hitPE values (simple one-liner like neutronTofCorrection
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
        CvZ = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[2]))'''

        multi_cluster_df['CvX'] = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[0]))
        multi_cluster_df['CvY'] = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[1]))
        multi_cluster_df['CvZ'] = multi_cluster_df['clusterDirection'].apply(lambda v: float(v.strip('[]').split()[2]))
        
        if 'neutronTofCorrection' not in multi_cluster_df.columns:
            multi_cluster_df['neutronTofCorrection'] = np.nan
        else:
            multi_cluster_df["neutronTofCorrection"] = multi_cluster_df["neutronTofCorrection"].apply(
                lambda x: np.array([float(val) for val in str(x).split()]) if pd.notna(x) and str(x).strip() else np.array([])
            )
            valid_arrays = [arr for arr in multi_cluster_df["neutronTofCorrection"].values if len(arr) > 0]
            all_neutron_tof = np.concatenate(valid_arrays) if valid_arrays else np.array([])

        multi_cluster_df['first_cluster_time'] = multi_cluster_df.groupby('eventTankTime')['clusterTime'].transform('min')
        multi_cluster_df['delta_t'] = multi_cluster_df['clusterTime'] - multi_cluster_df['first_cluster_time']
        multi_cluster_df['is_first_cluster'] = multi_cluster_df['clusterTime'] == multi_cluster_df['first_cluster_time']
        multi_cluster_df['cluster_type'] = np.where(multi_cluster_df['is_first_cluster'], 'first', 'subsequent')
        #subsequent_clusters = multi_cluster_df[(multi_cluster_df['delta_t'] < 1) & (multi_cluster_df['cluster_type'] == 'subsequent')]
        first_clusters = multi_cluster_df[(multi_cluster_df['delta_t'] < 1) &(multi_cluster_df['cluster_type'] == 'first')]
        valid_event_ids = first_clusters['eventTankTime'].unique()
        subsequent_clusters = multi_cluster_df[(multi_cluster_df['cluster_type'] == 'subsequent') & (multi_cluster_df['eventTankTime'].isin(valid_event_ids))]


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
            #'Neutron_vertex_tof': all_neutron_tof,
            #'all_delta_t_tof_corrected': all_delta_t_tof_corrected,
            'all_hits_pe': all_hits_pe,
            'first_clusters': first_clusters,
            'subsequent_clusters': subsequent_clusters,
            'EventID': EventID

        }

    def plot_2d_histograms(self, data_dict: Dict, source_key: Tuple, pdf):
        """Generate all 2D histogram plots."""
        PE = data_dict['PE']
        CCB = data_dict['CCB']
        CT = data_dict['CT']
        CvX = data_dict['CvX']
        CvY = data_dict['CvY']
        CvZ = data_dict['CvZ']
        first_clusters = data_dict['first_clusters']
        subsequent_clusters = data_dict['subsequent_clusters']


        sx, sy, sz = (int(v) for v in source_key)

 

        # Plot first clusters in red
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns


        axes[0].scatter(first_clusters['CvX'], first_clusters['CvY'],
                        c='red', label='First cluster', alpha=0.7, s=50)
        axes[0].scatter(subsequent_clusters['CvX'], subsequent_clusters['CvY'],
                        c='blue', label='Subsequent clusters', alpha=0.5, s=30)
        axes[0].set_xlabel('CvX')
        axes[0].set_ylabel('CvY')
        axes[0].set_title('Δt < 1 μs')
        axes[0].legend()


        axes[1].scatter(first_clusters['CvX'], first_clusters['CvZ'],
                        c='red', label='First cluster', alpha=0.7, s=50)
        axes[1].scatter(subsequent_clusters['CvX'], subsequent_clusters['CvZ'],
                        c='blue', label='Subsequent clusters', alpha=0.5, s=30)
        axes[1].set_xlabel('CvX')
        axes[1].set_ylabel('CvZ')
        axes[1].set_title('Δt < 1 μs')
        axes[1].legend()


        axes[2].scatter(first_clusters['CvY'], first_clusters['CvZ'],
                        c='red', label='First cluster', alpha=0.7, s=50)
        axes[2].scatter(subsequent_clusters['CvY'], subsequent_clusters['CvZ'],
                        c='blue', label='Subsequent clusters', alpha=0.5, s=30)
        axes[2].set_xlabel('CvY')
        axes[2].set_ylabel('CvZ')
        axes[2].set_title('Δt < 1 μs')
        axes[2].legend()

        plt.tight_layout()  # adjust spacing
        pdf.savefig(fig, bbox_inches='tight')
        #plt.show()


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

        plt.suptitle(f'{self.selection_label}\nCluster vector distributions, {self.campaign_label}, source position ({sx}, {sy}, {sz})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        #plt.show()
        plt.close(fig)

        Hpeccb, _, _ = np.histogram2d(PE, CCB, bins=100, range=[[-10, 120], [0.1, 0.5]])
        Hctpe, _, _  = np.histogram2d(CT, PE,  bins=100, range=[[0.1, 70], [-10, 120]])
        Hctccb, _, _ = np.histogram2d(CT, CCB, bins=100, range=[[0, 70], [0.1, 0.5]])

        vmax2 = max(Hpeccb.max(), Hctpe.max(), Hctccb.max())
        norm2 = mcolors.Normalize(vmin=0, vmax=vmax2)

        # PE vs Charge Balance
        fig2, ax2 = plt.subplots(1,3, figsize=(18, 6))
        im3= ax2[0].hist2d(PE, CCB, bins=100, cmap='viridis', 
                range=[[-10, 150], [0.1, 1]], cmin=1, norm=norm2)
        fig2.colorbar(im3[3], ax=ax2[0], label='Counts')
        ax2[0].set_title(f"Cluster PE vs Charge Balance")
        ax2[0].set_xlabel("Cluster PE")
        ax2[0].set_ylabel("Cluster Charge Balance")


        # PE vs Cluster Time
        im4 = ax2[1].hist2d(CT, PE, bins=100, cmap='viridis', range=[[0.1, 70], [-10, 150]], cmin=1, norm=norm2)
        fig2.colorbar(im4[3], ax=ax2[1], label='Counts')
        ax2[1].set_title(f"Cluster Time vs PE")
        ax2[1].set_xlabel("Cluster Time (μs)")
        ax2[1].set_ylabel("Cluster PE")

        # Cluster Time vs Charge Balance
        im5 = ax2[2].hist2d(CT, CCB, bins=100, cmap='viridis', 
                      range=[[0, 70], [0.1, 1]], cmin=1, norm=norm2)
        fig2.colorbar(im5[3], ax=ax2[2], label='Counts')
        ax2[2].set_title(f"Cluster Time vs Charge Balance")
        ax2[2].set_xlabel("Cluster Time (μs)")
        ax2[2].set_ylabel("Cluster Charge Balance")

        plt.suptitle(f'{self.selection_label}\nSingle-cluster PE, charge balance and time, {self.campaign_label}, source position ({sx}, {sy}, {sz})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig2, bbox_inches='tight')
        #plt.show()
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
       # Neutron_vertex_tof = data_dict['Neutron_vertex_tof']
        all_hits_pe = data_dict['all_hits_pe']
        EventID = data_dict['EventID']

        sx, sy, sz = (int(v) for v in source_key)

        plt.figure(figsize=(10, 6))
        plt.hist(delta_t_values, bins=70, color='coral', edgecolor='black')
        plt.title(f'Time Difference (Δt) Between First and Subsequent Clusters for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (μs)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()

        # Neutron multiplicity
        plt.figure(figsize=(10, 6))
        plt.hist(EventTime, bins=range(1, 10, 1), edgecolor='blue', 
                color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'{self.selection_label}\nNeutron multiplicity, {self.campaign_label}, source position ({sx}, {sy}, {sz})')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(EventID, bins=range(1, 10, 1), edgecolor='blue', 
                color="lightblue", linewidth=0.5, align='left', density=False, log=True)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'{self.selection_label}\nNeutron multiplicity per trigger, {self.campaign_label}, source position ({sx}, {sy}, {sz})')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()



        # PE spectrum
        plt.figure(figsize=(10, 6))
        plt.hist(PE, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
        plt.xlabel("Cluster PE")
        plt.ylabel("Counts")
        plt.title(f"{self.selection_label}\nCluster PE spectrum, {self.campaign_label}, source position ({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(hit_delta_t, bins=200, color='coral', edgecolor='black')
        plt.xlabel("hit Δt (ns)")
        plt.ylabel("Counts")
        plt.title(f"{self.selection_label}\nΔt for cluster collection, {self.campaign_label}, source position ({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(single_hit_delta_t, bins=50, color='coral', edgecolor='black')
        plt.title(f'SINGLE CLUSTER - hit collection window for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (ns)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(multi_hit_delta_t, bins=50, color='coral', edgecolor='black')
        plt.title(f'MULTI CLUSTER - hit collection window for ({sx}, {sy}, {sz})', fontsize=16)
        plt.xlabel('Δt (ns)', fontsize=12)
        plt.ylabel('Number of Subsequent Clusters', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()

        '''plt.figure(figsize=(10, 6))
        plt.hist(Neutron_vertex_tof, bins=300, range=(-20, 50), color='coral', edgecolor='black')
        plt.xlabel("Neutron Vertex Distance from Source (ns)")
        plt.ylabel("Counts")
        plt.title(f"{self.selection_label}\nMulti-neutron vertex ToF, {self.campaign_label}, source position ({sx}, {sy}, {sz})")
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        plt.close()'''

        # Plot all hits PE values
        if len(all_hits_pe) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(all_hits_pe, bins=50, range=(0, 20), log=True, color='skyblue', edgecolor='black')
            plt.xlabel("Hit PE Values")
            plt.ylabel("Counts")
            plt.title(f"{self.selection_label}\nAll cluster-hit PE, {self.campaign_label}, source position ({sx}, {sy}, {sz})")
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            #plt.show()
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
        plt.title(f"{self.selection_label}\nAll hits Δt, ToF corrected, {self.campaign_label}, source position ({sx}, {sy}, {sz})")
        plt.tight_layout()
        #plt.show()
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
        #plt.show()
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
            plt.title(f"Neutron Capture Time (scipy) for {self.campaign_label}, run positions:({sx}, {sy}, {sz})")
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
            plt.title(f"Fit Residuals (scipy) for {self.campaign_label}, run positions:({sx}, {sy}, {sz})")
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
            
            # method="least_squares", NOT "basinhopping" and NOT lmfit's default
            # "leastsq". Two separate reasons, both measured on this campaign:
            #
            #   leastsq (lmfit default) imposes min/max by transforming the
            #   parameters internally, and on the high-statistics positions that
            #   transform settles in a worse local minimum and reports no error bars
            #   at all. At (0,-100,-75) it returns therm = 0.33 -- effectively pinned
            #   at the 0.1 bound -- and tau = 33.21 with stderr None; 2 of 26
            #   positions failed outright that way.
            #
            #   basinhopping is a global-minimum search wrapped around a local one.
            #   It is far slower per position and its covariance is not on the same
            #   footing as curve_fit's, so the per-page numbers could not be compared
            #   with CaptureTimeFits_*.csv or with the 30.53 +- 0.26 anchor.
            #
            # least_squares IS scipy.optimize.least_squares -- the bounded TRF solver
            # curve_fit uses -- so these pages reproduce the frozen fits to every
            # digit they print. scale_covar=False keeps the stderr absolute
            # (curve_fit's absolute_sigma=True) rather than rescaled by sqrt(redchi).
            result = model.fit(ydata, params, t=xdata, weights=1 / ydata_errors,
                               method="least_squares", scale_covar=False)
            print(f"LMFIT results for position {source_key}:")
            print(result.fit_report())
            
            # Evaluate the fitted model on a dense grid over the DISPLAY range
            # (plot_min_time -> fit_max_time), not on xdata. result.best_fit is the
            # model at the fit points only, i.e. 10-67 us, which is why the curve
            # used to start at 10. Same parameters either way -- this extends the
            # drawn line across the thermalisation rise the fit already describes.
            xplot = np.linspace(self.fitting_config['plot_min_time'],
                                self.fitting_config['fit_max_time'], 500)
            best_fit_curve = self.NeutCapture(
                xplot,
                result.params['A'].value, result.params['therm'].value,
                result.params['tau'].value, result.params['B'].value)

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
            label = (
                fr"$\mathrm{{therm}} = {result.params['therm'].value:.2f} \pm {result.params['therm'].stderr:.2f}\ \mu s$" + "\n"
                fr"$\tau = {result.params['tau'].value:.2f} \pm {result.params['tau'].stderr:.2f}\ \mu s$" + "\n"
                fr"$\chi^2 = {result.chisqr:.2f},\ \mathrm{{ndof}} = {result.nfree}$, " + "\n"
                fr"$\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {result.redchi:.2f}$" + "\n"
                # The window is stated on every page. It is not decoration: these
                # numbers are NOT comparable with anything fitted on the old 10-67
                # window, and a page that does not say which window it used cannot be
                # told apart from one that did. Built from the config rather than
                # written out, so it cannot drift from the fit that produced it.
                + (fr"fit {self.fitting_config['fit_min_time']:.0f}–"
                   fr"{self.fitting_config['fit_max_time']:.0f} $\mu s$"
                   if self.fitting_config['plot_min_time']
                   >= self.fitting_config['fit_min_time'] else
                   fr"fit {self.fitting_config['fit_min_time']:.0f}–"
                   fr"{self.fitting_config['fit_max_time']:.0f} $\mu s$, "
                   fr"drawn {self.fitting_config['plot_min_time']:.0f}–"
                   fr"{self.fitting_config['fit_max_time']:.0f} $\mu s$")
            )
            plt.plot(xplot, best_fit_curve, 'g-', linewidth=2,
                    label=label)
            plt.xlabel("Cluster Time [μs]")
            plt.ylabel("Counts")
            plt.legend()
            plt.title(f"Neutron Capture Time (LMFIT) for {self.campaign_label}, run positions:({sx}, {sy}, {sz})")
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"LMFIT analysis failed for position {source_key}: {e}")

    def pymc_analysis(self, data_dict: Dict, source_key: Tuple, pdf):
        """Perform Bayesian analysis using PyMC."""
        import pymc as pm
        import arviz as az
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
        """
        Generate summary heatmaps and statistics using lmfit results.
        
        This method creates heatmaps for capture time and thermal time based on 
        lmfit fitting results rather than scipy curve_fit results. The plots show
        thermal time and capture time with their uncertainties across different
        source positions and ports.
        
        Requires that lmfit_analysis() has been run first to populate self.lmfit_summary.
        """
        if not self.lmfit_summary:
            print("No lmfit results to summarize")
            return

        # Create summary DataFrame from lmfit results instead of scipy
        lmfit_data = []
        for result in self.lmfit_summary:
            lmfit_data.append([
                round(result['Thermal'], 2),
                round(result['Thermal_err'] if result['Thermal_err'] is not None else 0.0, 2),
                round(result['Tau'], 2),
                round(result['Tau_err'] if result['Tau_err'] is not None else 0.0, 2),
                result['Coordination'],
                0,  # LenEvents - we'll need to get this separately
                round(result['reduced_chi2'], 2),
                round(result['p_value'], 3),
                0.0  # Background - not used in lmfit version
            ])
        
        Info = pd.DataFrame(lmfit_data, columns=['ThermalTime','ThermalTimeErr', 'CaptureTime',
                                               'CaptureTimeErr', 'SourcePosition', 'LenEvents', 
                                               "Chi2Ndof", "p-value", "B"])
        Info['ThermalTime'] = Info['ThermalTime'].astype(float)
        Info['ThermalTimeErr'] = Info['ThermalTimeErr'].astype(float)
        Info['CaptureTime'] = Info['CaptureTime'].astype(float)
        Info['CaptureTimeErr'] = Info['CaptureTimeErr'].astype(float)
        
        # Get event counts from source_groups
        for i, (source_key, df_list) in enumerate(self.source_groups.items()):
            if i < len(Info):
                try:
                    combined_df = pd.concat(df_list, ignore_index=True)
                    data_dict = self.prepare_data(combined_df)
                    Info.loc[i, 'LenEvents'] = len(data_dict['EventTime'])
                except:
                    # Fallback: just count total events directly
                    combined_df = pd.concat(df_list, ignore_index=True)
                    Info.loc[i, 'LenEvents'] = len(combined_df['eventTankTime'].unique()) if 'eventTankTime' in combined_df.columns else len(combined_df)
        
        print("LMFIT Summary")
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
            return f"{eff:.2f}$\\pm{err:.2f}$"

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
        plt.title(f"Capture Time of {self.campaign_label} (LMFIT)")
        plt.xlabel("Port")
        plt.ylabel("Y Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"OutputPlots/CaptureTime_AmBeNeutrons_{self.campaign_tag}_LMFIT.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Thermal time heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_thermal_time, annot=labels_SE_thermal, fmt="", cmap="YlOrBr", 
                   cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, 
                   cbar_kws={"label": "Thermal Time (μs)"})
        plt.title(f"Thermal Time of {self.campaign_label} (LMFIT)")
        plt.xlabel("Port")
        plt.ylabel("Y Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"OutputPlots/ThermalTime_AmBeNeutrons_{self.campaign_tag}_LMFIT.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Statistics heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_lenEvents, annot=True, fmt="", cmap="YlOrBr", cbar=True, 
                   annot_kws={"size": 12}, linecolor='black', linewidths=0.2, 
                   cbar_kws={"label": "%"})
        plt.title(f"Normalized Statistics of all AmBe Neutron-like Events for {self.campaign_label}")
        plt.xlabel("Port")
        plt.ylabel("Y Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"OutputPlots/Statistics_AmBeNeutronEvents_{self.campaign_tag}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Calculate weighted averages
        def weighted_average(df, value, error):
            # Handle potential zero or NaN errors
            valid_mask = (df[error] > 0) & (~df[error].isna())
            valid_df = df[valid_mask]
            if len(valid_df) == 0:
                return df[value].mean(), df[value].std()
            
            weights = 1.0 / (valid_df[error] ** 2)
            weighted_avg = ((valid_df[value] * weights).sum()) / (weights.sum())
            weighted_err = np.sqrt(1 / weights.sum())
            return weighted_avg, weighted_err

        NeutCaptureTime, NeutCaptureTimeErr = weighted_average(Info, 'CaptureTime', 'CaptureTimeErr')
        print(f"Weighted Average Capture Time (LMFIT): {NeutCaptureTime:.2f} ± {NeutCaptureTimeErr:.2f} μs")

        NeutThermalTime, NeutThermalTimeErr = weighted_average(Info, 'ThermalTime', 'ThermalTimeErr')
        print(f"Weighted Average Thermal Time (LMFIT): {NeutThermalTime:.2f} ± {NeutThermalTimeErr:.2f} μs")

        # Print PyMC and LMFIT summaries if available
        if self.pyMC_summary:
            pyMC_df = pd.DataFrame(self.pyMC_summary)
            print("PyMC Summary:")
            print(pyMC_df)
        
        if self.scipy_curve_fit:
            scipy_df = pd.DataFrame(self.time_fit_values, columns=[
                'ThermalTime', 'ThermalTimeErr', 'CaptureTime', 'CaptureTimeErr', 
                'SourcePosition', 'LenEvents', 'Chi2Ndof', 'p-value', 'B'])
            print("Scipy Curve Fit Summary:")
            print(scipy_df)


        return Info

    def run_analysis(self, file_pattern: str = 'EventAmBeNeutronCandidates_fullwindowtest_*.csv',
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
            tasks = ['2d_histograms', '1d_histograms', 'lmfit_fit', 'summary']

        # Load and group data. A list of patterns is accepted so a campaign split
        # across several tags on disk can be loaded as one sample: grouping is by
        # source position and load_and_group_data appends rather than resetting, so
        # the tags merge per position. Only pass several patterns when they cover
        # DISJOINT positions -- overlapping tags would double-count silently.
        for pat in ([file_pattern] if isinstance(file_pattern, str) else file_pattern):
            self.load_and_group_data(pat)
        print(f"[basic] grouped into {len(self.source_groups)} source positions")

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
        if 'summary' in tasks:
            if self.lmfit_summary:
                print("Generating summary plots using LMFIT results...")
                summary_info = self.generate_summary_plots()
                return summary_info
            elif self.time_fit_values:
                print("Warning: No LMFIT results available, but scipy results found.")
                print("To use LMFIT results in summary plots, include 'lmfit_fit' in tasks.")
            else:
                print("No fit results available for summary plots.")

        return None


def main():
    """Example usage of the AmBeNeutronAnalyzer class."""
    
    # Initialize analyzer
    analyzer = AmBeNeutronAnalyzer()
    
    # Example 1: Run analysis with LMFIT-based summary plots
    print("=== Running analysis with LMFIT-based summary plots ===")
    analyzer.run_analysis(tasks=['2d_histograms', '1d_histograms'])

    # Example 2: Run full analysis with all tasks (both scipy and lmfit, summary uses lmfit)
    # print("=== Running full analysis ===")
    # analyzer.run_analysis(tasks=['2d_histograms', '1d_histograms', 'scipy_fit', 'lmfit_fit', 'pymc_fit', 'summary'])
    
    # Example 3: Run only lmfit fitting and summary
    # print("=== Running only LMFIT fitting and summary ===")
    # analyzer.run_analysis(tasks=['lmfit_fit', 'summary'])


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ambe CLI integration
# ---------------------------------------------------------------------------
def run(ctx, argv=None):
    """Run basic plots using paths and cuts from RunContext.

    This is the per-source-position book: load_and_group_data groups the candidate
    CSVs by (x, y, z) and the histogram tasks emit one page per position, which is
    the per-port-position view for whichever selection the config names.
    """
    from ..io import inputs_from_ctx
    from pathlib import Path
    import glob as _glob
    csv_paths = inputs_from_ctx(ctx, "candidate_csvs")
    data_dir = str(ctx.run_dir / "candidate_csvs") if not csv_paths else str(csv_paths[0].parent)
    # ctx.plot_path() composes the campaign/run suffix itself, so it takes the BARE
    # base name. Passing ctx.filename(...) into it applied the suffix twice and wrote
    # "basic_plots__<c>__<r>.pdf__<c>__<r>.png" -- a .png-suffixed file containing a
    # PDF, which no downstream stage could find.
    out_pdf = str(ctx.plot_path("basic_plots", "pdf"))

    # Derive the glob from the config instead of letting run_analysis fall back to
    # its default 'EventAmBeNeutronCandidates_fullwindowtest_*.csv'. That default
    # matches nothing for any current campaign, so this entry point used to produce
    # an empty book in silence -- load_and_group_data simply found no files.
    patterns = sorted({Path(str(p)).name for p in ctx.inputs.get("candidate_csvs", [])})
    if not patterns:
        raise SystemExit("config declares no inputs.candidate_csvs")

    n_found = 0
    for pat in patterns:
        n = len(_glob.glob(os.path.join(data_dir, pat)))
        print(f"[plots.basic] {data_dir}/{pat} -> {n} candidate CSVs")
        n_found += n
    if n_found == 0:
        raise SystemExit(
            f"no candidate CSVs match {patterns} in {data_dir} -- run "
            f"`ambe data process --config <cfg> --selection <box|mva>` first")

    # Plot-only hits cut, taken from the config so both decks state it and use the
    # same value. 0 / null disables it. Legacy default is 10 (see prepare_data).
    mch = ctx.cuts.get("plot_min_cluster_hits", 10)
    print(f"[plots.basic] plot-only cut clusterHits >= {mch}"
          f"{'  (disabled)' if not mch else ''}")
    analyzer = AmBeNeutronAnalyzer(data_directory=data_dir, output_pdf=out_pdf,
                                   min_cluster_hits=int(mch or 0),
                                   campaign_label=ctx.campaign or 'AmBe 2.0v4',
                                   selection_label=ctx.cuts.get('selection_label', ''))
    fc = ctx.fit_params
    if fc:
        analyzer.update_fitting_config(**{k: v for k, v in fc.items()
                                          if k in analyzer.fitting_config})
    # 'lmfit_fit' was NOT in this list, which is why the per-position capture-time
    # fits did not exist anywhere in the decks: lmfit_analysis() emits one fitted page
    # per source position carrying BOTH time constants -- therm and tau with their
    # errors, chi2 and chi2/ndof -- and it was unreachable from this entry point, so
    # the per-position book stopped at the raw histograms. Adding it here puts the
    # fits in the book for WHICHEVER selection the config names, which is what makes
    # the box and MVA appendices twins.
    #
    # 'summary' is deliberately NOT added. It runs generate_summary_plots(), whose
    # capture/thermal port x y heatmaps are now produced as deck figures by
    # boxcut_v4_campaign.py --do captureheat (J9/J10, K9/K10) with a convergence gate,
    # masked cells and the weighted mean in the title. Worse, it writes to
    # OutputPlots/{Capture,Thermal}Time_AmBeNeutrons_<campaign_tag>_LMFIT.png, and
    # campaign_tag is 'AmBe2.0v4' for BOTH selections -- so running the box book and
    # then the MVA book would silently overwrite one with the other under a filename
    # that names neither. Two differently-computed versions of the same map is the
    # trap; the deck figures are the ones to quote.
    analyzer.run_analysis(file_pattern=patterns,
                          tasks=["2d_histograms", "1d_histograms", "lmfit_fit"])
    print(f"[plots.basic] wrote {out_pdf}")


def cli(ctx, argv=None):
    run(ctx, argv)