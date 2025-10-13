import os          
import numpy as np
import uproot
from tqdm import trange
from scipy.stats import norm
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Tuple, Optional, Set, Any
import gc
from dataclasses import dataclass
import pandas as pd



@dataclass
class WaveformConfig:
    """Configuration parameters for waveform analysis."""
    pulse_start: int = 300
    pulse_end: int = 1200
    pulse_gamma: int = 400
    lower_pulse: int = 175
    pulse_max: int = 675
    NS_PER_ADC_SAMPLE: int = 2
    ADC_IMPEDANCE: int = 50
    ADC_TO_VOLT: float = 2.415 / (2 ** 12)
    ref_integral: float = 2.6e-2
    REF_ENERGY: float = 4.42  # Me


@dataclass
class CutCriteria:
    """Event selection criteria."""
    pe_min: float = 0
    pe_max: float = 700
    ccb_min: float = 0
    ccb_max: float = 1.0
    ct_min: float = 2000
    chits_min: int = 0
    cosmic_ct_threshold: float = 2000
    cosmic_pe_threshold: float = 700


class AmBeNeutronProcessing:
    """
    Efficient analyzer for AmBe neutron efficiency with memory management and modularity.
    """
    
    def __init__(self, config: Optional[WaveformConfig] = None, 
                 cuts: Optional[CutCriteria] = None):
        self.config = config or WaveformConfig()
        self.cuts = cuts or CutCriteria()
        
        # Source position mapping
        self.source_positions = {
            # Port 5 data
            4506: (0, 100, 0), 4505: (0, 50, 0), 4499: (0, 0, 0), 
            4507: (0, -50, 0), 4508: (0, -100, 0), 4496: (0, 0, 0),
            
            # Port 1 data
            4593: (0, 100, -75), 4590: (0, 50, -75), 4589: (0, 0, -75), 
            4596: (0, -50, -75), 4598: (0, -100, -75),
            
            # Port 4 data
            4656: (75, 100, 0), 4658: (75, 50, 0), 4660: (75, 0, 0),
            4662: (75, -50, 0), 4664: (75, -50, 0), 4665: (75, -50, 0), 
            4666: (75, -50, 0), 4667: (75, -50, 0), 4670: (75, -50, 0),
            4678: (75, -100, 0), 4683: (75, -100, 0), 4687: (75, -100, 0),
            
            # Port 3 data
            4633: (0, 50, 102), 4628: (0, -100, 102), 4629: (0, -100, 102),
            4635: (0, 0, 102), 4636: (0, 0, 102), 4640: (0, 0, 102), 4646: (0, 0, 102),
            4649: (0, -50, 102), 4651: (0, -100, 102),

            # Port 2 data
            4453: (0, 0, 75), 4603: (0, 100, 75), 4604: (0, 50, 75), 
            4605: (0, 50, 75), 4625: (0, -50, 75),

            # Outside the tank data
            4707: (0, 328, 0), 4708: (0, 328, 0),

            # Analysis Quality C2 - AmBe v2 Campaign 2 - July 2025
            # Old PMT - Port 5
            5682: (0, 0, 0), 5696: (0, -100, 0), 5707: (0, 55.3, 0), 
            5708: (0, 100, 0), 5710: (0, 100, 0),
            
            # Old PMT - Port 1
            5711: (0, -100, -75), 5712: (0, 0, -75), 5715: (0, 0, -75), 
            5716: (0, 0, -75), 5730: (0, 100, -75),
            
            # New PMT - Port 5
            5740: (0, 0, 0), 5797: (0, 100, 0), 5815: (0, -100, 0), 5816: (0, -100, 0), 5820: (0, -100, 0), 5822: (0, -100, 0), 5823: (0, 100, 0),

            # New PMT - Port 4
            5741: (75, 100, 0), 5742: (75, 100, 0), 5774: (75, 0, 0), 5775: (75, -100, 0),

            # New PMT - Port 3
            5776: (0, 100, 102), 5780: (0, 100, 102), 5782: (0, 0, 102), 5783: (0, -105.5, 102),

            # New PMT - Port 2
            5785: (0, 100, 75), 5789: (0, 0, 75), 5791: (0, -100, 75),

            # new PMT - Port 1
            5824:(0, 100, -75), 5825:(0, 0, -75),  5826:(0, -100, -75), 5828:(0, -100, -75),

            # Outside the tank without source
            5743: (0, 328, 0), 5778: (0, 328, 0), 5779: (0, 328, 0)
        }

    def get_source_location(self, run: int) -> Tuple[float, float, float]:
        """Get source position for a given run number."""
        if run in self.source_positions:
            return self.source_positions[run]
        
        raise ValueError(f'RUN NUMBER {run} DOESNT HAVE A SOURCE LOCATION!')

    def ambe_single_cut(self, cpe: float, ccb: float, ct: float, cn: int, chits: int) -> bool:
        """Apply AmBe single neutron selection cuts."""
        return (self.cuts.pe_min < cpe <= self.cuts.pe_max and
                self.cuts.ccb_min < ccb < self.cuts.ccb_max and
                ct >= self.cuts.ct_min and chits >= self.cuts.chits_min and
                cn == 1)

    def ambe_multiple_cut(self, cpe: float, ccb: float, ct: float, cn: int, chits: int) -> bool:
        """Apply AmBe multiple neutron selection cuts."""
        return (self.cuts.pe_min < cpe <= self.cuts.pe_max and
                self.cuts.ccb_min < ccb < self.cuts.ccb_max and
                ct >= self.cuts.ct_min and chits >= self.cuts.chits_min and
                cn != 1)

    def cosmic_cut(self, ct: float, cpe: float) -> bool:
        """Apply cosmic muon selection cuts."""
        return (ct < self.cuts.cosmic_ct_threshold or 
                cpe > self.cuts.cosmic_pe_threshold)

    def pairwise_relative_direction(self, hitX, hitY, hitZ, hitPE, hitT) -> Tuple[np.ndarray]:
        """
        Calculate cluster anisotropy/coherence from hit positions and timing.
        
        Args:
            hitX, hitY, hitZ: Hit position arrays
            hitPE: Hit photoelectron values
            hitT: Hit timing values
            
        Returns:
            Normalized direction vector (3D numpy array)
            PE weighted neutron vertex distance from source (float)
        """
        # pass cluster hits to this function, return anisotropy / coherence of the cluster
        
        hitX = np.asarray(hitX, dtype=np.float64)
        hitY = np.asarray(hitY, dtype=np.float64)
        hitZ = np.asarray(hitZ, dtype=np.float64)
        hitPE = np.asarray(hitPE, dtype=np.float64)
        hitT = np.asarray(hitT, dtype=np.float64)


        # combine and sort by time
        hits = list(zip(hitX, hitY, hitZ, hitPE, hitT))
        hits.sort(key=lambda h: h[4])
        
        mini = min(hitT)
        filtered_hits = []
        
        for hit in hits:
            x_i, y_i, z_i, pe_i, t_i = hit

            # remove very late times (chained to the first cluster hit)
            if (t_i - mini) > 50:  # 50 ns is the maximum distance between the PMT rack (shortest path for direct light)
                continue           # any hit times after this are assumed to be from reflections
                
            filtered_hits.append(hit)

        # extract filtered values
        filtX = np.array([h[0] for h in filtered_hits])
        filtY = np.array([h[1] for h in filtered_hits])
        filtZ = np.array([h[2] for h in filtered_hits])
        filtPE = np.array([h[3] for h in filtered_hits])
        
        n = len(filtX)
        if n < 2:
            return np.array([0.0, 0.0, 0.0])

        vec = np.zeros(3)
        for i in range(n):
            for j in range(i+1, n):
                dx = filtX[j] - filtX[i]
                dy = filtY[j] - filtY[i]
                dz = filtZ[j] - filtZ[i]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                if r == 0:
                    continue
                
                # weighted direction between hits
                dvec = np.array([dx, dy, dz]) / r
                weight = filtPE[i] * filtPE[j]
                vec += weight * dvec
                #Neutron_vertex = np.sqrt((dvec[0]-(sourceX/100))**2 + (dvec[1]-(sourceY/100))**2 + (dvec[2]-(sourceZ/100))**2)

        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])

        return vec / norm
    
    def time_of_flight_correction(self, hitX, hitY, hitZ, hitPE, hitT, sourceX, sourceY, sourceZ, event_hits_data=None) -> str:
        """
        Calculate time of flight correction from hit positions and timing.
        
        Args:
            hitX, hitY, hitZ: Hit position arrays
            hitPE: Hit photoelectron values
            hitT: Hit timing values
            sourceX, sourceY, sourceZ: Source position coordinates
            event_hits_data: Optional list of tuples for multi-cluster calculation
        Returns:
            Time of flight corrected time differences as space-separated string
        """
        SoL = 0.299792458 * 3/4
        
        # If event_hits_data is provided, use multi-cluster approach
        if event_hits_data is not None:
            all_hits = []
            for cluster_hits in event_hits_data:
                cluster_hitX, cluster_hitY, cluster_hitZ, cluster_hitPE, cluster_hitT = cluster_hits
                cluster_hitX = np.asarray(cluster_hitX, dtype=np.float64)
                cluster_hitY = np.asarray(cluster_hitY, dtype=np.float64)
                cluster_hitZ = np.asarray(cluster_hitZ, dtype=np.float64)
                cluster_hitPE = np.asarray(cluster_hitPE, dtype=np.float64)
                cluster_hitT = np.asarray(cluster_hitT, dtype=np.float64)
                
                cluster_hit_list = list(zip(cluster_hitX, cluster_hitY, cluster_hitZ, cluster_hitPE, cluster_hitT))
                all_hits.extend(cluster_hit_list)
            
            hits = all_hits
        else:
            # Single cluster approach (original logic)
            hitX = np.asarray(hitX, dtype=np.float64)
            hitY = np.asarray(hitY, dtype=np.float64)
            hitZ = np.asarray(hitZ, dtype=np.float64)
            hitPE = np.asarray(hitPE, dtype=np.float64)
            hitT = np.asarray(hitT, dtype=np.float64)
            hits = list(zip(hitX, hitY, hitZ, hitPE, hitT))
        
        # Sort by time and calculate differences (same logic for both cases)
        hits.sort(key=lambda h: h[4])
        n = len(hits)
        if n < 2:
            return ""
        
        tdiffs = []
        for i in range(n):
            x_i, y_i, z_i, pe_i, t_i = hits[i]
            dist_i = np.sqrt((x_i - sourceX)**2 + (y_i - sourceY)**2 + (z_i - sourceZ)**2)
            tcorr_i = t_i - dist_i/SoL
            
            for j in range(i+1, n):
                x_j, y_j, z_j, pe_j, t_j = hits[j]
                dist_j = np.sqrt((x_j - sourceX)**2 + (y_j - sourceY)**2 + (z_j - sourceZ)**2)
                tcorr_j = t_j - dist_j/SoL
                
                tdiffs.append(tcorr_j - tcorr_i)
        
        # Return as space-separated string for easy parsing
        return " ".join(f"{val:.6f}" for val in tdiffs)

    def analyze_waveform(self, hist_values: np.ndarray, hist_edges: np.ndarray) -> Tuple[float, float, bool]:
        """
        Analyze a single waveform and return IC_adjusted, baseline, and acceptance status.
        
        Returns:
            Tuple of (IC_adjusted, baseline, is_accepted)
        """
        # Fit baseline
        baseline, sigma = norm.fit(hist_values)
        
        # Calculate pulse integral
        pulse_mask = ((hist_edges[:-1] > self.config.pulse_start) & 
                     (hist_edges[:-1] < self.config.pulse_end))
        IC = np.sum(hist_values[pulse_mask] - baseline)
        IC_adjusted = (self.config.NS_PER_ADC_SAMPLE / self.config.ADC_IMPEDANCE) * IC
        IC_adjusted *= self.config.ADC_TO_VOLT
        
        # Check acceptance criteria
        if not (0.7 < IC_adjusted < 1.5):
            return IC_adjusted, baseline, False
            
        # Check for second pulse
        post_pulse_mask = hist_edges[:-1] > self.config.pulse_end
        post_pulse_values = hist_values[post_pulse_mask]
        another_pulse = np.any(post_pulse_values > (7 + sigma + baseline))
        
        return IC_adjusted, baseline, not another_pulse

    def plot_waveform_sample(self, hist_edges: np.ndarray, hist_values: np.ndarray, 
                           baseline: float, timestamp: str, run: str, 
                           IC_adjusted: float, status: str, save_dir: str = 'verbose'):
        """Plot a sample waveform for visualization."""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 4))
        hist_values_bs = hist_values - baseline
        # Convert ADC counts to volts
        hist_values_volts = hist_values_bs * self.config.ADC_TO_VOLT
        plt.plot(hist_edges[:-1], hist_values_volts, label='Waveform')
        plt.axvspan(self.config.pulse_start, self.config.pulse_end, 
                   color='yellow', alpha=0.3, label='Integration Window')
        plt.title(f'{status} Waveform (timestamp: {timestamp}, Run: {run}), IC: {IC_adjusted:.2f}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{save_dir}/{status}Waveform_{timestamp}_Run{run}.png', dpi=300)
        plt.close()

    def process_run_waveforms(self, run: str, waveform_dir: str, 
                            campaign: int = 1, 
                            save_waveform_samples: bool = False,
                            max_samples: int = 10) -> Dict[str, Any]:
        """
        Process all waveforms for a single run with memory-efficient approach.
        
        Args:
            run: Run number as string
            waveform_dir: Directory containing waveform files
            campaign: Campaign number (1 or 2)
            save_waveform_samples: Whether to save sample waveforms
            max_samples: Maximum number of sample waveforms to save
            
        Returns:
            Dictionary with run results
        """
        print(f"\nProcessing Run: {run}")
        print('-' * 50)
        
        x_pos, y_pos, z_pos = self.get_source_location(int(run))
        print(f'Source position (x,y,z): ({x_pos},{y_pos},{z_pos})')

        # Setup folder pattern based on campaign
        if campaign == 1:
            folder_pattern = re.compile(r'^RWM_\d+')
        elif campaign == 2:
            folder_pattern = re.compile(r'^BRF_\d+')
        else:
            raise ValueError("Campaign must be 1 or 2")

        # Initialize counters and storage
        good_events = []
        accepted_events = 0
        rejected_events = 0
        seen_timestamps = set()
        
        # Use generators and batch processing for memory efficiency
        ic_values_batch = []
        ic_accepted_batch = []
        
        waveform_files = os.listdir(os.path.join(waveform_dir, run))
        
        print('Loading and processing waveforms...')
        for file_idx in trange(len(waveform_files)):
            waveform_filepath = os.path.join(waveform_dir, run, waveform_files[file_idx])
            
            with uproot.open(waveform_filepath) as root:
                folder_names = [name for name in root.keys() if folder_pattern.match(name)]
                
                for folder in folder_names:
                    timestamp = folder.split('_')[-1].split(';')[0]
                    if timestamp in seen_timestamps:
                        continue
                    seen_timestamps.add(timestamp)
                    
                    try:
                        hist = root[folder]
                        hist_values = hist.values()
                        hist_edges = hist.axes[0].edges()
                        
                        IC_adjusted, baseline, is_accepted = self.analyze_waveform(hist_values, hist_edges)
                        ic_values_batch.append(IC_adjusted)
                        
                        if is_accepted:
                            good_events.append(int(timestamp))
                            accepted_events += 1
                            ic_accepted_batch.append(IC_adjusted)
                            
                            # Save sample waveforms if requested
                            if save_waveform_samples and accepted_events <= max_samples:
                                self.plot_waveform_sample(hist_edges, hist_values, baseline, 
                                                        timestamp, run, IC_adjusted, "Accepted")
                        else:
                            rejected_events += 1
                            
                            # Save sample rejected waveforms
                            if save_waveform_samples and rejected_events in [1, 10, 20]:
                                self.plot_waveform_sample(hist_edges, hist_values, baseline, 
                                                        timestamp, run, IC_adjusted, "Rejected")
                    
                    except Exception as e:
                        print(f"Could not access '{folder}': {e}")
                        continue
            
            # Periodic garbage collection for large datasets
            if file_idx % 100 == 0:
                gc.collect()

        total = accepted_events + rejected_events
        acceptance_rate = (100 * accepted_events / total) if total > 0 else 0
        rejection_rate = (100 * rejected_events / total) if total > 0 else 0
        
        print(f'Total acquisitions: {total}')
        print(f'Accepted: {accepted_events} ({acceptance_rate:.2f}%)')
        print(f'Rejected: {rejected_events} ({rejection_rate:.2f}%)')
        
        return {
            'source_position': (x_pos, y_pos, z_pos),
            'accepted_events': accepted_events,
            'rejected_events': rejected_events,
            'total_waveforms': total,
            'good_events': set(good_events),
            'ic_values': ic_values_batch,
            'ic_accepted': ic_accepted_batch
        }

    def analyze_multiple_runs(self, data_directory: str, waveform_dir: str, 
                            file_pattern: re.Pattern, campaign: int = 1,
                            runinfo: str = 'default',
                            save_waveform_samples: bool = False,
                            plot_ic_distributions: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple runs with efficient memory management.
        
        Args:
            data_directory: Directory containing data files
            waveform_dir: Directory containing waveform files
            file_pattern: Regex pattern to match filenames
            campaign: Campaign number (1 or 2)
            runinfo: Run information string for output filenames
            save_waveform_samples: Whether to save sample waveforms
            plot_ic_distributions: Whether to plot IC distributions
            
        Returns:
            Dictionary with all run results
        """
        # Get run numbers from file names
        run_numbers = []
        file_names = []
        
        for file_name in os.listdir(data_directory):
            match = file_pattern.match(file_name)
            if match:
                run_number = int(match.group(1))
                run_numbers.append(str(run_number))
                file_names.append(os.path.join(data_directory, file_name))
        
        if not run_numbers:
            raise ValueError(f"No files found matching pattern in {data_directory}")
        
        print(f"Found {len(run_numbers)} runs to process")
        
        results = {}
        all_ic_values = []
        all_ic_accepted = []
        
        # Create PDF for plots
        os.makedirs('verbose', exist_ok=True)
        with PdfPages(f'verbose/AllRuns_IC_adjusted_{runinfo}.pdf') as pdf:
            
            for run in run_numbers:
                try:
                    run_result = self.process_run_waveforms(
                        run, waveform_dir, campaign, 
                        save_waveform_samples=save_waveform_samples
                    )
                    results[run] = run_result
                    
                    # Extend combined lists efficiently
                    all_ic_values.extend(run_result['ic_values'])
                    all_ic_accepted.extend(run_result['ic_accepted'])
                    
                    # Plot per-run distributions
                    if plot_ic_distributions:
                        self._plot_run_ic_distribution(run_result['ic_values'], run, pdf)
                    
                    # Clear run-specific data to free memory
                    run_result['ic_values'] = []  # Keep structure but clear data
                    run_result['ic_accepted'] = []
                    
                except Exception as e:
                    print(f"Failed to process run {run}: {e}")
                    continue
                
                # Periodic garbage collection
                gc.collect()
        
        # Generate summary plots
        if plot_ic_distributions and all_ic_values:
            self._plot_combined_ic_distributions(all_ic_values, all_ic_accepted, runinfo)
        
        print(f"Analysis complete. Processed {len(results)} runs successfully.")
        return results, run_numbers, file_names

    def _plot_run_ic_distribution(self, ic_values: List[float], run: str, pdf):
        """Plot IC distribution for a single run."""
        if not ic_values:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.hist(ic_values, bins=200, alpha=0.7, color='blue', range=(0, 1400))
        ax1.set_xlabel('IC_adjusted')
        ax1.set_ylabel('Number of Events')
        ax1.set_title(f'IC adjusted values for run: {run}')
        
        # Log scale
        ax2.hist(ic_values, bins=200, alpha=0.7, color='blue', range=(0, 1400), log=True)
        ax2.set_xlabel('IC_adjusted')
        ax2.set_ylabel('Number of Events (log scale)')
        ax2.set_title(f'IC adjusted values (log) for run: {run}')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_combined_ic_distributions(self, all_ic_values: List[float], 
                                       all_ic_accepted: List[float], runinfo: str):
        """Plot combined IC distributions for all runs."""
        os.makedirs('verbose', exist_ok=True)
        
        # All IC values
        plt.figure(figsize=(10, 6))
        plt.hist(all_ic_values, bins=200, alpha=0.7, color='blue', range=(0, 1400))
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events')
        plt.title('All IC adjusted Values for all runs')
        plt.tight_layout()
        plt.savefig(f'verbose/IC_adjusted_AllEvents_{runinfo}.png', dpi=300)
        plt.close()
        
        # All IC values (log scale)
        plt.figure(figsize=(10, 6))
        plt.hist(all_ic_values, bins=200, alpha=0.7, color='blue', range=(0, 1400), log=True)
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events (log scale)')
        plt.title('All IC adjusted Values for all runs (log scale)')
        plt.tight_layout()
        plt.savefig(f'verbose/Log_IC_adjusted_AllEvents_{runinfo}.png', dpi=300)
        plt.close()
        
        # Accepted IC values
        if all_ic_accepted:
            plt.figure(figsize=(10, 6))
            plt.hist(all_ic_accepted, bins=200, alpha=0.7, color='orange', range=(0, 1400))
            plt.xlabel('IC_adjusted accepted')
            plt.ylabel('Number of Events')
            plt.title('Accepted IC_adjusted Values for all runs')
            plt.tight_layout()
            plt.savefig(f'verbose/IC_adjusted_AcceptedEvents_{runinfo}.png', dpi=300)
            plt.close()

    def load_event_data(self, file_path: str, which_tree: int = 0) -> Dict[str, np.ndarray]:
        """
        Load event data from ROOT file with memory-efficient approach.
        
        Args:
            file_path: Path to ROOT file
            which_tree: Tree type (0 or other)
            
        Returns:
            Dictionary containing event data arrays
        """
        print(f'Loading AmBe event data from {file_path}...')
        
        with uproot.open(file_path) as file_1:
            Event = file_1["Event"]
            
            # Load common branches
            data = {
                "eventNumber": Event["eventNumber"].array(),
                "eventTimeTank": Event["eventTimeTank"].array(),
                "clusterTime": Event["clusterTime"].array(),
                "clusterPE": Event["clusterPE"].array(),
                "clusterChargeBalance": Event["clusterChargeBalance"].array(),
                "clusterHits": Event["clusterHits"].array(),
                "hitX": Event["Cluster_HitX"].array(),
                "hitY": Event["Cluster_HitY"].array(),
                "hitZ": Event["Cluster_HitZ"].array()
            }
            
            # Tree-dependent branches
            if which_tree == 0:
                data.update({
                    "clusterNumber": Event["clusterNumber"].array(),
                    "hitT": Event["hitT"].array(),
                    "hitPE": Event["hitPE"].array(),
                    "hitDetID": Event["hitDetID"].array(),
                })
            else:
                data.update({
                    "clusterNumber": Event["numberOfClusters"].array(),
                    "hitT": Event["Cluster_HitT"].array(),
                    "hitPE": Event["Cluster_HitPE"].array(),
                    "hitDetID": Event["Cluster_HitDetID"].array(),
                })
        
        return data

    def process_events_efficient(self, event_data: Dict[str, np.ndarray], 
                                good_events: Set[int], x_pos: float, y_pos: float, z_pos: float,
                                efficiency_data: Optional[Dict] = None) -> Tuple[List, Dict[str, int]]:
        """
        Process events with memory-efficient vectorized operations where possible.
        
        Args:
            event_data: Dictionary containing event data
            good_events: Set of good event timestamps
            x_pos, y_pos, z_pos: Source position coordinates
            efficiency_data: Optional efficiency tracking dictionary
            
        Returns:
            Tuple of (processed_data_lists, statistics_dict)
        """
        # Extract data arrays
        EN = event_data["eventNumber"]
        ETT = event_data["eventTimeTank"]
        CT = event_data["clusterTime"]
        CPE = event_data["clusterPE"]
        CCB = event_data["clusterChargeBalance"]
        CH = event_data["clusterHits"]
        CN = event_data["clusterNumber"]
        hT = event_data["hitT"]
        hX = event_data["hitX"]
        hY = event_data["hitY"]
        hZ = event_data["hitZ"]
        hPE = event_data["hitPE"]
        hID = event_data["hitDetID"]

        
        # Initialize storage lists
        processed_data = {
            'cluster_time': [], 'cluster_charge': [], 'cluster_QB': [], 'cluster_hits': [],
            'hit_times': [], 'hit_x': [], 'hit_y': [], 'hit_z': [], 'hit_charges': [], 'hit_ids': [],
            'source_position': [[], [], []], 'event_ids': [], 'event_tank_time': [],
            'prompt_cluster_time': [], 'prompt_cluster_charge': [], 'prompt_cluster_QB': [], 'hit_delta_t': [],
            'cluster_direction': [], 'neutron_tof_correction': []
        }
        
        # Initialize counters
        stats = {
            'total_events': 0,
            'cosmic_events': 0,
            'neutron_cand_count': 0,
            'multiple_neutron_cand_count': 0
        }
        
        key = (x_pos, y_pos, z_pos)
        repeated_event_id = set()
        
        # Dictionary to group clusters by eventTankTime for multi-cluster ToF correction
        event_clusters = defaultdict(list)
        
        print(f"Processing {len(EN)} events...")
        
        for i in trange(len(EN)):
            if ETT[i] not in good_events:
                continue
                
            stats['total_events'] += 1
            if efficiency_data is not None:
                efficiency_data[key][0] += 1
            
            # Process clusters for this event
            event_has_cosmic = False
            event_neutron_candidates = []
            event_multiple_candidates = []
            
            for k in range(len(CT[i])):
                # Check cosmic cut first (highest priority)
                if self.cosmic_cut(CT[i][k], CPE[i][k]):
                    processed_data['prompt_cluster_time'].append(CT[i][k])
                    processed_data['prompt_cluster_charge'].append(CPE[i][k])
                    processed_data['prompt_cluster_QB'].append(CCB[i][k])
                    
                    if not event_has_cosmic:  # Count only once per event
                        stats['cosmic_events'] += 1
                        event_has_cosmic = True
                        if efficiency_data is not None:
                            efficiency_data[key][1] += 1
                    break  # Skip other cuts for this event
                
                # Check single neutron cuts
                if self.ambe_single_cut(CPE[i][k], CCB[i][k], CT[i][k], CN[i], CH[i][k]):
                    if CPE[i][k] != float('-inf'):
                        event_neutron_candidates.append(k)
                
                # Check multiple neutron cuts
                if self.ambe_multiple_cut(CPE[i][k], CCB[i][k], CT[i][k], CN[i], CH[i][k]):
                    if CPE[i][k] != float('-inf'):
                        event_multiple_candidates.append(k)
            
            # Collect hit data for all clusters in this event for multi-cluster ToF calculation
            all_candidates = event_neutron_candidates + event_multiple_candidates
            if all_candidates:
                event_hit_data = []
                for k in all_candidates:
                    event_hit_data.append((hX[i][k], hY[i][k], hZ[i][k], hPE[i][k], hT[i][k]))
                
                # Store event-level hit data for multi-cluster ToF calculation
                event_clusters[ETT[i]] = event_hit_data
            
            # Process single neutron candidates
            for k in event_neutron_candidates:
                self._append_cluster_data(processed_data, i, k, EN, ETT, CT, CPE, CCB, CH, 
                                        hT, hX, hY, hZ, hPE, hID, x_pos, y_pos, z_pos, 
                                        event_clusters.get(ETT[i]))
                stats['neutron_cand_count'] += 1
                if efficiency_data is not None:
                    efficiency_data[key][2] += 1
            
            # Process multiple neutron candidates (count event only once)
            if event_multiple_candidates:
                for k in event_multiple_candidates:
                    self._append_cluster_data(processed_data, i, k, EN, ETT, CT, CPE, CCB, CH, 
                                            hT, hX, hY, hZ, hPE, hID, x_pos, y_pos, z_pos,
                                            event_clusters.get(ETT[i]))

                if EN[i] not in repeated_event_id:
                    stats['multiple_neutron_cand_count'] += 1
                    repeated_event_id.add(EN[i])
                    if efficiency_data is not None:
                        efficiency_data[key][3] += 1
        
        # Print statistics
        self._print_processing_stats(stats)
        
        return processed_data, stats

    def _append_cluster_data(self, processed_data: Dict, i: int, k: int,
                           EN, ETT, CT, CPE, CCB, CH, hT, hX, hY, hZ, hPE, hID,
                           x_pos: float, y_pos: float, z_pos: float,
                           event_hit_data=None):
        """Helper function to append cluster data to processed_data dictionary."""
        processed_data['cluster_time'].append(CT[i][k])
        processed_data['cluster_charge'].append(CPE[i][k])
        processed_data['cluster_QB'].append(CCB[i][k])
        processed_data['cluster_hits'].append(CH[i][k])
        processed_data['hit_times'].append(hT[i][k])
        processed_data['hit_x'].append(hX[i][k])
        processed_data['hit_y'].append(hY[i][k])
        processed_data['hit_z'].append(hZ[i][k])
        processed_data['hit_charges'].append(hPE[i][k])
        processed_data['hit_ids'].append(hID[i][k])
        processed_data['source_position'][0].append(x_pos)
        processed_data['source_position'][1].append(y_pos)
        processed_data['source_position'][2].append(z_pos)
        processed_data['event_ids'].append(EN[i])
        processed_data['event_tank_time'].append(ETT[i])
        processed_data['hit_delta_t'].append(np.max(hT[i][k] - np.min(hT[i][k])) if CH[i][k] > 1 else 0)
        
        # Calculate cluster direction/anisotropy
        direction_vec = self.pairwise_relative_direction(hX[i][k], hY[i][k], hZ[i][k], hPE[i][k], hT[i][k])
        processed_data['cluster_direction'].append(direction_vec)
        
        # Calculate ToF correction - multi-cluster if event_hit_data provided, otherwise single-cluster
        neutron_tof_correction = self.time_of_flight_correction(
            hX[i][k], hY[i][k], hZ[i][k], hPE[i][k], hT[i][k], 
            x_pos, y_pos, z_pos, event_hit_data
        )
        processed_data['neutron_tof_correction'].append(neutron_tof_correction)


    def _print_processing_stats(self, stats: Dict[str, int]):
        """Print processing statistics."""
        total_events = stats['total_events']
        cosmic_events = stats['cosmic_events']
        neutron_cand_count = stats['neutron_cand_count']
        multiple_neutron_cand_count = stats['multiple_neutron_cand_count']
        
        print(f'\nProcessing Statistics:')
        print(f'Total AmBe events after waveform cuts: {total_events}')
        
        if total_events == 0:
            print("No events found. Cannot calculate cosmic percentage.")
        else:
            cosmic_percent = round(100 * cosmic_events / total_events, 2)
            print(f'Cosmic events: {cosmic_events} ({cosmic_percent}%)')
            print(f'AmBe-triggered events: {total_events - cosmic_events}')
        
        print(f'Single neutron candidates: {neutron_cand_count}')
        print(f'Multiple neutron candidates: {multiple_neutron_cand_count}')

    def run_complete_analysis_pipeline(self, data_directory: str, waveform_dir: str, 
                                      file_pattern: re.Pattern, campaign: int = 1,
                                      runinfo: str = 'default', which_tree: int = 1,
                                      save_waveform_samples: bool = False,
                                      plot_ic_distributions: bool = True) -> Dict[str, Any]:
        """
        Complete analysis pipeline that replicates AnalysisRun.py functionality.
        This integrates waveform analysis, event processing, and CSV output generation.
        
        Args:
            data_directory: Directory containing BeamClusterAnalysis ntuples
            waveform_dir: Directory containing raw AmBe PMT waveforms
            file_pattern: Regex pattern to match filenames
            campaign: Campaign number (1 or 2)
            runinfo: Run information string for output filenames
            which_tree: PhaseIITreeMaker (0) or ANNIEEventTreeMaker (1) tool
            save_waveform_samples: Whether to save sample waveforms
            plot_ic_distributions: Whether to plot IC distributions
            
        Returns:
            Dictionary with complete analysis results
        """
        print("="*80)
        print("STARTING COMPLETE AmBe NEUTRON ANALYSIS PIPELINE")
        print("="*80)
        
        # Create output directories
        os.makedirs('TriggerSummary', exist_ok=True)
        os.makedirs('EventAmBeNeutronCandidatesData', exist_ok=True)
        os.makedirs('verbose', exist_ok=True)
        
        # Initialize efficiency tracking
        efficiency_data = defaultdict(lambda: [0, 0, 0, 0])  # [total, cosmic, single, multiple]
        
        # Step 1: Analyze waveforms
        print("\n" + "="*60)
        print("STEP 1: Waveform Analysis")
        print("="*60)
        
        waveform_results, run_numbers, file_names = self.analyze_multiple_runs(
            data_directory, waveform_dir, file_pattern, campaign,
            runinfo, save_waveform_samples, plot_ic_distributions
        )
        
        # Create waveform summary DataFrame and save
        waveform_df = pd.DataFrame.from_dict(waveform_results, orient='index')
        
        if 'good_events' in waveform_df.columns:
            waveform_df = waveform_df.drop(columns=['good_events'])
        
        waveform_df[['x_pos', 'y_pos', 'z_pos']] = pd.DataFrame(
            waveform_df['source_position'].tolist(), index=waveform_df.index)
        waveform_df = waveform_df.drop(columns=['source_position'])
        
        waveform_df = waveform_df.groupby(['x_pos', 'y_pos', 'z_pos'], as_index=False).sum()
        waveform_df.to_csv(f'TriggerSummary/AmBeWaveformResults_{runinfo}.csv', index=False)
        print(f"✓ Saved waveform summary to TriggerSummary/AmBeWaveformResults_{runinfo}.csv")
        
        # Step 2: Process each run for event analysis
        print("\n" + "="*60)
        print("STEP 2: Event Analysis for Each Run")
        print("="*60)
        
        for c1, run in enumerate(run_numbers):
            print(f"\nProcessing run {run} ({c1+1}/{len(run_numbers)})")
            print('-' * 50)
            
            # Get waveform results for this run
            good_events = waveform_results[run]["good_events"]
            x_pos, y_pos, z_pos = waveform_results[run]["source_position"]
            file_path = file_names[c1]
            
            print(f"Source position: ({x_pos}, {y_pos}, {z_pos})")
            print(f"Good events from waveform analysis: {len(good_events)}")
            
            # Load event data for this specific run
            event_data = self.load_event_data(file_path, which_tree)
            
            # Process events for this run
            processed_data, event_stats = self.process_events_efficient(
                event_data, good_events, x_pos, y_pos, z_pos, efficiency_data
            )
            
            print('----------------------------------------------------------------')
            print(f'Total AmBe neutron candidates: {len(processed_data["cluster_time"])}')
            
            # Create DataFrames for this run
            run_int = int(run)
            print(f'Event run number: {run_int}')
                        
            # Main neutron candidates DataFrame
            df = pd.DataFrame({
                "clusterTime": processed_data['cluster_time'],
                "clusterPE": processed_data['cluster_charge'],
                "clusterChargeBalance": processed_data['cluster_QB'],
                "clusterHits": processed_data['cluster_hits'],
                "hitT": processed_data['hit_times'],  # Keep as lists for now
                "hitX": processed_data['hit_x'],
                "hitY": processed_data['hit_y'],
                "hitZ": processed_data['hit_z'],
                "hitQ": processed_data['hit_charges'],
                "hitPE": processed_data['hit_charges'],
                "hitID": processed_data['hit_ids'],
                "hit_delta_t": processed_data['hit_delta_t'],
                "sourceX": processed_data['source_position'][0],
                "sourceY": processed_data['source_position'][1],
                "sourceZ": processed_data['source_position'][2],
                "eventID": processed_data['event_ids'],
                "eventTankTime": processed_data['event_tank_time'],
                "clusterDirection": processed_data['cluster_direction'],
                'neutronTofCorrection': processed_data['neutron_tof_correction']

            })
            
            print("Sample of neutron candidates data:")
            print(df.head())
            
            # Save neutron candidates data
            output_file = f'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_{run_int}.csv'
            df.to_csv(output_file, float_format="%.6f", index=False)
            print(f"✓ Saved neutron candidates to {output_file}")
            
            # Prompt neutrons DataFrame
            prompt_df = pd.DataFrame({
                "prompt_clusterTime": processed_data['prompt_cluster_time'],
                "prompt_clusterPE": processed_data['prompt_cluster_charge'],
                "prompt_clusterChargeBalance": processed_data['prompt_cluster_QB'],
            })
            
            # Save prompt neutrons data
            prompt_file = f'EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_{run_int}.csv'
            prompt_df.to_csv(prompt_file, index=False)
            print(f"✓ Saved prompt neutron candidates to {prompt_file}")
            
            # Clear data to free memory
            del processed_data, df, prompt_df
            gc.collect()
        
        # Step 3: Create efficiency summary
        print("\n" + "="*60)
        print("STEP 3: Efficiency Summary")
        print("="*60)
        
        df_eff = pd.DataFrame([
            {
                "x_pos": key[0], "y_pos": key[1], "z_pos": key[2],
                "total_events": val[0],
                "cosmic_events": val[1],
                "ambe_triggers": val[0] - val[1],
                "single_neutron_candidates": val[2],
                "multiple_neutron_candidates": val[3],
                "unique_neutron_triggers": val[2] + val[3]
            }
            for key, val in efficiency_data.items()
        ])
        
        efficiency_file = f'TriggerSummary/AmBeTriggerSummary_{runinfo}.csv'
        df_eff.to_csv(efficiency_file, index=False)
        print(f"✓ Saved efficiency summary to {efficiency_file}")
        
        # Print final summary
        print("\n" + "="*80)
        print("ANALYSIS PIPELINE COMPLETE")
        print("="*80)
        print(f"✓ Processed {len(run_numbers)} runs successfully")
        print(f"✓ Generated {len(run_numbers) * 2} CSV files for individual runs")
        print(f"✓ Generated 2 summary CSV files")
        print("\nOutput files generated:")
        print(f"  - TriggerSummary/AmBeWaveformResults_{runinfo}.csv")
        print(f"  - TriggerSummary/AmBeTriggerSummary_{runinfo}.csv")
        print(f"  - EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_<run>.csv (for each run)")
        print(f"  - EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_<run>.csv (for each run)")
        
        if plot_ic_distributions:
            print(f"  - verbose/AllRuns_IC_adjusted_{runinfo}.pdf")
            print(f"  - verbose/IC_adjusted_AllEvents_{runinfo}.png")
            print(f"  - verbose/Log_IC_adjusted_AllEvents_{runinfo}.png")
            print(f"  - verbose/IC_adjusted_AcceptedEvents_{runinfo}.png")
        
        return {
            'waveform_results': waveform_results,
            'run_numbers': run_numbers,
            'file_names': file_names,
            'efficiency_data': dict(efficiency_data),
            'output_files': {
                'waveform_summary': f'TriggerSummary/AmBeWaveformResults_{runinfo}.csv',
                'efficiency_summary': efficiency_file,
                'neutron_candidates_pattern': f'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_*.csv',
                'prompt_candidates_pattern': f'EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_*.csv'
            }
        }


def main():
    """
    Main function that replicates AnalysisRun.py functionality with interactive prompts.
    """
    print("="*80)
    print("AmBe NEUTRON ANALYSIS - Integrated Pipeline")
    print("="*80)
    
    # Interactive configuration (matching AnalysisRun.py)
    print("\nConfiguration Setup:")
    print("-" * 20)
    
    # Get campaign information
    while True:
        try:
            campaign = int(input('What campaign is this? (1/2): '))
            if campaign in [1, 2]:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number (1 or 2)")
    
    # Set file pattern based on campaign
    if campaign == 1:
        file_pattern = re.compile(r'AmBe_(\d+)_v\d+\.ntuple\.root')
        print("✓ Using Campaign 1 file pattern: AmBe_<run>_v<version>.ntuple.root")
    elif campaign == 2:
        file_pattern = re.compile(r'BeamCluster_(\d+)\.root')
        print("✓ Using Campaign 2 file pattern: BeamCluster_<run>.root")
    
    # Get run information
    runinfo = input('What type of run is this? (AmBe/Outside_source/C1/C2/ClusterCuts): ')
    runinfo = str(runinfo)
    print(f"✓ Run type: {runinfo}")
    
    # Configuration options
    which_tree = 1  # ANNIEEventTreeMaker (1) - matching AnalysisRun.py default
    print(f"✓ Using tree type: {'ANNIEEventTreeMaker' if which_tree == 1 else 'PhaseIITreeMaker'}")
    
    # Directory configuration (matching AnalysisRun.py)
    data_directory = '../AmBe_BeamCluster/'
    waveform_dir = '../AmBe_waveforms/'

    #data_directory = '/Volumes/One Touch/AmBe/'
    #waveform_dir = '/Volumes/One Touch/AmBe/'
    
    print(f"\nDirectory Configuration:")
    print(f"  Data directory: {data_directory}")
    print(f"  Waveform directory: {waveform_dir}")
    
    # Optional: Ask user if they want to modify directories
    modify_dirs = input('\nUse default directories? (y/n) [y]: ').strip().lower()
    if modify_dirs == 'n':
        data_directory = input('Enter data directory path: ').strip()
        waveform_dir = input('Enter waveform directory path: ').strip()
    
    # Analysis options
    save_samples = input('Save waveform samples? (y/n) [n]: ').strip().lower() == 'y'
    plot_ic = input('Plot IC distributions? (y/n) [y]: ').strip().lower() != 'n'
    
    print(f"\nAnalysis Options:")
    print(f"  Save waveform samples: {save_samples}")
    print(f"  Plot IC distributions: {plot_ic}")
    
    # Create analyzer with default configuration
    config = WaveformConfig()
    cuts = CutCriteria()
    analyzer = AmBeNeutronProcessing(config, cuts)
    
    print(f"\nAnalyzer Configuration:")
    print(f"  Pulse window: {config.pulse_start} - {config.pulse_end} ns")
    print(f"  PE range: {cuts.pe_min} - {cuts.pe_max}")
    print(f"  Charge balance max: {cuts.ccb_max}")
    print(f"  Cluster time min: {cuts.ct_min} ns")
    
    # Confirm before starting
    print("\n" + "="*60)
    proceed = input('Start analysis? (y/n) [y]: ').strip().lower()
    if proceed == 'n':
        print("Analysis cancelled.")
        return
    
    try:
        # Run the complete analysis pipeline
        results = analyzer.run_complete_analysis_pipeline(
            data_directory=data_directory,
            waveform_dir=waveform_dir,
            file_pattern=file_pattern,
            campaign=campaign,
            runinfo=runinfo,
            which_tree=which_tree,
            save_waveform_samples=save_samples,
            plot_ic_distributions=plot_ic
        )
        
        # Final summary
        print(f"\n ANALYSIS COMPLETE!")
        print(f"✓ Successfully processed {len(results['run_numbers'])} runs")
        print(f"✓ Output files saved in TriggerSummary/ and EventAmBeNeutronCandidatesData/")
        
        if plot_ic:
            print(f"✓ IC distribution plots saved in verbose/")
        
        return results
        
    except Exception as e:
        print(f"\n ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()