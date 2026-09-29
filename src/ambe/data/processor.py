import os
from pathlib import Path
from ..paths import REPOSITORY_ROOT
import glob
import pickle
import numpy as np
import uproot
#from tqdm import trange
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
    pulse_gamma: int = 700 #v4
    #pulse_gamma: int = 590 #v3
    #pulse_gamma: int = 400 #v1
    lower_pulse: int = 175
    pulse_max: int = 1200 #v4
    #pulse_max: int = 1340 #v3
    #pulse_max: int = 575 #v1
    NS_PER_ADC_SAMPLE: int = 2
    ADC_IMPEDANCE: int = 50
    ADC_TO_VOLT: float = 2.415 / (2 ** 12)
    REF_ENERGY: float = 4.42  # MeV



# Re-export CutCriteria for callers that construct the processor directly.
from .selection import (CutCriteria, Selection, BoxSelection,  # noqa: E402,F401
                        cluster_key, get_selection)


class AmBeNeutronProcessing:
    """
    Efficient analyzer for AmBe neutron efficiency with memory management and modularity.
    """
    
    def __init__(self, config: Optional[WaveformConfig] = None,
                 cuts: Optional[CutCriteria] = None,
                 selection: Optional[Selection] = None,
                 extra_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
                 cosmic_veto: bool = True):
        self.config = config or WaveformConfig()
        self.cuts = cuts or CutCriteria()
        self.cosmic_veto = bool(cosmic_veto)
        # Per-run positions that must NOT join the campaign map. See
        # get_source_location for why the special runs are kept out of it.
        self.extra_positions = dict(extra_positions or {})
        # Defaults to the box so every existing caller keeps its current behaviour
        # -- scripts/workflows/summarize_ambe_neutrons.py, scripts/presentation/boxcut_v4_special.py, scripts/workflows/reprocess_ambe_v1.py
        # and the rest construct this class without a selection and must not change.
        self.selection = selection or BoxSelection(self.cuts)
        
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
            5743: (0, 328, 0), 5778: (0, 328, 0), 5779: (0, 328, 0),

            # Background/no-source runs (no AmBe source present anywhere; detector-only trigger)
            6254: (0, 328, 0), 6256: (0, 328, 0),  # placeholder coord reused

            ##AmBe v4 Campaign 3 - March 2026
            6046: (0, 0, 0), 6062: (0, -100, 0),                       ## Port 5 data
            6056: (75, 0 , 0), 6060: (75, -100, 0), 6061: (75, 100, 0), ## Port 4 data

            ##AmBe v4 Campaign 3 continued - June, July 2026
            6165: (0, 100, 0),                                              ## Port 5 data
            6230: (0, 100, -75), 6231: (0, 0, -75),
            6232: (0, -100, -75), 6234: (0, -50, -75), 6235: (0, 50, -75),  ## Port 1 data
            6188: (0, 100, 75), 6189: (0, 100, 75),
            6237: (0, -100, 75), 6239: (0, -50, 75), 6241: (0, 0, 75), 6242: (0, 50, 75), ## Port 2 data
            6166: (0, 0, 102), 6186: (0, 100, 102), 6187: (0, -60, 102),
            6243: (0, -100, 102), 6244: (0, -50, 102), 6246: (0, 50, 102),  ## Port 3 data
            6247: (75, -50, 0), 6248: (75, 50, 0),                         ## Port 4 data
            6251: (0, -50, 0), 6252: (0, 50, 0),                          ## Port 5 data

            6273: (0, 0, 0), 6274: (0, 100, 0), 6275: (0, -100, 0),       ## Port 5 data

            ## LAPPD AmBe debug runs, Port 4, y=50 (same position as 6248)
            6249: (75, 50, 0), 6250: (75, 50, 0),
        }

    def get_source_location(self, run: int) -> Tuple[float, float, float]:
        """Return the per-config override or campaign source position in centimetres."""
        if run in self.extra_positions:
            return self.extra_positions[run]
        if run in self.source_positions:
            return self.source_positions[run]

        raise ValueError(f'RUN NUMBER {run} DOESNT HAVE A SOURCE LOCATION!')

    def ambe_single_cut(self, cpe: float, ccb: float, ct: float, cn: int,
                        chits: int, key=None) -> bool:
        """Single-neutron selection under the active neutron definition."""
        return self.selection.single(cpe, ccb, ct, cn, chits, key)

    def ambe_multiple_cut(self, cpe: float, ccb: float, ct: float, cn: int,
                          chits: int, key=None) -> bool:
        """Multiple-neutron selection under the active neutron definition."""
        return self.selection.multiple(cpe, ccb, ct, cn, chits, key)

    def cosmic_cut(self, ct: float, cpe: float) -> bool:
        """Cosmic veto. Identical for every selection -- it is a Stage-1 cut that
        sits upstream of the neutron definition, so it must not vary between the two
        analyses or their efficiencies stop sharing a denominator.

        `self.cosmic_veto=False` disables it for the WHOLE dataset (see __init__ for
        why that is a dataset-level and not a per-selection switch). Selection.cosmic
        itself is deliberately left alone, so the box/MVA objects still cannot diverge
        from each other however this is set.
        """
        if not self.cosmic_veto:
            return False
        return self.selection.cosmic(ct, cpe)

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
        if len(hits) < 1:
            return np.array([0.0, 0.0, 0.0]), 0.0

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

        UNIT CONVENTION (important):
            hitX, hitY, hitZ    — in METRES  (from PMT geometry CSV)
            sourceX, sourceY, sourceZ — in CM  (from self.source_positions dict)

        The method converts source coordinates to metres internally before
        computing distances.  Do NOT pre-convert the source coordinates; pass
        them directly from get_source_location() / the source_positions dict.

        Args:
            hitX, hitY, hitZ: Hit position arrays in metres
            hitPE: Hit photoelectron values
            hitT: Hit timing values in ns
            sourceX, sourceY, sourceZ: Source position in CM
                (e.g. Port 4 centre → sourceX=75 meaning 75 cm = 0.75 m)
            event_hits_data: Optional list of (hitX, hitY, hitZ, hitPE, hitT)
                tuples for multi-cluster calculation; same unit convention.
        Returns:
            Tuple of two space-separated strings:
                (ToF-corrected pairwise time differences,
                 raw pairwise time differences)
        """
        SoL = 0.299792458 * 3.0 / 4.0   # speed of light in water, m/ns

        # Convert source position from cm → metres to match hit coordinates
        src_x_m = float(sourceX) / 100.0
        src_y_m = float(sourceY) / 100.0
        src_z_m = float(sourceZ) / 100.0

        # If event_hits_data is provided, use multi-cluster approach
        if event_hits_data is not None:
            all_hits = []
            for cluster_hits in event_hits_data:
                cluster_hitX, cluster_hitY, cluster_hitZ, cluster_hitPE, cluster_hitT = cluster_hits
                cluster_hitX  = np.asarray(cluster_hitX,  dtype=np.float64)
                cluster_hitY  = np.asarray(cluster_hitY,  dtype=np.float64)
                cluster_hitZ  = np.asarray(cluster_hitZ,  dtype=np.float64)
                cluster_hitPE = np.asarray(cluster_hitPE, dtype=np.float64)
                cluster_hitT  = np.asarray(cluster_hitT,  dtype=np.float64)
                cluster_hit_list = list(zip(cluster_hitX, cluster_hitY, cluster_hitZ,
                                            cluster_hitPE, cluster_hitT))
                all_hits.extend(cluster_hit_list)
            hits = all_hits
        else:
            # Single cluster approach
            hitX  = np.asarray(hitX,  dtype=np.float64)
            hitY  = np.asarray(hitY,  dtype=np.float64)
            hitZ  = np.asarray(hitZ,  dtype=np.float64)
            hitPE = np.asarray(hitPE, dtype=np.float64)
            hitT  = np.asarray(hitT,  dtype=np.float64)
            hits  = list(zip(hitX, hitY, hitZ, hitPE, hitT))

        # Sort by time and calculate pairwise differences
        hits.sort(key=lambda h: h[4])
        n = len(hits)
        if n < 1:
            return "", ""

        tdiffs = []
        all_hits_delta_t_TofCorrected = []
        for i in range(n):
            x_i, y_i, z_i, pe_i, t_i = hits[i]
            # Distance from PMT (metres) to source (metres)
            dist_i  = np.sqrt((x_i - src_x_m)**2 + (y_i - src_y_m)**2 + (z_i - src_z_m)**2)
            tcorr_i = t_i - dist_i / SoL

            for j in range(i + 1, n):
                x_j, y_j, z_j, pe_j, t_j = hits[j]
                dist_j  = np.sqrt((x_j - src_x_m)**2 + (y_j - src_y_m)**2 + (z_j - src_z_m)**2)
                tcorr_j = t_j - dist_j / SoL
                delta_t_raw = t_j - t_i

                tdiffs.append(tcorr_j - tcorr_i)
                all_hits_delta_t_TofCorrected.append(delta_t_raw)

        # Return as space-separated strings for easy parsing
        return (" ".join(f"{val:.6f}" for val in tdiffs),
                " ".join(f"{val:.6f}" for val in all_hits_delta_t_TofCorrected))

    def compute_waveform_features(self, hist_values: np.ndarray,
                                  hist_edges: np.ndarray) -> Dict[str, Any]:
        """Compute integrated charge, full-record baseline and second-pulse veto."""
        cfg = self.config
        bins = hist_edges[:-1]
        baseline, sigma = norm.fit(hist_values)
        pulse_mask = (bins > cfg.pulse_start) & (bins < cfg.pulse_end)
        integral = np.sum(hist_values[pulse_mask] - baseline)
        adjusted = (cfg.NS_PER_ADC_SAMPLE / cfg.ADC_IMPEDANCE) * integral
        second_pulse = bool(np.any(
            hist_values[bins > cfg.pulse_end] > (7 + sigma + baseline)))
        return {
            "IC_adjusted": float(adjusted),
            "baseline": float(baseline),
            "baseline_sigma": float(sigma),
            "second_pulse": second_pulse,
        }

    def passes_selection(self, features: Dict[str, Any]) -> bool:
        """Apply the open IC window and second-pulse veto."""
        return bool(self.config.pulse_gamma < features["IC_adjusted"]
                    < self.config.pulse_max and not features["second_pulse"])

    def analyze_waveform(self, hist_values: np.ndarray,
                         hist_edges: np.ndarray) -> Tuple[float, float, bool, Dict[str, Any]]:
        """
        Analyze a single waveform.

        Returns:
            Tuple of (IC_adjusted, baseline, is_accepted, features)
        """
        features = self.compute_waveform_features(hist_values, hist_edges)
        return (features['IC_adjusted'], features['baseline'],
                self.passes_selection(features), features)

    def plot_waveform_sample(self, hist_edges: np.ndarray, hist_values: np.ndarray, 
                           baseline: float, timestamp: str, run: str, 
                           IC_adjusted: float, status: str, save_dir: str = str(REPOSITORY_ROOT / 'verbose')):
        """Plot a sample waveform for visualization."""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 4))
        hist_values_bs = hist_values - baseline
        # Convert ADC counts to volts
        #hist_values_volts = hist_values_bs * self.config.ADC_TO_VOLT
        plt.plot(hist_edges[:-1], hist_values_bs, label='Waveform')
        plt.axvspan(self.config.pulse_start, self.config.pulse_end, 
                   color='yellow', alpha=0.3, label='Integration Window')
        plt.title(f'{status} Waveform (timestamp: {timestamp}, Run: {run}), IC: {IC_adjusted:.2f}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{status}Waveform_{timestamp}_Run{run}.png', dpi=300)
        plt.close()

    def process_run_waveforms(self, run: str, waveform_dir: str,
                            campaign: int = 1,
                            save_waveform_samples: bool = False,
                            max_samples: int = 10,
                            feature_dump_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all waveforms for a single run with memory-efficient approach.

        Args:
            run: Run number as string
            waveform_dir: Directory containing waveform files
            campaign: Campaign number (1 or 2)
            save_waveform_samples: Whether to save sample waveforms
            max_samples: Maximum number of sample waveforms to save
            feature_dump_path: If given, write the per-waveform feature table to
                this parquet path. The table carries the tank timestamp, so the
                whole Stage-1 decision becomes re-derivable offline -- retuning
                the IC window turns into a pandas query instead of
                another multi-hour read over dCache.

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
        feature_rows = [] if feature_dump_path else None

        # Recursive glob so waveforms are found whether they sit directly in
        # <run>/ or are nested one level deeper (e.g. <run>/RWM/...root) --
        # storage layout has varied between runs/campaigns.
        waveform_files = sorted(
            glob.glob(os.path.join(waveform_dir, run, '**', '*.root'), recursive=True)
        )

        print('Loading and processing waveforms...')
        for file_idx in range(len(waveform_files)):
            waveform_filepath = waveform_files[file_idx]

            try:
                with uproot.open(waveform_filepath) as root:
                    folder_names = [name for name in root.keys() if folder_pattern.match(name)]

                    # Some files nest the RWM/BRF histograms one level deeper
                    # inside a same-named TDirectory instead of storing them
                    # at top level -- recurse into it if we didn't find any
                    # matches directly.
                    if not folder_names:
                        nested_dirs = [name for name in root.keys()
                                       if name.split(';')[0] in ('RWM', 'BRF')]
                        for nested in nested_dirs:
                            sub = root[nested]
                            folder_names.extend(
                                f"{nested}/{k}" for k in sub.keys() if folder_pattern.match(k)
                            )

                    for folder in folder_names:
                        timestamp = folder.split('_')[-1].split(';')[0]
                        if timestamp in seen_timestamps:
                            continue
                        seen_timestamps.add(timestamp)

                        try:
                            hist = root[folder]
                            hist_values = hist.values()
                            hist_edges = hist.axes[0].edges()

                            IC_adjusted, baseline, is_accepted, features = self.analyze_waveform(
                                hist_values, hist_edges
                            )
                            ic_values_batch.append(IC_adjusted)

                            if feature_rows is not None:
                                feature_rows.append({
                                    'timestamp': int(timestamp),
                                    'accepted': is_accepted,
                                    **features,
                                })

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

            except Exception as e:
                print(f"Could not open '{waveform_filepath}': {e}")
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

        if feature_rows:
            features_df = pd.DataFrame(feature_rows)
            os.makedirs(os.path.dirname(feature_dump_path) or '.', exist_ok=True)
            try:
                features_df.to_parquet(feature_dump_path, index=False)
            except (ImportError, ValueError) as e:
                # No pyarrow/fastparquet available -- a CSV keeps the retuning
                # workflow working rather than losing the features entirely.
                fallback = os.path.splitext(feature_dump_path)[0] + '.csv'
                print(f'Parquet write failed ({e}); falling back to {fallback}')
                features_df.to_csv(fallback, index=False)
                feature_dump_path = fallback
            print(f'Feature table: {len(features_df)} rows -> {feature_dump_path}')

        return {
            'source_position': (x_pos, y_pos, z_pos),
            'accepted_events': accepted_events,
            'rejected_events': rejected_events,
            'total_waveforms': total,
            'good_events': set(good_events),
            'ic_values': ic_values_batch,
            'ic_accepted': ic_accepted_batch
        }

    def _plot_run_ic_distribution(self, ic_values: List[float], run: str, pdf):
        """Plot IC distribution for a single run."""
        if not ic_values:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.hist(ic_values, bins=200, alpha=0.7, color='blue', range=(0, 2000))
        ax1.set_xlabel('IC_adjusted')
        ax1.set_ylabel('Number of Events')
        ax1.set_title(f'IC adjusted values for run: {run}')
        
        # Log scale
        ax2.hist(ic_values, bins=200, alpha=0.7, color='blue', range=(0, 2000), log=True)
        ax2.set_xlabel('IC_adjusted')
        ax2.set_ylabel('Number of Events (log scale)')
        ax2.set_title(f'IC adjusted values (log) for run: {run}')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_combined_ic_distributions(self, all_ic_values: List[float], 
                                       all_ic_accepted: List[float], runinfo: str):
        """Plot combined IC distributions for all runs."""
        os.makedirs(str(REPOSITORY_ROOT / 'verbose'), exist_ok=True)
        
        # All IC values
        plt.figure(figsize=(10, 6))
        plt.hist(all_ic_values, bins=200, alpha=0.7, color='blue', range=(0, 2000))
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events')
        plt.title('All IC adjusted Values for all runs')
        plt.tight_layout()
        plt.savefig(str(REPOSITORY_ROOT / f'verbose/IC_adjusted_AllEvents_{runinfo}.png'), dpi=300)
        plt.close()
        
        # All IC values (log scale)
        plt.figure(figsize=(10, 6))
        plt.hist(all_ic_values, bins=200, alpha=0.7, color='blue', range=(0, 2000), log=True)
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events (log scale)')
        plt.title('All IC adjusted Values for all runs (log scale)')
        plt.tight_layout()
        plt.savefig(str(REPOSITORY_ROOT / f'verbose/Log_IC_adjusted_AllEvents_{runinfo}.png'), dpi=300)
        plt.close()
        
        # Accepted IC values
        if all_ic_accepted:
            plt.figure(figsize=(10, 6))
            plt.hist(all_ic_accepted, bins=200, alpha=0.7, color='orange', range=(0, 2000))
            plt.xlabel('IC_adjusted accepted')
            plt.ylabel('Number of Events')
            plt.title('Accepted IC_adjusted Values for all runs')
            plt.tight_layout()
            plt.savefig(str(REPOSITORY_ROOT / f'verbose/IC_adjusted_AcceptedEvents_{runinfo}.png'), dpi=300)
            plt.close()

    def _plot_combined_ic_from_histograms(self, histogram_counts: np.ndarray,
                                          accepted_counts: np.ndarray,
                                          bin_edges: np.ndarray, runinfo: str):
        """Plot combined IC distributions from pre-computed histogram counts."""
        os.makedirs(str(REPOSITORY_ROOT / 'verbose'), exist_ok=True)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # All IC values
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, histogram_counts, width=bin_width, alpha=0.7, color='blue')
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events')
        plt.title('All IC adjusted Values for all runs')
        plt.tight_layout()
        plt.savefig(str(REPOSITORY_ROOT / f'verbose/IC_adjusted_AllEvents_{runinfo}.png'), dpi=300)
        plt.close()
        
        # All IC values (log scale)
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, histogram_counts, width=bin_width, alpha=0.7, color='blue', log=True)
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events (log scale)')
        plt.title('All IC adjusted Values for all runs (log scale)')
        plt.tight_layout()
        plt.savefig(str(REPOSITORY_ROOT / f'verbose/Log_IC_adjusted_AllEvents_{runinfo}.png'), dpi=300)
        plt.close()
        
        # Accepted IC values
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, accepted_counts, width=bin_width, alpha=0.7, color='orange')
        plt.xlabel('IC_adjusted accepted')
        plt.ylabel('Number of Events')
        plt.title('Accepted IC_adjusted Values for all runs')
        plt.tight_layout()
        plt.savefig(str(REPOSITORY_ROOT / f'verbose/IC_adjusted_AcceptedEvents_{runinfo}.png'), dpi=300)
        plt.close()

    def load_event_data(self, file_path: str, which_tree: int = 0,
                        event_key_branch: str = "eventTimeTank",
                        expect_run: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Load event data from ROOT file with memory-efficient approach.

        Args:
            file_path: Path to ROOT file
            which_tree: Tree type (0 or other)
            event_key_branch: Which branch supplies the per-event identity. Data keeps
                the default `eventTimeTank`, a globally unique ns timestamp.

                MC HAS NO SUCH TIMESTAMP: `eventTimeTank` is 0 in every event of the
                AmBe MC production, so `eventNumber` (0-999, unique within a file) is
                substituted here. This is the ONLY place the substitution happens, on
                purpose: `process_events_efficient` uses ETT in six places (the Stage-1
                gate, the MVA cluster_key, `event_clusters`, `repeated_event_id`, the
                event-index row, and `_append_cluster_data`) and patching them one by
                one is six chances to miss one. With a raw ETT of 0, `event_clusters`
                would feed one event's hits into another's multi-cluster ToF path and
                `repeated_event_id` would count multiple-neutron events exactly ONCE
                for the whole dataset.

                The substitution is legitimate only under the invariant that
                (run, eventNumber) is globally unique. Guards 1-3 below establish the
                per-file half; the caller (run_mc) establishes that run numbers are
                unique across files.
            expect_run: If given, assert the tree's own `runNumber` matches. Catches a
                renamed or misfiled ROOT file before it is silently attributed to the
                wrong source position.

        Returns:
            Dictionary containing event data arrays
        """
        print(f'Loading AmBe event data from {file_path}...')

        with uproot.open(file_path) as file_1:
            Event = file_1["Event"]
            
            # Load common branches
            data = {
                "eventNumber": Event["eventNumber"].array(library="np"),
                "eventTimeTank": Event["eventTimeTank"].array(library="np"),
                "clusterTime": Event["clusterTime"].array(library="np"),
                "clusterPE": Event["clusterPE"].array(library="np"),
                "clusterChargeBalance": Event["clusterChargeBalance"].array(library="np"),
                "clusterHits": Event["clusterHits"].array(library="np"),
                "hitX": Event["Cluster_HitX"].array(library="np"),
                "hitY": Event["Cluster_HitY"].array(library="np"),
                "hitZ": Event["Cluster_HitZ"].array(library="np")
            }
            
            # Tree-dependent branches
            if which_tree == 0:
                data.update({
                    "clusterNumber": Event["clusterNumber"].array(library="np"),
                    "hitT": Event["hitT"].array(library="np"),
                    "hitPE": Event["hitPE"].array(library="np"),
                    "hitDetID": Event["hitDetID"].array(library="np"),
                })
            else:
                data.update({
                    "clusterNumber": Event["numberOfClusters"].array(library="np"),
                    "hitT": Event["Cluster_HitT"].array(library="np"),
                    "hitPE": Event["Cluster_HitPE"].array(library="np"),
                    "hitDetID": Event["Cluster_HitDetID"].array(library="np"),
                })

            if expect_run is not None:
                got = int(Event["runNumber"].array(library="np", entry_stop=1)[0])
                if got != int(expect_run):
                    raise SystemExit(
                        f"[load_event_data] {file_path}: tree runNumber is {got} but the "
                        f"manifest says {expect_run}. A misfiled ROOT file would be "
                        f"attributed to the wrong source position -- refusing to guess.")

        if event_key_branch != "eventTimeTank":
            ett_raw = data["eventTimeTank"]
            key = data[event_key_branch]
            # (1) Never overwrite a real key.
            if not np.all(ett_raw == 0):
                raise SystemExit(
                    f"[load_event_data] {file_path}: eventTimeTank is not identically 0, "
                    f"so substituting {event_key_branch!r} would DISCARD a real event key. "
                    f"This file is not the AmBe MC production.")
            # (2) Uniqueness within the file -- otherwise (run, key) is not unique either.
            if len(np.unique(key)) != len(key):
                raise SystemExit(
                    f"[load_event_data] {file_path}: {event_key_branch!r} is not unique "
                    f"within the file ({len(np.unique(key))} distinct of {len(key)}); "
                    f"(run, {event_key_branch}) would collide.")
            # (3) cluster_key() does int(...); a negative key silently collides with
            #     nothing in particular and is impossible to notice downstream.
            if key.min() < 0:
                raise SystemExit(
                    f"[load_event_data] {file_path}: {event_key_branch!r} has negative "
                    f"values (min {key.min()}); the cluster key must be non-negative.")
            data["eventTimeTank"] = key.astype("int64")

        return data

    def select_event_clusters(self, times, pe, cb, hits, count, run=None, timestamp=None):
        """Return veto, single-category and multiple-category cluster indices."""
        cosmic = [k for k in range(len(times)) if self.cosmic_cut(times[k], pe[k])]
        if cosmic:
            return cosmic, [], []
        single, multiple = [], []
        for k in range(len(times)):
            key = cluster_key(run, timestamp, times[k]) if run is not None else None
            if pe[k] == float('-inf'):
                continue
            if self.ambe_single_cut(pe[k], cb[k], times[k], count, hits[k], key):
                single.append(k)
            if self.ambe_multiple_cut(pe[k], cb[k], times[k], count, hits[k], key):
                multiple.append(k)
        return cosmic, single, multiple

    def process_events_efficient(self, event_data: Dict[str, np.ndarray],
                                good_events: Set[int], x_pos: float, y_pos: float, z_pos: float,
                                efficiency_data: Optional[Dict] = None,
                                run: Optional[int] = None,
                                event_rows: Optional[List[Dict]] = None) -> Tuple[List, Dict[str, int]]:
        """
        Process events with memory-efficient vectorized operations where possible.

        Args:
            event_data: Dictionary containing event data
            good_events: Set of good event timestamps, or None to admit every event.
                None disables the Stage-1 waveform gate entirely and is passed ONLY by
                the MC entry point (run_mc): MC has no waveforms, so there is no IC
                window to apply and every cluster is treated as having passed the good-
                waveform selection. `run()` never passes None -- a data deck silently
                ungated would report an efficiency against the wrong denominator.
            x_pos, y_pos, z_pos: Source position coordinates
            efficiency_data: Optional efficiency tracking dictionary
            run: Run number. Required by the MVA selection, which identifies a
                cluster by (run, eventTimeTank, clusterTime) to look up its score.
                Optional so existing box-only callers are unaffected.
            event_rows: Optional list to receive one dict per Stage-1-admitted
                event, recording the cosmic-veto verdict and the accepted-cluster
                counts. This is the per-event index that lets an offline consumer
                re-tally the efficiency for a NARROWER IC window without
                reprocessing: cosmic_events is the one ingredient the published
                CSVs cannot supply, because PromptAmBeNeutronCandidates carries
                no event key and so a cosmic event cannot be assigned an IC
                value. None (the default) appends nothing.

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
            'cluster_direction': [], 'neutron_tof_correction': [], 'all_hits_delta_t_TofCorrected': []
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
        
        for i in range(len(EN)):
            if good_events is not None and ETT[i] not in good_events:
                continue

            stats['total_events'] += 1
            if efficiency_data is not None:
                efficiency_data[key][0] += 1

            # Process clusters for this event
            event_has_cosmic = False
            event_neutron_candidates = []
            event_multiple_candidates = []

            cosmic_ks, event_neutron_candidates, event_multiple_candidates = self.select_event_clusters(
                CT[i], CPE[i], CCB[i], CH[i], CN[i], run=run, timestamp=ETT[i])
            if cosmic_ks:
                event_has_cosmic = True
                stats['cosmic_events'] += 1
                if efficiency_data is not None:
                    efficiency_data[key][1] += 1
                for k in cosmic_ks:
                    processed_data['prompt_cluster_time'].append(CT[i][k])
                    processed_data['prompt_cluster_charge'].append(CPE[i][k])
                    processed_data['prompt_cluster_QB'].append(CCB[i][k])

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
                if ETT[i] not in repeated_event_id:
                    stats['multiple_neutron_cand_count'] += 1
                    repeated_event_id.add(ETT[i])
                    if efficiency_data is not None:
                        efficiency_data[key][3] += 1

            if event_rows is not None:
                event_rows.append({
                    'run': int(run) if run is not None else -1,
                    'eventID': int(EN[i]),
                    'eventTankTime': int(ETT[i]),
                    'numberOfClusters': int(CN[i]),
                    'is_cosmic': bool(event_has_cosmic),
                    'n_accepted_single': len(event_neutron_candidates),
                    'n_accepted_multiple': len(event_multiple_candidates),
                    'sourceX': x_pos,
                    'sourceY': y_pos,
                    'sourceZ': z_pos,
                })


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
        hit_T_str = '[' + ', '.join([str(t) for t in hT[i][k]]) + ']'
        processed_data['hit_times'].append(hit_T_str)
        hit_x_str = '[' + ', '.join([str(x) for x in hX[i][k]]) + ']'
        hit_y_str = '[' + ', '.join([str(y) for y in hY[i][k]]) + ']'
        hit_z_str = '[' + ', '.join([str(z) for z in hZ[i][k]]) + ']'
        processed_data['hit_x'].append(hit_x_str)
        processed_data['hit_y'].append(hit_y_str)
        processed_data['hit_z'].append(hit_z_str)
        hit_PE_str = '[' + ', '.join([str(pe) for pe in hPE[i][k]]) + ']'
        processed_data['hit_charges'].append(hit_PE_str)
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
                                      plot_ic_distributions: bool = True,
                                      dump_features: bool = True,
                                      dump_good_events: bool = False,
                                      dump_event_index: bool = False,
                                      run_manifest: Optional[List[Dict[str, Any]]] = None,
                                      event_key_branch: str = "eventTimeTank") -> Dict[str, Any]:
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
            dump_features: Whether to write the per-waveform feature table for
                each run to TriggerSummary/WaveformFeatures_<runinfo>_<run>.parquet
            dump_good_events: Whether to write the Stage-1 IC-gated eventTimeTank set
                for each run to TriggerSummary/GoodEvents_<runinfo>_<run>.pkl.
                Off by default so existing callers write nothing new. `ambe leak
                features` consumes these: without them it has to repeat Stage 1 and
                re-read the whole waveform tree, which for a 3-run campaign is ~10 GB
                off /pnfs for a set of integers this loop already has in hand.
            dump_event_index: Whether to write the per-event Stage-2 index for each
                run to TriggerSummary/EventIndex_<runinfo>_<run>.parquet. Off by
                default so existing callers write nothing new. `ambe data icscan`
                consumes these: joined to WaveformFeatures on
                eventTankTime == timestamp, they carry the cosmic verdict and the
                accepted-cluster counts per event, which is what makes the whole
                IC-window scan a pandas groupby instead of one /pnfs pass per
                window. total_events and unique_neutron_triggers are already
                recoverable from the feature parquet and the candidate CSVs;
                cosmic_events is not, and ambe_triggers needs it.
            run_manifest: MC ONLY. A list of {"run", "path", "position"} dicts. When
                given, the directory scan is replaced by this list and Stage 1 is
                skipped entirely: there are no waveforms, so `good_events` is None and
                every event is admitted. `run()` never passes it, so the data path is
                bit-for-bit unchanged.
            event_key_branch: forwarded to load_event_data. See there for why MC needs
                eventNumber and what would break if the substitution were made anywhere
                else.

        Returns:
            Dictionary with complete analysis results
        """
        mc_mode = run_manifest is not None

        print("="*80)
        print("STARTING COMPLETE AmBe NEUTRON ANALYSIS PIPELINE")
        print("="*80)
        if mc_mode:
            # Nothing about the IC window applies -- saying so beats printing a window
            # that was never consulted.
            print("Stage 1        : NONE (MC: no waveforms; every event admitted)")
            print(f"Event key      : {event_key_branch}")
            print(f"Cosmic veto    : {'ON' if self.cosmic_veto else 'OFF'}")
        else:
            print(f"IC window      : {self.config.pulse_gamma} < IC_adjusted < {self.config.pulse_max}")
        print(f"Feature dump   : {'on' if dump_features else 'off'}")

        run_numbers = []
        file_names = []

        if mc_mode:
            # No waveform stage, so nothing to plot or dump from it.
            plot_ic_distributions = False
            dump_features = False
            dump_good_events = False
            for entry in run_manifest:
                run_numbers.append(str(entry["run"]))
                file_names.append(str(entry["path"]))
        else:
            for file_name in os.listdir(data_directory):
                match = file_pattern.match(file_name)
                if match:
                    run_number = int(match.group(1))
                    run_numbers.append(str(run_number))
                    file_names.append(os.path.join(data_directory, file_name))

        if not run_numbers:
            raise ValueError(f"No files found matching pattern in {data_directory}")
        
        print(f"Found {len(run_numbers)} runs to process")

        # Initialize tracking structures
        waveform_summary_list = []
        efficiency_data = defaultdict(lambda: [0, 0, 0, 0]) 

        # For IC distributions (if plotting) - use histograms for memory efficiency
        if plot_ic_distributions:
            ic_hist_bins = np.linspace(0, 1400, 201)
            all_ic_histogram = np.zeros(200)  # Accumulate histogram counts
            all_ic_accepted_histogram = np.zeros(200)
            (REPOSITORY_ROOT / "verbose").mkdir(parents=True, exist_ok=True)
            pdf_file = PdfPages(str(REPOSITORY_ROOT / f'verbose/AllRuns_IC_adjusted_{runinfo}.pdf'))

        # Create output directories
        os.makedirs(str(REPOSITORY_ROOT / 'TriggerSummary'), exist_ok=True)
        os.makedirs(str(REPOSITORY_ROOT / 'EventAmBeNeutronCandidatesData'), exist_ok=True)
        os.makedirs(str(REPOSITORY_ROOT / 'verbose'), exist_ok=True)
        # Step 1: Waveform Analysis for All Runs and Process Tank data
        print("\n" + "="*60)
        print("STEP 1: Waveform Analysis for All Runs" if not mc_mode
              else "STEP 1: SKIPPED (MC has no waveforms)")
        print("="*60)

        for c1, run in enumerate(run_numbers):

            if mc_mode:
                # No waveform stage at all. good_events=None makes the Stage-1 gate in
                # process_events_efficient a no-op, which is the whole "every cluster is
                # assumed to pass the good-waveform selection" premise of this deck.
                good_events = None
                x_pos, y_pos, z_pos = run_manifest[c1]["position"]
                run_result = None
            else:
                run_result = self.process_run_waveforms(
                    run, waveform_dir, campaign,
                    save_waveform_samples=save_waveform_samples,
                    feature_dump_path=(
                        str(REPOSITORY_ROOT / f'TriggerSummary/WaveformFeatures_{runinfo}_{run}.parquet')
                        if dump_features else None
                    ),
                )

                # Get waveform results for this run
                good_events = run_result["good_events"]
                x_pos, y_pos, z_pos = run_result["source_position"]
            file_path = file_names[c1]

            if not mc_mode:
                waveform_summary_list.append({
                    "run": int(run),
                    "sourceX": x_pos,
                    "sourceY": y_pos,
                    "sourceZ": z_pos,
                    "accepted_waveforms": run_result["accepted_events"],
                    "rejected_waveforms": run_result["rejected_events"],
                    "total_waveforms": run_result["total_waveforms"]
                })

            if plot_ic_distributions:
                if run_result["ic_values"]:
                    self._plot_run_ic_distribution(run_result["ic_values"], run, pdf_file)
                
                # Accumulate histogram counts (memory-efficient)
                ic_hist, _ = np.histogram(run_result['ic_values'], bins=ic_hist_bins)
                all_ic_histogram += ic_hist
                
                ic_accepted_hist, _ = np.histogram(run_result['ic_accepted'], bins=ic_hist_bins)
                all_ic_accepted_histogram += ic_accepted_hist

            
            print(f"Source position: ({x_pos}, {y_pos}, {z_pos})")
            if good_events is None:
                print("Good events: ALL (no waveform gate)")
            else:
                print(f"Good events from waveform analysis: {len(good_events)}")

            if dump_good_events:
                ge_path = str(REPOSITORY_ROOT / f'TriggerSummary/GoodEvents_{runinfo}_{run}.pkl')
                os.makedirs(os.path.dirname(ge_path), exist_ok=True)
                with open(ge_path, 'wb') as _fh:
                    pickle.dump(set(good_events), _fh)
                print(f"Good events dumped -> {ge_path}")
            
            # Load event data for this specific run
            event_data = self.load_event_data(
                file_path, which_tree, event_key_branch=event_key_branch,
                expect_run=int(run) if mc_mode else None)

            if mc_mode:
                # Synthetic Stage-1 row so AmBeWaveformResults_<tag>.csv keeps its
                # schema for downstream summaries. total == accepted by construction:
                # there is no waveform to reject. The consumer must NOT read this as a
                # measured waveform acceptance.
                n_ev = len(event_data["eventNumber"])
                waveform_summary_list.append({
                    "run": int(run), "sourceX": x_pos, "sourceY": y_pos, "sourceZ": z_pos,
                    "accepted_waveforms": n_ev, "rejected_waveforms": 0,
                    "total_waveforms": n_ev,
                })

            # Process events for this run
            event_rows = [] if dump_event_index else None
            processed_data, event_stats = self.process_events_efficient(
                event_data, good_events, x_pos, y_pos, z_pos, efficiency_data,
                run=int(run), event_rows=event_rows
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
                #"hitQ": processed_data['hit_charges'],
                "hitPE": processed_data['hit_charges'],
                "hitID": processed_data['hit_ids'],
                "hit_delta_t": processed_data['hit_delta_t'],
                "sourceX": processed_data['source_position'][0],
                "sourceY": processed_data['source_position'][1],
                "sourceZ": processed_data['source_position'][2],
                "eventID": processed_data['event_ids'],
                "eventTankTime": processed_data['event_tank_time'],
                # Constant per file. Downstream multiplicity/delta-t code must group on
                # (run, eventTankTime), never eventTankTime alone: in MC the event key is
                # eventNumber (0-999) and every run would otherwise merge into 1000 groups.
                # For data this is a strict refinement that changes nothing.
                "run": run_int,
                "clusterDirection": processed_data['cluster_direction'],
                #'neutronTofCorrection': processed_data['neutron_tof_correction'],
                #'allHitsDeltaT_TofCorrected': processed_data['all_hits_delta_t_TofCorrected']

            })
            
            print("Sample of neutron candidates data:")
            print(df.head())
            
            # Save neutron candidates data
            output_file = str(REPOSITORY_ROOT / f'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_{run_int}.csv')
            df.to_csv(output_file, float_format="%.6f", index=False)
            print(f"✓ Saved neutron candidates to {output_file}")
            
            # Prompt neutrons DataFrame
            prompt_df = pd.DataFrame({
                "prompt_clusterTime": processed_data['prompt_cluster_time'],
                "prompt_clusterPE": processed_data['prompt_cluster_charge'],
                "prompt_clusterChargeBalance": processed_data['prompt_cluster_QB'],
            })
            
            # Save prompt neutrons data
            prompt_file = str(REPOSITORY_ROOT / f'EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_{run_int}.csv')
            prompt_df.to_csv(prompt_file, index=False)
            print(f"✓ Saved prompt neutron candidates to {prompt_file}")

            if event_rows is not None:
                index_df = pd.DataFrame(event_rows)
                index_df['n_accepted_clusters'] = (index_df['n_accepted_single']
                                                   + index_df['n_accepted_multiple'])
                index_path = str(REPOSITORY_ROOT / f'TriggerSummary/EventIndex_{runinfo}_{run_int}.parquet')
                os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
                try:
                    index_df.to_parquet(index_path, index=False)
                except (ImportError, ValueError) as e:
                    # Same fallback as the feature dump: no pyarrow/fastparquet
                    # should not cost us the index entirely.
                    fallback = os.path.splitext(index_path)[0] + '.csv'
                    print(f'Parquet write failed ({e}); falling back to {fallback}')
                    index_df.to_csv(fallback, index=False)
                    index_path = fallback
                print(f"✓ Saved event index ({len(index_df)} events, "
                      f"{int(index_df['is_cosmic'].sum())} cosmic, "
                      f"{int((~index_df['is_cosmic'] & (index_df['n_accepted_clusters'] > 0)).sum())} "
                      f"neutron-bearing) to {index_path}")
                del index_df

            # Clear data to free memory
            del processed_data, df, prompt_df, run_result, event_data, good_events, event_rows
            gc.collect()
        if plot_ic_distributions:
            pdf_file.close()
        
        waveform_df = pd.DataFrame(waveform_summary_list)
        waveform_df = waveform_df.groupby(['sourceX', 'sourceY', 'sourceZ'], as_index=False).sum()
        waveform_df.to_csv(str(REPOSITORY_ROOT / f'TriggerSummary/AmBeWaveformResults_{runinfo}.csv'), index=False)

        if plot_ic_distributions:
            self._plot_combined_ic_from_histograms(
                all_ic_histogram, all_ic_accepted_histogram, ic_hist_bins, runinfo
            )
        
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
        
        efficiency_file = str(REPOSITORY_ROOT / f'TriggerSummary/AmBeTriggerSummary_{runinfo}.csv')
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
        if dump_event_index:
            print(f"  - TriggerSummary/EventIndex_{runinfo}_<run>.parquet (for each run)")


        return {
            'run_numbers': run_numbers,
            'file_names': file_names,
            'efficiency_data': dict(efficiency_data),
            'output_files': {
                'waveform_summary': str(REPOSITORY_ROOT / f'TriggerSummary/AmBeWaveformResults_{runinfo}.csv'),
                'efficiency_summary': efficiency_file,
                'neutron_candidates_pattern': str(REPOSITORY_ROOT / f'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_*.csv'),
                'prompt_candidates_pattern': str(REPOSITORY_ROOT / f'EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_*.csv'),
                'event_index_pattern': (str(REPOSITORY_ROOT / f'TriggerSummary/EventIndex_{runinfo}_*.parquet')
                                        if dump_event_index else None)
            }
        }


# Output tags used by historical results require an explicit CLI override.
PROTECTED_TAGS = {
    "AmBe2.0v4_gated", "AmBe2.0v4_ext", "AmBe2.0v4_fprompt",
    "AmBe2.0v1", "AmBe2.0v3", "baseline", "fprompt2d", "recovered",
}


def build_selection(ctx, name: str):
    """Build box or MVA selection using configured thresholds and score files."""
    cuts = CutCriteria(
        pe_max=float(ctx.cuts.get("pe_max", 100)),
        ccb_max=float(ctx.cuts.get("charge_balance_max", 0.45)),
        ct_min=float(ctx.cuts.get("time_min_ns", 2000)),
        ct_max=float(ctx.cuts.get("time_max_ns", 67000)),
        chits_min=int(ctx.cuts.get("cluster_hits_min", 5)),
    )
    if cuts.ct_max <= cuts.ct_min:
        raise SystemExit(
            f"[processor] cuts.time_max_ns ({cuts.ct_max:g}) must exceed "
            f"cuts.time_min_ns ({cuts.ct_min:g}); the delayed window is empty.")
    if name == "box":
        return get_selection("box", cuts=cuts), cuts
    mva = dict(ctx.extra.get("mva") or {})
    if not mva:
        raise SystemExit(
            "--selection mva needs an `mva:` block in the config declaring "
            "`scored` (one or more scored parquets), `threshold` and optionally "
            "`score_col` / `point`.")
    scored = mva.get("scored")
    if isinstance(scored, str):
        scored = [scored]
    point = mva.get("point", "eff80")
    score_col, threshold = mva.get("score_col", "gbt_score"), mva.get("threshold")
    if score_col == "auto" or threshold == "auto":
        # Column and cut both come from the frozen artifact: its primary model (the
        # NN for artifacts frozen since 2026-09-23) at the stored working point. The
        # pair is never mixed, so a cut can't be applied to the wrong model's score.
        if not mva.get("model"):
            raise SystemExit("mva: score_col/threshold 'auto' needs `model: <frozen.pkl>`")
        from ..mc.mva_analysis import frozen_working_point
        score_col, threshold = frozen_working_point(mva["model"], point)
        print(f"[processor] mva: {score_col} > {threshold:.6f} from "
              f"{Path(mva['model']).name} ({point})")
    sel = get_selection("mva", cuts=cuts, scored=scored,
                        threshold=float(threshold),
                        score_col=score_col,
                        point=point)
    return sel, cuts


def run(ctx, argv=None):
    """Run the waveform gate and neutron selection from a Stage-1 configuration."""
    import argparse

    p = argparse.ArgumentParser(prog="ambe data process")
    # No default: the two selections write DIFFERENT candidate sets under different
    # tags. Defaulting to either one is how a deck ends up labelled as the other.
    p.add_argument("--selection", required=True, choices=["box", "mva"],
                   help="neutron cluster definition to apply")
    p.add_argument("--tag", default=None,
                   help="output tag (runinfo). Defaults to stage1.tag in the config.")
    p.add_argument("--allow-protected-tag", action="store_true",
                   help="permit writing to a tag that backs a published number")
    a = p.parse_args(list(argv) if argv else [])

    st = dict(ctx.extra.get("stage1") or {})
    if not st:
        raise SystemExit("config needs a `stage1:` block with data_directory, "
                         "waveform_dir and tag")

    tag = a.tag or st.get("tag")
    if not tag:
        raise SystemExit("no output tag: pass --tag or set stage1.tag")
    if tag in PROTECTED_TAGS and not a.allow_protected_tag:
        raise SystemExit(
            f"refusing to write tag {tag!r}: TriggerSummary/AmBeWaveformResults_{tag}"
            f".csv and AmBeTriggerSummary_{tag}.csv back published numbers and are "
            f"written with a plain to_csv (no append, no backup). Choose a new tag, "
            f"or pass --allow-protected-tag if you really mean to re-derive them.")

    selection, cuts = build_selection(ctx, a.selection)
    print(f"[data.process] tag={tag}  selection={selection.name}  "
          f"({selection.label})")

    if st.get("selection_mode", "ic") != "ic":
        raise ValueError("Stage 1 supports only the IC window and second-pulse veto")
    wf = WaveformConfig(
        pulse_gamma=int(st.get("ic_min", WaveformConfig.pulse_gamma)),
        pulse_max=int(st.get("ic_max", WaveformConfig.pulse_max)),
    )
    print(f"[data.process] Stage-1 gate: {wf.pulse_gamma} < IC_adjusted < "
          f"{wf.pulse_max}, plus the second-pulse veto")

    # Special-run positions, supplied per config so they never enter the campaign map.
    extra = {int(k): tuple(float(x) for x in v)
             for k, v in (st.get("extra_source_positions") or {}).items()}
    if extra:
        print(f"[data.process] extra source positions (not in the campaign map): "
              f"{ {k: v for k, v in extra.items()} }")

    if "legacy_ordering" in st:
        raise SystemExit("[data.process] stage1.legacy_ordering was removed: the cosmic "
                         "veto is decided over every cluster before any candidate is "
                         "banked. Delete the key from the config.")

    processor = AmBeNeutronProcessing(config=wf, cuts=cuts, selection=selection,
                                      extra_positions=extra)
    result = processor.run_complete_analysis_pipeline(
        data_directory=st["data_directory"],
        waveform_dir=st.get("waveform_dir", st["data_directory"]),
        file_pattern=re.compile(st.get("file_pattern",
                                       r"BeamCluster_(\d+)\.root")),
        campaign=int(st.get("campaign", 2)),
        runinfo=tag,
        which_tree=int(st.get("which_tree", 1)),
        save_waveform_samples=bool(st.get("save_waveform_samples", False)),
        plot_ic_distributions=bool(st.get("plot_ic", True)),
        dump_good_events=bool(st.get("dump_good_events", False)),
        dump_event_index=bool(st.get("dump_event_index", False)),
    )

    # For the MVA selection, say how much of THIS Stage-1 sample the scoring stage
    # actually covered. Without this the run looks successful while quietly counting
    # every unscored cluster as a rejection -- see MvaSelection.accepts.
    if hasattr(selection, "report_coverage"):
        print()
        selection.report_coverage()
    return result


def cli(ctx, argv=None):
    run(ctx, argv)



# Port -> (x, z) in cm. The label's z token is the VERTICAL coordinate and maps to y,
# the same convention as the v5sandi runs 6273/6274/6275 (see source_positions above).
MC_PORT_XZ = {1: (0.0, -75.0), 2: (0.0, 75.0), 3: (0.0, 102.0),
              4: (75.0, 0.0), 5: (0.0, 0.0)}
MC_Z_TOKEN = {"z0": 0.0, "z50": 50.0, "z100": 100.0,
              "zminus50": -50.0, "zminus100": -100.0}
MC_LABEL_RE = re.compile(r"^port(?P<port>\d+)_(?P<z>z(?:minus)?\d+)QE")


def mc_label_position(label: str) -> Tuple[float, float, float]:
    """(x, y, z) cm for an AmBe MC label such as 'port3_zminus50QE_1.50'."""
    m = MC_LABEL_RE.match(label)
    if not m:
        raise SystemExit(f"[data.processmc] cannot parse MC label {label!r}")
    port = int(m.group("port"))
    if port not in MC_PORT_XZ:
        raise SystemExit(f"[data.processmc] unknown port {port} in label {label!r}")
    tok = m.group("z")
    if tok not in MC_Z_TOKEN:
        raise SystemExit(f"[data.processmc] unknown z token {tok!r} in label {label!r}")
    x, z = MC_PORT_XZ[port]
    return (x, MC_Z_TOKEN[tok], z)


def build_mc_manifest(dataset_root: str, label_map_path: str,
                      limit_files: Optional[int] = None) -> List[Dict[str, Any]]:
    """One entry per MC ROOT file: {run, path, label, position}.

    The run number encodes the position: run = label_index*1000 + file_index, with
    label_index taken from label_map.txt written by the grid workflow. We parse that
    file rather than re-encoding the LABELS list from
    grid_wcsim/AmBe_MC_toolchain/main.py, so the two cannot drift apart. Every entry's
    run number is cross-checked against the tree's own runNumber at load time.
    """
    from ..io import resolve_inputs, filter_files_with_tree

    label_index: Dict[str, int] = {}
    with open(label_map_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx, lab = line.split()
            label_index[lab] = int(idx)
    if not label_index:
        raise SystemExit(f"[data.processmc] no labels parsed from {label_map_path}")

    files = resolve_inputs([os.path.join(dataset_root, "*",
                                         "ANNIEEvent_MC_AmBe_*.root")])
    if not files:
        raise SystemExit(f"[data.processmc] no MC ROOT files under {dataset_root}")
    # An interrupted grid job leaves a zero-key file behind; without this the whole
    # multi-hour run dies on a bare KeyError halfway through.
    files, _bad = filter_files_with_tree(files, "Event")
    if limit_files:
        files = files[:limit_files]

    manifest: List[Dict[str, Any]] = []
    for path in files:
        path = str(path)
        label = os.path.basename(os.path.dirname(path))
        if label not in label_index:
            raise SystemExit(f"[data.processmc] {path}: label {label!r} is not in "
                             f"{label_map_path}")
        m = re.search(r"_(\d+)_(\d+)\.root$", os.path.basename(path))
        if not m:
            raise SystemExit(f"[data.processmc] cannot parse file index from {path}")
        file_index = int(m.group(1)) // 1000
        manifest.append({
            "run": label_index[label] * 1000 + file_index,
            "path": path,
            "label": label,
            "position": mc_label_position(label),
        })

    runs = [e["run"] for e in manifest]
    if len(set(runs)) != len(runs):
        dupes = sorted({r for r in runs if runs.count(r) > 1})
        raise SystemExit(
            f"[data.processmc] run numbers are not unique across files (e.g. {dupes[:5]}). "
            f"(run, eventNumber) is then not a unique cluster key and the MVA scores "
            f"would be joined onto the wrong clusters.")
    return manifest


def run_mc(ctx, argv=None):
    """Stage 2 over the AmBe MC production: no waveform gate, box or MVA selection."""
    import argparse

    p = argparse.ArgumentParser(prog="ambe data processmc")
    p.add_argument("--selection", required=True, choices=["box", "mva"],
                   help="neutron cluster definition to apply")
    # No default, and printed in the log: the two settings produce different event
    # denominators and different candidate sets under the same downstream plot chain.
    p.add_argument("--cosmic-veto", required=True, choices=["on", "off"],
                   help="apply the <2us / >100PE prompt veto (on = as the data decks do)")
    p.add_argument("--tag", default=None,
                   help="output tag (runinfo). Defaults to stage1_mc.tag in the config.")
    p.add_argument("--limit-files", type=int, default=None,
                   help="process only the first N files (smoke test)")
    p.add_argument("--allow-protected-tag", action="store_true")
    a = p.parse_args(list(argv) if argv else [])

    st = dict(ctx.extra.get("stage1_mc") or {})
    if not st:
        raise SystemExit(
            "config needs a `stage1_mc:` block with dataset_root, label_map and tag. "
            "The key is deliberately NOT `stage1:` so a data config can never be run "
            "through processmc (which would silently drop its IC gate) and an MC config "
            "can never be run through process (which would look for waveforms).")

    tag = a.tag or st.get("tag")
    if not tag:
        raise SystemExit("no output tag: pass --tag or set stage1_mc.tag")
    if tag in PROTECTED_TAGS and not a.allow_protected_tag:
        raise SystemExit(f"refusing to write protected tag {tag!r}")

    veto_on = (a.cosmic_veto == "on")
    selection, cuts = build_selection(ctx, a.selection)

    # The scored parquet must belong to the SAME veto setting: the veto changes which
    # events contribute clusters, so crossing them would not crash, it would quietly
    # depress MVA coverage and look like a scoring failure.
    if a.selection == "mva":
        want = "vetoON" if veto_on else "vetoOFF"
        other = "vetoOFF" if veto_on else "vetoON"
        for pth in (ctx.extra.get("mva") or {}).get("scored") or []:
            base = os.path.basename(str(pth))
            if other in base or want not in base:
                raise SystemExit(
                    f"[data.processmc] --cosmic-veto {a.cosmic_veto} but the scored "
                    f"parquet is {base!r}. The MVA scores must come from the extraction "
                    f"run with the SAME veto setting, or coverage silently collapses.")

    print(f"[data.processmc] tag={tag}  selection={selection.name} ({selection.label})")
    print(f"[data.processmc] cosmic veto: {'ON' if veto_on else 'OFF'}")
    print("[data.processmc] Stage 1: NONE. MC has no waveforms, so every cluster is "
          "treated as having passed the good-waveform selection.")

    manifest = build_mc_manifest(st["dataset_root"], st["label_map"],
                                 limit_files=a.limit_files)
    if a.limit_files:
        print(f"[data.processmc] *** PARTIAL SAMPLE: {len(manifest)} of the full "
              f"production. Nothing from this run is quotable. ***")

    # Validate every position up front. get_source_location raises per run, mid-loop,
    # after earlier runs have already written their CSVs.
    extra = {e["run"]: e["position"] for e in manifest}
    positions = sorted(set(extra.values()))
    print(f"[data.processmc] {len(manifest)} files, {len(positions)} source positions")

    processor = AmBeNeutronProcessing(config=WaveformConfig(), cuts=cuts,
                                      selection=selection, extra_positions=extra,
                                      cosmic_veto=veto_on)
    result = processor.run_complete_analysis_pipeline(
        data_directory=st["dataset_root"],
        waveform_dir=st["dataset_root"],          # unused: Stage 1 never runs
        file_pattern=re.compile(r"(?!)"),          # unused: the manifest supplies files
        campaign=1,
        runinfo=tag,
        which_tree=int(st.get("which_tree", 1)),
        plot_ic_distributions=False,
        dump_features=False,
        dump_event_index=bool(st.get("dump_event_index", False)),
        run_manifest=manifest,
        event_key_branch=str(st.get("event_key_branch", "eventNumber")),
    )

    if hasattr(selection, "report_coverage"):
        print()
        selection.report_coverage()
    return result


def cli_mc(ctx, argv=None):
    run_mc(ctx, argv)
