import pandas as pd
import ast  
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import glob
import re

path = './'  

traditional_files = glob.glob(os.path.join(path, 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_test_*.csv'))
optics_files= glob.glob(os.path.join(path, 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_test_*_OPTICS.csv'))

def run_number(filename):
    match = re.search(r'EventAmBeNeutronCandidates_test_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return -1

optics_runs = {run_number(f): f for f in optics_files}
traditional_runs = {run_number(f): f for f in traditional_files}

run_sets = sorted(set(optics_runs.keys()).intersection(set(traditional_runs.keys())))


def analyze_row_labels(row):
    try:
        labels = ast.literal_eval(row['optics_labels'])
    except (ValueError, SyntaxError):
        labels = []
    if not labels:
        return pd.Series(['No Hits', 0, 0, 0])


    total_hits = len(labels)
    noise_hits = labels.count(-1)
    signal_hits = total_hits - noise_hits
    unique_clusters = set([l for l in labels if l != -1])
    num_unique_clusters = len(unique_clusters)
    category = 99 # Default/Other
    
    if signal_hits == 0 and noise_hits > 0:
        category = 0  # Pure Noise
    elif signal_hits > 0 and noise_hits == 0:
        if num_unique_clusters == 1:
            category = 1  # Pure Signal (Single)
        else:
            category = 2  # Pure Signal (Multi-Cluster)
    elif signal_hits > 0 and noise_hits > 0:
        category = -1 # Mixed (Signal + Noise)
    
    return pd.Series([category, total_hits, noise_hits, signal_hits])

with PdfPages('output.pdf') as pdf:
    for run in run_sets:
        opticsfile = optics_runs[run]
        traditional_file = traditional_runs[run]
        print(f"\nProcessing Run {run}:")
        df_optics= pd.read_csv(opticsfile)
        df_traditional = pd.read_csv(traditional_file)
        df_optics_columns = df_optics.apply(analyze_row_labels, axis=1)
        df_optics_columns.columns = ['cluster_category', 'total_hits_in_row', 'noise_hits_in_row','signal_hits_in_row']
        df_optics = pd.concat([df_optics, df_optics_columns], axis=1)

        df_optics_filtered = df_optics[(df_optics["clusterPE"] < 80) & (df_optics["clusterChargeBalance"] < 0.45) & (df_optics["clusterHits"] > 9)]
        df_traditional_filtered = df_traditional[(df_traditional["clusterPE"] < 80) & (df_traditional["clusterChargeBalance"] < 0.45) & (df_traditional["clusterHits"] > 9)]

        optics_good_events = set(df_optics_filtered[df_optics_filtered['cluster_category'] == 1]['eventTankTime'].unique())
        print(f"total OPTICS events: {len(optics_good_events)} ")

        traditional_good_events = set(df_traditional_filtered['eventTankTime'].unique())
        print(f"total Traditional good events: {len(traditional_good_events)} ")

        agreed_events = optics_good_events.intersection(traditional_good_events)
        print(f"Agreement: {len(agreed_events)}")

        optics_only_events = optics_good_events - traditional_good_events
        print(f"OPTICS Only: {len(optics_only_events)} ")
        
        traditional_only_events = traditional_good_events - optics_good_events
        print(f"Traditional Only: {len(traditional_only_events)}")

        if not traditional_only_events:
            print("No disagreements to plot. The sets matched perfectly for 'Pure Signal' events!")
            sys.exit()

        df_disagree = df_optics[df_optics['eventTankTime'].isin(traditional_only_events)].copy()

        # Map the numeric category back to a readable string for the plot legend
        category_map = {
            0: 'Pure Noise (OPTICS)',
            1: 'Pure Signal (OPTICS)', 
            -1: 'Mixed (OPTICS)',
            2: 'Pure Signal Multi-Cluster (OPTICS)',
            'No Hits': 'No Hits (OPTICS)',
            99: 'Other/Uncategorized (OPTICS)'
        }

        # Create a new column for plotting, mapping the numbers to strings
        df_disagree['plot_category'] = df_disagree['cluster_category'].map(category_map).fillna(df_disagree['cluster_category'].astype(str))

        # --- Create Plot 1: Scatter Plot ---
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_disagree,
            x='clusterPE',
            y='clusterChargeBalance',
            hue='plot_category',  # Color by what OPTICS *thought* these clusters were
            alpha=0.7,
            s=50 # dot size
        )
        plt.title(f'Disagreement: {len(traditional_only_events)} only Traditional Set, but Rejected by OPTICS for run {run}', fontsize=16)
        plt.xlabel('Cluster PE (from OPTICS file)', fontsize=12)
        plt.ylabel('Cluster Charge Balance (from OPTICS file)', fontsize=12)
        plt.legend(title='OPTICS Classification')
        plt.tight_layout()
        pdf.savefig()
        plt.close() 

        
        plt.figure(figsize=(10, 6))
        disagree_clusters = df_disagree[df_disagree['cluster_category'] != 1]

        plt.hist2d(disagree_clusters['clusterPE'], disagree_clusters['clusterChargeBalance'], 
                bins=100, cmap='viridis', 
                range=[[-10, 150], [0.1, 1.0]], cmin=1)
        plt.colorbar(label='Counts')
        plt.title(f"PE vs CCB for Disagreed Clusters (Traditional=Good, OPTICS!=Good)")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        event_multiplicity = df_disagree['eventTankTime'].value_counts()

        plt.figure(figsize=(10, 6))
        max_multiplicity = event_multiplicity.max()
        bins = np.arange(1, 10) 

        plt.hist(event_multiplicity, bins=10, edgecolor='blue', 
                color="lightblue", linewidth=0.5, align='left')
        plt.xlabel('Number of Disagreed Clusters per Event')
        plt.ylabel('Counts of Events')
        plt.title(f'Multiplicity of Disagreed Clusters for run {run}') 
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        if not agreed_events:
            print("No agreement events to plot.")
        else:

            df_agree = df_optics_filtered[
                (df_optics_filtered['eventTankTime'].isin(agreed_events)) &
                (df_optics_filtered['cluster_category'] == 1)
            ]

            # Count how many *clusters* per event were in the agreement set
            agreement_multiplicity = df_agree['eventTankTime'].value_counts()
            plt.figure(figsize=(10, 6))
            plt.hist(agreement_multiplicity, bins=range(1, 10, 1), edgecolor='blue', 
                    color="lightblue", linewidth=0.5, align='left', density=False)
            plt.xlabel('Neutron multiplicity for Events')
            plt.ylabel('Counts')
            plt.title(f'AmBe Neutron multiplicity distribution for run {run} (PE < 80, CCB < 0.45, CH > 9) - AGREED EVENTS')
            plt.tight_layout()
            pdf.savefig()
            plt.close()


# ---------------------------------------------------------------------------
# ambe CLI integration
# ---------------------------------------------------------------------------
def run(ctx, argv=None):
    """Not yet ported to RunContext API. Run this script directly."""
    raise NotImplementedError(
        f"This module ({__name__}) has not been ported to the ambe CLI yet. "
        "Run it directly as a Python script."
    )


def cli(ctx, argv=None):
    run(ctx, argv)
