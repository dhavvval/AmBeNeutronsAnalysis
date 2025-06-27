import os          
import numpy as np
import uproot
from tqdm import trange
from scipy.stats import norm
import re
import AmBeNeutronEff as ane
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as scm
import scipy.optimize as scp
from collections import defaultdict
import pandas as pd


# edit accordingly

data_directory = '../AmBe_BeamCluster/'                                     # directory containing BeamClusterAnalysis ntuples
waveform_dir = '../AmBe_waveforms/'                             # directory containing raw AmBe PMT waveforms

file_pattern = re.compile(r'AmBe_(\d+)_v\d+\.ntuple\.root')      # Pattern to extract run numbers from the files: R<run_number>_AmBe.ntuple.root -- edit to match your filename pattern


which_Tree = 1                                               # PhaseIITreeMaker (0) or ANNIEEventTreeMaker (1) tool

Events = {4506, 4505, 4499}
    
Backgrounds = {4496}

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)
mypoissons = lambda x,R1,mu1,R2,mu2: R1*(mu1**x)*np.exp(-mu2)/scm.factorial(x) + R2*(mu2**x)*np.exp(-mu2)/scm.factorial(x)


waveform_results, run_numbers, file_names = ane.AmBePMTWaveforms(data_directory, waveform_dir, file_pattern, ane.source_loc)


cluster_time = []
cluster_charge = []
cluster_QB = []
cluster_hits = []
hit_times = []
hit_charges = []
hit_ids = []
source_position = [[], [], []]
event_ids = []
efficiency_data = defaultdict(lambda: [0, 0, 0])

for c1, run in enumerate(run_numbers):
    print(f"\nProcessing run {run} ({c1+1}/{len(run_numbers)})")

    cluster_time = []
    cluster_charge = []
    cluster_QB = []
    cluster_hits = []
    hit_times = []
    hit_charges = []
    hit_ids = []
    source_position = [[], [], []]
    event_ids = []
   
    
    good_events = waveform_results[run]["good_events"]
    x_pos, y_pos, z_pos = waveform_results[run]["source_position"]
    file_path = file_names[c1]

    event_data = ane.LoadBeamCluster(file_path, which_Tree)

    total_events, cosmic_events, neutron_cand_count, event_ids = ane.process_events(
        event_data, good_events, x_pos, y_pos, z_pos,
        ane.cosmic, ane.AmBe,
        cluster_time, cluster_charge, cluster_QB, cluster_hits,
        hit_times, hit_charges, hit_ids, source_position, event_ids, efficiency_data
    )

    print('----------------------------------------------------------------\n')
    print('Total AmBe neutron candidates:', len(cluster_time))


    run = int(run)  # Ensure run is an integer for comparison
    print('Event run number:', run)
    df = pd.DataFrame({
        "clusterTime": cluster_time,
        "clusterPE": cluster_charge,
        "clusterChargeBalance": cluster_QB,
        "clusterHits": cluster_hits,
        "hitT": hit_times,
        "hitQ": hit_charges,  # optional if you have it separate
        "hitPE": hit_charges,  # assuming this is PE
        "hitID": hit_ids,
        "sourceX": source_position[0],
        "sourceY": source_position[1],
        "sourceZ": source_position[2],
        "eventID": event_ids
    })
    print(df.head())
    df.to_csv(f'EventAmBeNeutronCandidatesPE150CB0.3_{run}.csv', index=False) ##This files to do analysis for multiplicty, capture time, and other plots of Charge current vs Cluster time etc

    '''run = int(run)  # Ensure run is an integer for comparison

    print('Event run number:', run)
    df = pd.DataFrame({
        "clusterTime": cluster_time,
        "clusterPE": cluster_charge,
        "clusterChargeBalance": cluster_QB,
        "clusterHits": cluster_hits,
        "hitT": hit_times,
        "hitQ": hit_charges,  # optional if you have it separate
        "hitPE": hit_charges,  # assuming this is PE
        "hitID": hit_ids,
        "sourceX": source_position[0],
        "sourceY": source_position[1],
        "sourceZ": source_position[2],
        "eventID": event_ids
    })
    print(df.head())
    df.to_csv(f'EventAmBeNeutronCandidatesPE150CB0.3_{run}.csv', index=False)
elif run in Backgrounds:
    print('Background run number:', run)
    bdf = pd.DataFrame({
        "clusterTime": cluster_time,
        "clusterPE": cluster_charge,
        "clusterChargeBalance": cluster_QB,
        "clusterHits": cluster_hits,
        "hitT": hit_times,
        "hitQ": hit_charges,  # optional if you have it separate
        "hitPE": hit_charges,  # assuming this is PE
        "hitID": hit_ids,
        "sourceX": source_position[0],
        "sourceY": source_position[1],
        "sourceZ": source_position[2],
        "eventID": event_ids
    })
    print(bdf.head())
    bdf.to_csv(f'BackgroundAmBeNeutronCandidatesPE150CB0.3_{run}.csv', index=False)
else:
    print('Run number not recognized for AmBe neutron candidates:', run)'''

df_eff = pd.DataFrame([
    {
        "x_pos": key[0], "y_pos": key[1], "z_pos": key[2],
        "total_events": val[0],
        "cosmic_events": val[1],
        "ambe_triggers": val[0] - val[1],
        "neutron_candidates": val[2]
    }
    for key, val in efficiency_data.items()
])

df_eff.to_csv('AmBeTriggerSummaryPE150CB0.3.csv', index=False) ## This file to do analysis for efficiency heatmap.


