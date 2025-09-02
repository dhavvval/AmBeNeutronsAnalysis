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

data_directory = '../AmBe_BeamCluster/'                                    # directory containing BeamClusterAnalysis ntuples
waveform_dir = '../AmBe_waveforms/'                             # directory containing raw AmBe PMT waveforms


#data_directory = '../test/'                                    # directory containing BeamClusterAnalysis ntuples
#waveform_dir = '../test/'

#data_directory = '../Outside_source/'                                    # directory containing BeamClusterAnalysis ntuples
#waveform_dir = '../Outside_source/'

##Test for my gridjobs
#data_directory = '../NewRun/test'                                    # directory containing BeamClusterAnalysis ntuples
#waveform_dir = '../NewRun/test'

#data_directory = '../AnalysisQualityC2/newPMTPhysics'                                    # directory containing BeamClusterAnalysis ntuples
#waveform_dir = '../AnalysisQualityC2/newPMTPhysics'

campaign = int(input('What campaign is this? (1/2): '))  # Campaign 1 or 2

if campaign == 1:
    file_pattern = re.compile(r'AmBe_(\d+)_v\d+\.ntuple\.root')      # Pattern to extract run numbers from the files: AmBe_<run_number>_v<version>.ntuple.root
elif campaign == 2:
    file_pattern = re.compile(r'BeamCluster_(\d+)\.root')
     # Pattern to extract run numbers from the files: R<run_number>_AmBe.ntuple.root -- edit to match your filename pattern

##File pattern for the AmBe v1 Campaign 2 - July 2025
#file_pattern = re.compile(r'BeamCluster_(\d+)\.root') ## BeamCluster_4708.root

which_Tree = 1                                               # PhaseIITreeMaker (0) or ANNIEEventTreeMaker (1) tool

central_port = [4506, 4505, 4499, 4507, 4508] #port 5 with AmBe source


runinfo = input('What type of run is this? (AmBe/Outside_source//C1/C2/ClusterCuts): ')
runinfo = str(runinfo)

efficiency_data = defaultdict(lambda: [0, 0, 0, 0])

waveform_results, run_numbers, file_names = ane.AmBePMTWaveforms(data_directory, waveform_dir, file_pattern, ane.source_loc, runinfo=runinfo, campaign=campaign)

waveform_df = pd.DataFrame.from_dict(waveform_results, orient='index')

if 'good_events' in waveform_df.columns:
    waveform_df = waveform_df.drop(columns=['good_events'])

waveform_df[['x_pos', 'y_pos', 'z_pos']] = pd.DataFrame(waveform_df['source_position'].tolist(), index=waveform_df.index)
waveform_df = waveform_df.drop(columns=['source_position'])

waveform_df = waveform_df.groupby(['x_pos', 'y_pos', 'z_pos'], as_index=False).sum()

waveform_df.to_csv(f'TriggerSummary/AmBeWaveformResults_{runinfo}.csv', index = False)  # Save the waveform results to a CSV file

cluster_time = []
cluster_charge = []
cluster_QB = []
cluster_hits = []
hit_times = []
hit_charges = []
hit_ids = []
source_position = [[], [], []]
event_ids = []
prompt_cluster_time = []
prompt_cluster_charge = []
prompt_cluster_QB = []


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
    prompt_cluster_time = []
    prompt_cluster_charge = []
    prompt_cluster_QB = []
   
    
    good_events = waveform_results[run]["good_events"]
    x_pos, y_pos, z_pos = waveform_results[run]["source_position"]
    file_path = file_names[c1]

    event_data = ane.LoadBeamCluster(file_path, which_Tree)

    total_events, cosmic_events, neutron_cand_count, multiple_neutron_cand_count, event_ids = ane.process_events(
        event_data, good_events, x_pos, y_pos, z_pos,
        ane.cosmic, ane.AmBe, ane.AmBeMultiple,
        cluster_time, cluster_charge, cluster_QB, cluster_hits,
        hit_times, hit_charges, hit_ids, source_position, event_ids,
        prompt_cluster_time, prompt_cluster_charge, prompt_cluster_QB, efficiency_data
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
    df.to_csv(f'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_{runinfo}_{run}.csv', index=False) ##This files to do analysis for multiplicty, capture time, and other plots of Charge current vs Cluster time etc

    prompt_df = pd.DataFrame({
        "prompt_clusterTime": prompt_cluster_time,
        "prompt_clusterPE": prompt_cluster_charge,
        "prompt_clusterChargeBalance": prompt_cluster_QB,

    })
    prompt_df.to_csv(f'EventAmBeNeutronCandidatesData/PromptAmBeNeutronCandidates_{runinfo}_{run}.csv', index=False) ##This files to do analysis for capture time, charge balance, and PE of prompt neutrons

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

df_eff.to_csv(f'TriggerSummary/AmBeTriggerSummary_{runinfo}.csv', index=False) ## This file to do analysis for efficiency heatmap.



