import os          
import numpy as np
import uproot
from tqdm import trange
from scipy.stats import norm
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq



# AmBe neutrons
def AmBe(CPE, CCB, CT, CN, ETT):
    if(CPE<=0 or CPE>150):      # 0 < cluster PE < 150
        return False
    if(CCB>=0.4 or CCB<=0):   # Cluster Charge Balance < 0.4
        return False
    if(CT<2000):              # cluster time not in prompt window
        return False        
    return True

# cosmic muon clusters
def cosmic(CT, CPE):
    if(CT<2000):              # any cluster in the prompt (2us) window
        return True
    if(CPE>150):              # any cluster > 150 PE in the prompt or ext window
        return True
    return False

# grab source location based on the run number
def source_loc(run):
    
    run = int(run)
    
    source_positions = {
        
        # Port 5 data
        4506: (0, 100, 0),
        4505: (0, 50, 0),
        4499: (0, 0, 0),
        4507: (0, -50, 0),
        4508: (0, -100, 0),
        4496: (0, 0, 0), #AmBe run without source
        
        # Port 1 data
        4593: (0, 100, -75),
        4590: (0, 50, -75),
        4589: (0, 0, -75),
        4596: (0, -50, -75),
        4598: (0, -100, -75),
        
        # Port 4 data
        4656: (-75, 100, 0),
        4658: (-75, 50, 0),
        4660: (-75, 0, 0),
        4662: (-75, -50, 0), 4664: (-75, -50, 0), 4665: (-75, -50, 0), 4666: (-75, -50, 0), 4667: (-75, -50, 0), 4670: (-75, -50, 0),
        4678: (-75, -100, 0), 4683: (-75, -100, 0), 4687: (-75, -100, 0),
        
        # Port 3 data
        4628: (0, 100, 102), 4629: (0, 100, 102),
        4633: (0, 50, 102),
        4635: (0, 0, 102), 4636: (0, 0, 102), 4640: (0, 0, 102), 4646: (0, 0, 102),
        4649: (0, -50, 102),
        4651: (0, -100, 102),

        #port 2 data
        4453: (0, 0, 75),
        4603: (0, 100, 75),
        4604: (0, 50, 75),
        4605: (0, 50, 75),
        4625: (0, -50, 75),

        #outside the tank data
        4707: (0, 328, 0), 
        4708: (0, 328, 0),

        ##New runs for AmBe v1 Campaign 2 - July 2025
        5682: (0, 0 ,0),
        5680: (0, 0, 0),
        5681: (0, 0, 0),
        5683: (0, -100, 0),
        5684: (0, -100, 0),
        5688: (0, 100, 0), #dummy port number actual one is (0, -100, 0)
        5689: (0, 50, 0), #dummy port number actual one is (0, -100, 0)
        5691: (0, 0, 0), #dummy port number actual one is (0, -100, 0)
        5693: (0, 0, 0) #dummy port number actual one is (0, -100, 0)
        
    }
        
    if run in source_positions:
        return source_positions[run]

    print('\n##### RUN NUMBER '+str(run)+' DOESNT HAVE A SOURCE LOCATION!!! ERROR #####\n')
    exit()

def AmBePMTWaveforms(data_directory, waveform_dir, file_pattern, source_loc,
                      pulse_start=300, pulse_end=1200, pulse_gamma=400, lower_pulse=175,
                      pulse_max=1000, NS_PER_ADC_SAMPLE=2, ADC_IMPEDANCE=50, runinfo='default', campaign=1): #pulse_gamma = 400, lower_pulse = 175


    file_names = []
    run_numbers = []

    for file_name in os.listdir(data_directory):
        match = file_pattern.match(file_name)
        print('Checking file: ' + file_name)
        print('Match: ', match)
        if match:
            run_number = int(match.group(1))
            run_numbers.append(str(run_number))
            file_names.append(os.path.join(data_directory, file_name))
    if campaign == 1:
        folder_pattern = re.compile(r'^RWM_\d+')
    elif campaign == 2:
        folder_pattern = re.compile(r'^BRF_\d+')
    results = {}

    combined_IC_values = []
    combined_IC_accepted = []
    IC_values = [] 

    for c1, run in enumerate(run_numbers): 
        IC_accepted = []
        seen_timestamps = set()    
        print(f"\n\nRun: {run} ({c1+1}/{len(file_names)})")
        print('-----------------------------------------------------------------')

        x_pos, y_pos, z_pos = source_loc(run)
        key = (x_pos, y_pos, z_pos)
        print(f'Source position (x,y,z): ({x_pos},{y_pos},{z_pos})')

        good_events = []
        accepted_events = 0
        rejected_events = 0
        counter = 0

        waveform_files = os.listdir(os.path.join(waveform_dir, run))

        waveformsample = input("Do you want to see samples of waveforms? (y/n): ")
        IC_plots = input('Do you want to plot IC values? (y/n): ')


        print('\nLoading Raw Waveforms...')
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

                        baseline, sigma = norm.fit(hist_values)
                        
                        hist_values_bs = hist_values - baseline

                        pulse_mask = (hist_edges[:-1] > pulse_start) & (hist_edges[:-1] < pulse_end)
                        #baseline, sigma = norm.fit(pulse_mask)
                       # print(f'Baseline: {baseline}, Sigma: {sigma}')
                        IC = np.sum(hist_values[pulse_mask]-baseline ) #- baseline
                        IC_adjusted = (NS_PER_ADC_SAMPLE / ADC_IMPEDANCE) * IC
                       # IC_values.append(IC_adjusted)
                        combined_IC_values.append(IC_adjusted)
                        IC_values.append(IC)

                        if pulse_max > IC_adjusted > pulse_gamma:
                            post_pulse_mask = hist_edges[:-1] > pulse_end
                            post_pulse_values = hist_values[post_pulse_mask]
                            another_pulse = np.any(post_pulse_values > (7 + sigma + baseline))
                           # IC_accepted.append(IC_adjusted)
                            combined_IC_accepted.append(IC_adjusted)

                            if not another_pulse:
                                good_events.append(int(timestamp))
                                accepted_events += 1

                                if waveformsample == 'y':
                                    if accepted_events in [1, 10, 20]:
                                        plt.figure(figsize=(10, 4))
                                        plt.plot(hist_edges[:-1], hist_values_bs, label='Waveform')
                                        plt.axhline(baseline + sigma + 7, color='r', linestyle='--', label='Second Pulse Threshold')
                                        plt.axvspan(pulse_start, pulse_end, color='yellow', alpha=0.3, label='Integration Window')
                                        plt.title(f'Accepted Waveform (timestamp: {timestamp}, Run: {run}), IC: {IC_adjusted:.2f}')
                                        plt.xlabel('Time (ns)')
                                        plt.ylabel('ADC counts')
                                        plt.legend()
                                        plt.tight_layout()
                                        plt.show()
                                else:
                                    continue
                                    
                            else:
                                rejected_events += 1

                                if waveformsample == 'y':
                                    if rejected_events in [1, 10, 20]:
                                        plt.figure(figsize=(10, 4))
                                        plt.plot(hist_edges[:-1], hist_values_bs, label='Waveform')
                                        plt.axhline(baseline + sigma + 7, color='r', linestyle='--', label='Second Pulse Threshold')
                                        plt.axvspan(pulse_start, pulse_end, color='yellow', alpha=0.3, label='Integration Window')
                                        plt.title(f'Rejected Waveform (2nd pulse) (timestamp: {timestamp}), run: {run}, IC: {IC_adjusted:.2f}')
                                        plt.xlabel('Time (ns)')
                                        plt.ylabel('ADC counts')
                                        plt.legend()
                                        plt.tight_layout()
                                        plt.show()
                                else:
                                    continue

                        else:
                            rejected_events += 1
                            if waveformsample == 'y':
                                if rejected_events in [3, 6, 9]:
                                    plt.figure(figsize=(10, 4))
                                    plt.plot(hist_edges[:-1], hist_values_bs, label='Waveform')
                                    plt.axhline(baseline + sigma + 7, color='r', linestyle='--', label='Second Pulse Threshold')
                                    plt.axvspan(pulse_start, pulse_end, color='yellow', alpha=0.3, label='Integration Window')
                                    plt.title(f'Rejected Waveform (pulse max < IC_adjusted) (timestamp: {timestamp}), Run: {run}, IC: {IC_adjusted:.2f}')
                                    plt.xlabel('Time (ns)')
                                    plt.ylabel('ADC counts')
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.show()
                            else:
                                continue

                    except Exception as e:
                        print(f"Could not access '{folder}': {e}")

                    counter += 1
        plt.figure(figsize=(8, 5))
        plt.hist(combined_IC_values, bins=500, alpha=0.7, color='blue', range=(0, 1000))
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events')
        plt.title('All IC adjusted Values for run ' + run)
        plt.tight_layout()
        #plt.savefig('IC_adjusted_AllEvents.png', dpi=300)
        plt.show()
        total = accepted_events + rejected_events

        print(f'\nThere were a total of {total} acquisitions')
        print(f'{accepted_events} waveforms were accepted ({round(100*accepted_events/total,2)}%)')
        print(f'{rejected_events} waveforms were rejected ({round(100*rejected_events/total,2)}%)\n')


        results[run] = {
            'source_position': (x_pos, y_pos, z_pos),
            'accepted_events': accepted_events,
            'rejected_events': rejected_events,
            'total_waveforms': total,
            'good_events': set(good_events)
        }

    if IC_plots == 'y':
        plt.figure(figsize=(8, 5))
        plt.hist(combined_IC_values, bins=500, alpha=0.7, color='blue', range=(0, 1000))
        plt.xlabel('IC_adjusted')
        plt.ylabel('Number of Events')
        plt.title('All IC adjusted Values for all runs')
        plt.tight_layout()
        plt.savefig(f'IC_adjusted_AllEvents_{runinfo}.png', dpi=300)
        #plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(IC_values, bins=500, alpha=0.7, color='blue', range=(0, 1000))
        plt.xlabel('IC')
        plt.ylabel('Number of Events')
        plt.title('All IC Values for all runs')
        plt.tight_layout()
        plt.savefig(f'IC_adjusted_AllEvents_{runinfo}.png', dpi=300)
        #plt.show()

        # Second histogram
        plt.figure(figsize=(8, 5))
        plt.hist(combined_IC_accepted, bins=500, alpha=0.7, color='orange', range=(0, 1000))
        plt.xlabel('IC_adjusted accepted')
        plt.ylabel('Number of Events')
        plt.title('Accepted IC_adjusted Values for all runs')
        plt.tight_layout()
        plt.savefig(f'IC_adjusted_AcceptedEvents_{runinfo}.png', dpi=300)
        #plt.show()


    return results, run_numbers, file_names

def LoadBeamCluster(file_path, which_Tree):
    print('\nLoading AmBe event data...')

    with uproot.open(file_path) as file_1:

        if which_Tree == 0:
            Event = file_1["Event"]
        else:
            Event = file_1["Event"]

        data = {
            "eventNumber": Event["eventNumber"].array(),
            "eventTimeTank": Event["eventTimeTank"].array(),
            "clusterTime": Event["clusterTime"].array(),
            "clusterPE": Event["clusterPE"].array(),
            "clusterChargeBalance": Event["clusterChargeBalance"].array(),
            "clusterHits": Event["clusterHits"].array(),
        }

        # Tree-dependent branches
        if which_Tree == 0:
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

def process_events(event_data, good_events, x_pos, y_pos, z_pos,
                   cosmic, AmBe,
                   cluster_time, cluster_charge, cluster_QB, cluster_hits,
                   hit_times, hit_charges, hit_ids, source_position, event_ids,  efficiency_data=None):

    EN = event_data["eventNumber"]
    ETT = event_data["eventTimeTank"]
    CT = event_data["clusterTime"]
    CPE = event_data["clusterPE"]
    CCB = event_data["clusterChargeBalance"]
    CH = event_data["clusterHits"]
    CN = event_data["clusterNumber"]
    hT = event_data["hitT"]
    hPE = event_data["hitPE"]
    hID = event_data["hitDetID"]

    total_events = 0
    cosmic_events = 0
    neutron_cand_count = 0

    key = (x_pos, y_pos, z_pos)

    for i in trange(len(EN)):
        if ETT[i] in good_events:
            
            total_events += 1

            if efficiency_data is not None:
                efficiency_data[key][0] += 1  # total_events

            for k in range(len(CT[i])):
                if cosmic(CT[i][k], CPE[i][k]):
                    cosmic_events += 1
                    if efficiency_data is not None:
                        efficiency_data[key][1] += 1 # cosmic_events
                    break

                if AmBe(CPE[i][k], CCB[i][k], CT[i][k], CN[i], ETT[i]):
                    if CPE[i][k] != float('-inf'):
                        cluster_time.append(CT[i][k])
                        cluster_charge.append(CPE[i][k])
                        cluster_QB.append(CCB[i][k])
                        cluster_hits.append(CH[i][k])
                        hit_times.append(hT[i][k])
                        hit_charges.append(hPE[i][k])
                        hit_ids.append(hID[i][k])
                        source_position[0].append(x_pos)
                        source_position[1].append(y_pos)
                        source_position[2].append(z_pos)
                        event_ids.append(EN[i])
                        neutron_cand_count += 1
                        if efficiency_data is not None:
                            efficiency_data[key][2] += 1

    print(f'\nThere were a total of {total_events} AmBe events after waveform cuts')
    if total_events == 0:
        print("No events found. Cannot calculate cosmic percentage.")
    else:
        print(f'{cosmic_events} ({round(100 * cosmic_events / total_events, 2)}%) were cosmics')

    print(f'This leaves {total_events - cosmic_events} AmBe-triggered events')
    print(f'\nAfter selection cuts: {neutron_cand_count} AmBe neutron candidates\n')
    

    return total_events, cosmic_events, neutron_cand_count, event_ids


