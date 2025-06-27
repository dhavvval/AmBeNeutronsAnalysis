import os          
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

##This one to make plots of different ports and depths for AmBe neutron source positions####

files = './' 
csvs = glob.glob(os.path.join(files, 'EventAmBeNeutronCandidates_*.csv'))

source_groups = {}
def NeutCapture(t, A, tau, B):
    return A * np.exp(-t / tau) + B

with PdfPages('AllAmBePositionsPlots.pdf') as pdf:
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

        x, y, z = df["sourceX"].iloc[0], df["sourceY"].iloc[0], df["sourceZ"].iloc[0]
        source_key = (round(x, 3), round(y, 3), round(z, 3))
        if source_key not in source_groups:
            source_groups[source_key] = []
        source_groups[source_key].append(df)

    for source_key, df_list in source_groups.items():
        combined_df = pd.concat(df_list, ignore_index=True)
        

        events_counts = combined_df['eventID'].value_counts()
        PE = combined_df['clusterPE']
        CCB = combined_df['clusterChargeBalance']
        CT = combined_df['clusterTime']/1000 # Convert to microseconds

        # Plot
        ## Neutron multiplicity
        plt.hist(events_counts, bins=range(1, 10, 1), log=True, edgecolor='black', linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity')
        plt.ylabel('Counts')
        plt.title('AmBe Neutron multiplicity for position: ' + f"{source_key[0]}, {source_key[1]}, {source_key[2]}")
        pdf.savefig()
        plt.close()

        ## Neutron capture time 
        counts, bin_edges = np.histogram(CT, bins=100, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        init = [np.max(counts), 15, np.min(counts)]

        fit_mask = (bin_centers > 20) & (bin_centers < 67)
        fit_x = bin_centers[fit_mask]
        fit_y = counts[fit_mask]

        popt, pcov = curve_fit(NeutCapture, fit_x, fit_y, p0=init)
        perr = np.sqrt(np.diag(pcov))

        #popt, pcov = curve_fit(NeutCapture, bin_centers, counts, p0=init)


        plt.hist(CT, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
        plt.plot(bin_centers, NeutCapture(bin_centers, *popt), 'r-', linewidth=2, label=fr"Fit: $\tau = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$")
        plt.xlabel("Cluster Time [µs]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(f"Neutron Capture Time for Position:{source_key[0]}, {source_key[1]}, {source_key[2]}")
        pdf.savefig()
        plt.close()


        plt.hist2d(PE, CCB, bins=70, cmap='viridis', 
                range=[[0, 100], [0, 0.5]], cmin=1)
        plt.colorbar(label='Counts')
        plt.title(f"{data_type} Cluster PE vs Charge Balance for Position: {source_key[0]}, {source_key[1]}, {source_key[2]}")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        pdf.savefig()
        plt.close()