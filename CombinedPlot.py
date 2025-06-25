import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

path = './'  # Directory containing the CSV files
csv_files = glob.glob(os.path.join(path, 'EventAmBeNeutronCandidates_*.csv'))

def NeutCapture(t, A, tau, B):
    return A * np.exp(-t / tau) + B

all_df = []
for file in csv_files:
    df = pd.read_csv(file)
    all_df.append(df)

combined_df = pd.concat(all_df, ignore_index=True)
    
events_counts = combined_df['eventID'].value_counts()

PE = combined_df['clusterPE']
CCB = combined_df['clusterChargeBalance']
CT = combined_df['clusterTime'] / 1000  # Convert to microseconds

# neutron multiplicty
plt.hist(events_counts, bins=range(1, 10, 1), log=True, edgecolor='black', linewidth=0.5, align='left', density=False)
plt.xlabel('Neutron multiplicity')
plt.ylabel('Counts')
plt.title('AmBe Neutron multiplicity distribution from AmBe 2.0 campaign')
plt.show()

# Plot Neutron Capture Time
counts, bin_edges = np.histogram(CT, bins=100, range=(0, 70))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

init = [np.max(counts), 15, np.min(counts)]
print(f"Initial parameters for curve fit: {init}")

fit_mask = (bin_centers > 20) & (bin_centers < 67)
fit_x = bin_centers[fit_mask]
fit_y = counts[fit_mask]

popt, pcov = curve_fit(NeutCapture, fit_x, fit_y, p0=init)
perr = np.sqrt(np.diag(pcov))


#popt, pcov = curve_fit(NeutCapture, bin_centers, counts, p0=init)

plt.hist(CT, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
plt.plot(bin_centers, NeutCapture(bin_centers, *popt), 'r-', linewidth=2, label=fr"Fit: $\tau = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$")
plt.xlabel(fr"Cluster Time [$\mu s$]")
plt.ylabel("Counts")
plt.legend()
plt.title(f"Neutron Capture Time Fit")
plt.show()


plt.hist2d(PE, CCB, bins=70, cmap='viridis', 
        range=[[0, 100], [0, 0.5]], cmin=1)
plt.colorbar(label='Counts')
plt.title(f"Cluster PE vs Charge Balance")
plt.xlabel("Cluster PE")
plt.ylabel("Cluster Charge Balance")
plt.show()
