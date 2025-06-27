import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

path = './'  # Directory containing the CSV files
csv_files = glob.glob(os.path.join(path, 'EventAmBeNeutronCandidatesPE150CB0.3_*.csv'))

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
plt.hist(events_counts, bins=range(1, 10, 1), log=True, edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
plt.xlabel('Neutron multiplicity')
plt.ylabel('Counts')
plt.title('AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 150, CB < 0.3)')
plt.savefig("NeutronMultiplicity_AmBe2.0wPE150CB0.3.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot Neutron Capture Time
counts, bin_edges = np.histogram(CT, bins=70, range=(0, 70))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

init = [np.max(counts), 20, np.min(counts)]


fit_mask = (bin_centers > 15) & (bin_centers < 65)
fit_x = bin_centers[fit_mask]
fit_y = counts[fit_mask]
fit_y_errors = np.sqrt(fit_y)

popt, pcov = curve_fit(NeutCapture, fit_x, fit_y, p0=init, sigma=fit_y_errors, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))


fit_y_expected = NeutCapture(fit_x, *popt)
  # or provide your own array of errors
fit_y_errors[fit_y_errors == 0] = 1e-10
chi2 = np.sum(((fit_y - fit_y_expected) ** 2) / (fit_y_errors ** 2))
ndof = len(fit_y) - len(popt)
chi2_ndof = chi2 / ndof
print(fr"$\chi^2 = {chi2:.2f}$, ndof = {ndof}, $\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$")

plt.hist(CT, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
label = (
    fr"$\tau = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
    fr"$\chi^2 = {chi2:.2f},\ \mathrm{{ndof}} = {ndof},\ "
    fr"\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$"
)
plt.plot(fit_x, NeutCapture(fit_x, *popt), 'r-', linewidth=2, label=label)
plt.xlabel(fr"Cluster Time [$\mu s$]")
plt.ylabel("Counts")
plt.legend()
plt.title(f"Neutron Capture Time for AmBe 2.0 (PE < 150, CB < 0.3)")
plt.savefig("NeutronCaptureTime_AmBe2.0wPE150CB0.3.png", dpi=300, bbox_inches='tight')
plt.show()


plt.hist2d(PE, CCB, bins=70, cmap='viridis', 
        range=[[0, 100], [0, 0.5]], cmin=1)
plt.colorbar(label='Counts')
plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0 (PE < 150, CB < 0.3)")
plt.xlabel("Cluster PE")
plt.ylabel("Cluster Charge Balance")
plt.savefig("ClusterPE_vs_ChargeBalance_AmBe2.0wPE150CB0.3.png", dpi=300, bbox_inches='tight')
plt.show()

residuals = (fit_y - fit_y_expected) / fit_y_errors
plt.plot(fit_x, residuals)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Time [μs]")
plt.ylabel("Normalized Residual")
plt.title("Fit Residuals for AmBe 2.0 Neutron Capture Time (PE < 150, CB < 0.3)")
plt.savefig("FitResiduals_AmBe2.0wPE150CB0.3.png", dpi=300, bbox_inches='tight')
plt.show()

