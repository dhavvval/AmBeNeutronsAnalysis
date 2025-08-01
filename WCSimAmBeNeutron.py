import uproot
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from scipy.stats import chi2
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import trange


def NeutCapture(t, A, therm, tau, B):
    return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B


#going through multiple files
root_files = sorted(glob.glob(os.path.join("../WCSim_Neutrons/", "AmBe_Neutron_*.root")))


combined_clustertime = []
combined_pe= []
combined_charge_balance = []

clusterTime = []
clusterPE = []
clusterChargeBalance = []

   

for file in root_files:
    print(f"Processing: {file}")

    base = os.path.basename(file)
    match = re.search(r'AmBe_Neutron_(.*)\.root', base)
    tag = match.group(1) if match else "Unknown"
    file = uproot.open(file)


    Event = file["phaseIITankClusterTree"]

    EN = Event["eventNumber"].array()
    ETT = Event["eventTimeTank"].array()
    CT = Event["clusterTime"].array()
    CPE = Event["clusterPE"].array()
    CCB = Event["clusterChargeBalance"].array()
    CH = Event["clusterHits"].array()

    for i in range(len(EN)):

        if CT[i] > 2000.0 and CPE[i] < 100 and CCB[i] < 0.45 and CH[i] > 0:
            clusterTime.append(CT[i]/1000)  # Convert to microseconds
            clusterPE.append(CPE[i])
            clusterChargeBalance.append(CCB[i])
            #print(f"Cluster Time: {CT[i]}, PE: {CPE[i]}, Charge Balance: {CCB[i]}")


df = pd.DataFrame({
    "clusterTime": clusterTime,
    "clusterPE": clusterPE,
    "clusterChargeBalance": clusterChargeBalance,


})
print(df.head())

plt.figure(figsize=(8, 6))
plt.hist2d(clusterPE, clusterChargeBalance, bins=100, cmap="viridis")
plt.colorbar(label="Counts")
plt.xlabel("Cluster PE")
plt.ylabel("Cluster Charge Balance")
plt.title(f"Cluster PE vs Cluster Charge Balance for tilted PMT Simulation (PE < 100, CCB < 0.45)")
plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(8, 6))
plt.hist(clusterTime, bins=70, range=(0, 70), histtype='step')
plt.xlabel(fr"Cluster Time [$\mu s$]")
plt.ylabel("Counts")
plt.title(f"Cluster Time Distribution for tilted PMT Simulation (PE < 100, CCB < 0.45)")
plt.tight_layout()
plt.show()
plt.close()


counts, bin_edges = np.histogram(clusterTime, bins=200, range=(0, 70))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

init = [np.max(counts), 6, 25, np.min(counts)]


fit_mask = (bin_centers > 4) & (bin_centers < 66)
fit_x = bin_centers[fit_mask]

fit_y = counts[fit_mask]
fit_y_errors = np.sqrt(fit_y)
fit_y_errors[fit_y_errors == 0] = 1e-10

popt, pcov = curve_fit(NeutCapture, fit_x, fit_y, p0=init, sigma=fit_y_errors, absolute_sigma=True, 
                       bounds=([0, 0.1, 0.1, 0], [np.inf, 100, 100, np.inf]))
perr = np.sqrt(np.diag(pcov))

fit_y_expected = NeutCapture(fit_x, *popt)

chi2_value = np.sum(((fit_y - fit_y_expected) ** 2) / (fit_y_errors ** 2))
ndof = len(fit_y) - len(popt)
chi2_ndof = chi2_value / ndof
p_value = 1 - chi2.cdf(chi2_value, ndof)
print(fr"$\chi^2 = {chi2_value:.2f}$, ndof = {ndof}, $\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$, p-value = {p_value:.3f}")

plt.figure()
plt.hist(clusterTime, bins=200, range=(0, 70), histtype='step', color='blue', label="Data")
label = (
    fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
    fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
    fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof},\ "
    fr"\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$"
)

plt.plot(fit_x, NeutCapture(fit_x, *popt), 'r-', linewidth=2, label=label)
plt.xlabel(fr"Cluster Time [$\mu s$]")
plt.ylabel("Counts")
plt.legend()
plt.title(f"Neutron Capture Time for tilted PMT Simulation (PE < 100, CCB < 0.45)")
plt.show()
