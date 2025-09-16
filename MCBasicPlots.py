import uproot
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from scipy.stats import chi2
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import trange, tqdm
import lmfit
import pymc as pm
import arviz as az


def NeutCapture(t, A, therm, tau, B):
    return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B

root_files = sorted(glob.glob(os.path.join("../WCSim_Neutrons/", "AmBe_Neutron_*.root")))

combined_clustertime = []
combined_pe= []
combined_charge_balance = []

time_fit_values = []
lmfit_summary = []
pyMC_summary = []


for file in tqdm(root_files):
    print(f"Processing: {file}")

    base = os.path.basename(file)
    match = re.search(r'AmBe_Neutron_(.*)\.root', base)
    tag = match.group(1) if match else "Unknown"
    clusterTime = []
    clusterPE = []
    clusterChargeBalance = []
    try:
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

    except Exception as e:
        print(f"Error opening file {file}: {e}")
        continue


    counts, bin_edges = np.histogram(clusterTime, bins=70, range=(0, 70))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fit_mask = (bin_centers > 4) & (bin_centers < 67)
    xdata = bin_centers[fit_mask]

    init = [np.max(counts), 5, 25, np.min(counts)]

    ydata = counts[fit_mask]
    ydata_errors = np.sqrt(ydata)
    ydata_errors[ydata_errors == 0] = 1e-10

    popt, pcov = curve_fit(NeutCapture, xdata, ydata, p0=init, sigma=ydata_errors, absolute_sigma=True,
                            bounds=([0, 0.1, 0.1, 0], [np.inf, 100, 100, np.inf]))
    perr = np.sqrt(np.diag(pcov))
    ydata_expected = NeutCapture(xdata, *popt)  
    chi2_value = np.sum(((ydata - ydata_expected) ** 2) / (ydata_errors ** 2))
    ndof = len(ydata) - len(popt)
    chi2_ndof = chi2_value / ndof
    p_value = chi2.sf(chi2_value, ndof)

    time_fit_values.append((f"{popt[1]:.2f}", f"{perr[1]:.2f}", f"{popt[2]:.2f}", f"{perr[2]:.2f}", f"{chi2_ndof:.2f}", f"{p_value:.3f}"))
    
    plt.figure()
    plt.hist(clusterTime, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
    plt.errorbar(xdata, ydata,  yerr=ydata_errors,  color='blue', linestyle='None', alpha=0.7)

    label = (
        fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
        fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
        fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof}$" + "\n"
        fr"$\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$, p = {p_value:.3f}"
        
    )

    plt.plot(xdata, NeutCapture(xdata, *popt), 'r-', linewidth=2, label=label)
    plt.xlabel(fr"Cluster Time [$\mu s$]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"Neutron Capture Time for AmBe 2.0 C1 (PE < 100, CCB < 0.45), Run: {tag}")
    #plt.savefig("OutputPlots/NeutronCaptureTime_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.tight_layout()
    plt.show()
    plt.close()
    print(f"Thermal time:{popt[1]:.2f}", f"{perr[1]:.2f}", f"Capture time:{popt[2]:.2f}", f"{perr[2]:.2f}", f"{chi2_ndof:.2f}", f"{p_value:.3f}")

    ##########
    # Fit the model to the data using LMFIT
    ##########
    model = lmfit.Model(NeutCapture)
    params = model.make_params(A=200, therm=5, tau=25, B=0)
    params['A'].min = 0
    params['therm'].min = 1e-9
    params['tau'].min = 1e-9
    params['B'].min = -5
    result = model.fit(ydata, params, t=xdata, weights=1/ydata_errors)
    print(result.fit_report()) ## Make a dataframe out of it.
    best_fit_curve = result.best_fit
    
    lm_run_summary = {
        "Thermal": result.params['therm'].value,
        "Thermal_err": result.params['therm'].stderr,
        "Tau": result.params['tau'].value,
        "Tau_err": result.params['tau'].stderr,
        "chi2": result.chisqr,
        "reduced_chi2": result.redchi,
        "p_value": chi2.sf(result.chisqr, result.nfree)    
    }
    lmfit_summary.append(lm_run_summary)
    ##########


Info = pd.DataFrame(time_fit_values, columns=['ThermalTime','ThermalTimeErr', 'CaptureTime','CaptureTimeErr', "Chi2Ndof", "p-value"])
Info['ThermalTime'] = Info['ThermalTime'].astype(float)
Info['ThermalTimeErr'] = Info['ThermalTimeErr'].astype(float)
Info['CaptureTime'] = Info['CaptureTime'].astype(float)
Info['CaptureTimeErr'] = Info['CaptureTimeErr'].astype(float)



#######LMFIT############
lmfit_df = pd.DataFrame(lmfit_summary)
print(lmfit_df)
#######################

def weighted_average(df, value, error):
    weights = 1.0 / (df[error] ** 2)
    weighted_avg = ((df[value] * weights).sum()) / (weights.sum())
    weighted_err = np.sqrt(1 / weights.sum())
    return weighted_avg, weighted_err

NeutCaptureTime, NeutCaptureTimeErr = weighted_average(Info, 'CaptureTime', 'CaptureTimeErr')
print(f"Weighted Average Capture Time: {NeutCaptureTime:.2f} ± {NeutCaptureTimeErr:.2f} μs")

NeutThermalTime, NeutThermalTimeErr = weighted_average(Info, 'ThermalTime', 'ThermalTimeErr')
print(f"Weighted Average Thermal Time: {NeutThermalTime:.2f} ± {NeutThermalTimeErr:.2f} μs")

lmfitThermalTime, lmfitThermalTimeErr = weighted_average(lmfit_df, 'Thermal', 'Thermal_err')
print(f"Weighted Average LMFIT Thermal Time: {lmfitThermalTime:.2f} ± {lmfitThermalTimeErr:.2f} μs")

lmfitCaptureTime, lmfitCaptureTimeErr = weighted_average(lmfit_df, 'Tau', 'Tau_err')
print(f"Weighted Average LMFIT Capture Time: {lmfitCaptureTime:.2f} ± {lmfitCaptureTimeErr:.2f} μs")    



    
