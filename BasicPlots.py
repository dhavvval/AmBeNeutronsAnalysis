from cmath import tau
import os          
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2
import ast
import lmfit
import pymc as pm
import arviz as az


##This one to make plots of different ports and depths for AmBe neutron source positions####
##It has three different types of fitting models to determine the neutron capture time distribution

files = './EventAmBeNeutronCandidatesData/' 
csvs = glob.glob(os.path.join(files, 'EventAmBeNeutronCandidates_AmBeC1Gammaregion_*.csv'))

source_groups = {}
def NeutCapture(t, A, therm, tau, B):
    return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B


with PdfPages('AllAmBePositionsPlotstestAmBeC1gROI.pdf') as pdf:
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
        if df.empty:
            print(f"No data in file: {filename}")
            continue

        x, y, z = df["sourceX"].iloc[0], df["sourceY"].iloc[0], df["sourceZ"].iloc[0]
        source_key = (round(x, 3), round(y, 3), round(z, 3))
        if source_key == (0, -105.5, 102):
            source_key = (0, -100, 102)
        if source_key not in source_groups:
            source_groups[source_key] = []
        source_groups[source_key].append(df)
 
    time_fit_values = []
    pyMC_summary = []
    lmfit_summary = []
    for source_key, df_list in source_groups.items():
        combined_df = pd.concat(df_list, ignore_index=True)
        

        events_counts = combined_df['eventID'].value_counts()
        EventTime = combined_df['eventTankTime'].value_counts()
        PE = combined_df['clusterPE']
        CCB = combined_df['clusterChargeBalance']
        CT = combined_df['clusterTime']/1000 # Convert to microseconds

        # Plot
        sx, sy, sz = (int(v) for v in source_key)
        plt.figure()
        plt.hist(EventTime, bins=range(1, 10, 1), log=True, edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0v1 g-ROI (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})')
        #plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.hist(EventTime, bins=range(1, 10, 1), edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for Events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0v1 g-ROI (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})')
        #plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot Neutron Capture Time

        counts, bin_edges = np.histogram(CT, bins=70, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fit_mask = (bin_centers > 2) & (bin_centers < 67)
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

        time_fit_values.append((f"{popt[1]:.2f}", f"{perr[1]:.2f}", f"{popt[2]:.2f}", f"{perr[2]:.2f}", (sx, sy, sz), len(EventTime), f"{chi2_ndof:.2f}", f"{p_value:.3f}", f"{popt[3]:.2f}"))
        print(f"Thermal time:{popt[1]:.2f}", f"{perr[1]:.2f}", f"Capture time:{popt[2]:.2f}", f"{perr[2]:.2f}", (sx, sy, sz), len(EventTime), f"{chi2_ndof:.2f}", f"{p_value:.3f}")

        ##########
        # Fit the model to the data using LMFIT
        ##########
        model = lmfit.Model(NeutCapture)
        params = model.make_params(A=200, therm=5, tau=25, B=0)
        params['A'].min = 0
        params['therm'].min = 1e-9
        params['tau'].min = 1e-9
        params['B'].vary = True
        params['B'].min = 0
        params['B'].max = 18
        result = model.fit(ydata, params, t=xdata, weights=1/ydata_errors, method="basinhopping")
        print(result.fit_report()) ## Make a dataframe out of it.
        best_fit_curve = result.best_fit
        
        lm_run_summary = {
            "Thermal": result.params['therm'].value,
            "Thermal_err": result.params['therm'].stderr,
            "Tau": result.params['tau'].value,
            "Tau_err": result.params['tau'].stderr,
            "Coordination": (sx, sy, sz),
            "chi2": result.chisqr,
            "reduced_chi2": result.redchi,
            "p_value": chi2.sf(result.chisqr, result.nfree)    
        }
        lmfit_summary.append(lm_run_summary)
        ##########

        #########
        # Fit the model to the data using PyMC
        #########
        print("PyMC model running for run ", (sx, sy, sz))
        def NeutCapture_safe(t, A, therm, tau, B):
            epsilon = 1e-9
            rise_term = (1 - np.exp(-t / (therm + epsilon)))
            decay_term = np.exp(-t / (tau + epsilon))
            return (A * rise_term * decay_term) + B
    
        with pm.Model() as neutron_model:
            A = pm.HalfNormal("A", sigma=500)
            therm = pm.Uniform("therm", lower=1.0, upper=15.0)
            tau = pm.Uniform("tau", lower=15.0, upper=50.0)
            B = pm.Normal("B", mu=0, sigma=1)

            mu = NeutCapture_safe(xdata, A, therm, tau, B)
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=ydata_errors, observed=ydata)
            starting_values = {"A": 200, "therm": 6.00, "tau": 27.00, "B": 0}

            with neutron_model:
                idata = pm.sample(2000, tune=1000, initvals=starting_values,  chains=4, cores=4)
            print("Done. Still you don't trust me and want to check with frequentist methods? get a life bro")
            summary_df = az.summary(idata)
            var_names = ["therm", "tau"]

            az.plot_trace(idata, var_names=var_names)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            run_summary = {
                "Therm": summary_df.loc['therm']['mean'],
                "Thermerr": summary_df.loc['therm']['sd'],
                "Tau": summary_df.loc['tau']['mean'],
                "Tauerr": summary_df.loc['tau']['sd'],
                #"p_value": p_value,
                "Coordination": (sx, sy, sz),
                "ThermHDI": (summary_df.loc['therm']['hdi_3%'], summary_df.loc['therm']['hdi_97%']),
                "TauHDI": (summary_df.loc['tau']['hdi_3%'], summary_df.loc['tau']['hdi_97%'])
            } ## Add sample posterior predictive output to verify my model and check its confidence intervals
            pyMC_summary.append(run_summary)
        #########

        plt.figure()
        plt.hist(CT, bins=70, range=(0, 70), histtype='step', color='blue', label="Data")
        plt.errorbar(xdata, ydata,  yerr=ydata_errors,  color='blue', linestyle='None', alpha=0.7)

        label = (
            fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
            fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
            fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof}$, " + "\n"
            fr"$\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$"
            
        )

        plt.plot(xdata, NeutCapture(xdata, *popt), 'r-', linewidth=2, label=label)
        plt.xlabel(fr"Cluster Time [$\mu s$]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(f"Neutron Capture Time for AmBe 2.0v1 g-ROI (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})")
        #plt.savefig("OutputPlots/NeutronCaptureTime_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        

        '''plt.figure()
        plt.hist2d(PE, CCB, bins=300, cmap='viridis', 
                range=[[-10, 120], [0.1, 1.0]], cmin=1, )
        plt.colorbar(label='Counts')
        plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0 (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        #plt.savefig("OutputPlots/ClusterPE_vs_ChargeBalance_AmBe2.0PE100CBB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()'''

        residuals = (ydata - ydata_expected) / ydata_errors
        plt.figure()
        plt.plot(xdata, residuals)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Time [μs]")
        plt.ylabel("Normalized Residual")
        plt.title(f"Fit Residuals for AmBe 2.0 Neutron Capture Time (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})")
        #plt.savefig("OutputPlots/FitResiduals_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        #plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    Info = pd.DataFrame(time_fit_values, columns=['ThermalTime','ThermalTimeErr', 'CaptureTime','CaptureTimeErr', 'SourcePosition', 'LenEvents', "Chi2Ndof", "p-value", "B"])
    Info['ThermalTime'] = Info['ThermalTime'].astype(float)
    Info['ThermalTimeErr'] = Info['ThermalTimeErr'].astype(float)
    Info['CaptureTime'] = Info['CaptureTime'].astype(float)
    Info['CaptureTimeErr'] = Info['CaptureTimeErr'].astype(float)
    #mask = Info['ThermalTime'] < 50
    #Info = Info[mask]

    print(Info)

    ###########pyMC##########
    pyMC_df = pd.DataFrame(pyMC_summary)
    print(pyMC_df)
    #########################

    #######LMFIT############
    lmfit_df = pd.DataFrame(lmfit_summary)
    print(lmfit_df)
    #######################

# Your port information
    port_info = {
    (0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', (0, 55, 0): 'Port 5',
    (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1',
    (75, 100, 0): 'Port 4', (75, 50, 0): 'Port 4', (75, 0, 0): 'Port 4', (75, -50, 0): 'Port 4', (75, -100, 0): 'Port 4',
    (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3', (0, -50, 102): 'Port 3', (0, -100, 102): 'Port 3', (0, -105, 102): 'Port 3',
    (0, 100, 75): 'Port 2', (0, 50, 75): 'Port 2', (0, 0, 75): 'Port 2', (0, -50, 75): 'Port 2', (0, -100, 75): 'Port 2'}

    port_order = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

    #Info["SourcePosition"] = Info["SourcePosition"].apply(ast.literal_eval)
    Info["Port"] = Info["SourcePosition"].map(port_info)
    Info["Y"] = Info["SourcePosition"].apply(lambda pos: pos[1])

    #Capture time
    pivot_capturetime = Info.pivot(index="Y", columns="Port", values="CaptureTime")
    pivot_capturetimeerr = Info.pivot(index="Y", columns="Port", values="CaptureTimeErr")
    pivot_capturetime = pivot_capturetime.reindex(columns=port_order)
    pivot_capturetimeerr = pivot_capturetimeerr.reindex(columns=port_order)

    #Thermal time
    pivot_thermal_time = Info.pivot(index="Y", columns="Port", values="ThermalTime")
    pivot_thermal_time_err = Info.pivot(index="Y", columns="Port", values="ThermalTimeErr")
    pivot_thermal_time = pivot_thermal_time.reindex(columns=port_order)
    pivot_thermal_time_err = pivot_thermal_time_err.reindex(columns=port_order)

    def make_label_se(eff, err):
        return f"{eff}${{\\pm{err}}}$"

    vectorized_label = np.vectorize(make_label_se)
    labels_SE = vectorized_label(pivot_capturetime.values, pivot_capturetimeerr.values)

    Info['LenEvents'] = Info['LenEvents'].astype(int)
    Info['LenEvents'] = Info['LenEvents'] * (100/Info['LenEvents'].max())
    pivot_lenEvents = Info.pivot(index="Y", columns="Port", values="LenEvents").round(2)
    pivot_lenEvents = pivot_lenEvents.reindex(columns=port_order)

    #print(Info)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_capturetime, annot=labels_SE, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, cbar_kws={"label": "Capture Time (μs)"})
    plt.title("Capture Time of AmBe 2.0v1 g-ROI")
    plt.xlabel("Port")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("OutputPlots/CaptureTime_AmBeNeutrons_AmBe2.0v1gROI.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    vectorized_label2 = np.vectorize(make_label_se)
    labels_SE = vectorized_label2(pivot_thermal_time.values, pivot_thermal_time_err.values)

    #Thermal time
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_thermal_time, annot=labels_SE, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, cbar_kws={"label": "Thermal Time (μs)"})
    plt.title("Thermal Time of AmBe 2.0v1 g-ROI")
    plt.xlabel("Port")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("OutputPlots/ThermalTime_AmBeNeutrons_AmBe2.0v1gROI.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    def weighted_average(df, value, error):
        weights = 1.0 / (df[error] ** 2)
        weighted_avg = ((df[value] * weights).sum()) / (weights.sum())
        weighted_err = np.sqrt(1 / weights.sum())
        return weighted_avg, weighted_err

    NeutCaptureTime, NeutCaptureTimeErr = weighted_average(Info, 'CaptureTime', 'CaptureTimeErr')
    print(f"Weighted Average Capture Time: {NeutCaptureTime:.2f} ± {NeutCaptureTimeErr:.2f} μs")

    NeutThermalTime, NeutThermalTimeErr = weighted_average(Info, 'ThermalTime', 'ThermalTimeErr')
    print(f"Weighted Average Thermal Time: {NeutThermalTime:.2f} ± {NeutThermalTimeErr:.2f} μs")

    PyMCNeutThermalTime, PyMCNeutThermalTimeErr = weighted_average(pyMC_df, 'Therm', 'Thermerr')
    print(f"Weighted Average PyMC Thermal Time: {PyMCNeutThermalTime:.2f} ± {PyMCNeutThermalTimeErr:.2f} μs")

    PyMCNeutCaptureTime, PyMCNeutCaptureTimeErr = weighted_average(pyMC_df, 'Tau', 'Tauerr')
    print(f"Weighted Average PyMC Capture Time: {PyMCNeutCaptureTime:.2f} ± {PyMCNeutCaptureTimeErr:.2f} μs")

    lmfitThermalTime, lmfitThermalTimeErr = weighted_average(lmfit_df, 'Thermal', 'Thermal_err')
    print(f"Weighted Average LMFIT Thermal Time: {lmfitThermalTime:.2f} ± {lmfitThermalTimeErr:.2f} μs")

    lmfitCaptureTime, lmfitCaptureTimeErr = weighted_average(lmfit_df, 'Tau', 'Tau_err')
    print(f"Weighted Average LMFIT Capture Time: {lmfitCaptureTime:.2f} ± {lmfitCaptureTimeErr:.2f} μs")    

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_lenEvents, annot=True, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, cbar_kws={"label": "%"})
    plt.title("Normalized Statistics of all AmBe Neutron-like Events for AmBe 2.0v1 g-ROI")
    plt.xlabel("Port")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("OutputPlots/Statistics_AmBeNeutronEvents_AmBe2.0v1gROI.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()