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


##This one to make plots of different ports and depths for AmBe neutron source positions####

files = './EventAmBeNeutronCandidatesData/' 
csvs = glob.glob(os.path.join(files, 'EventAmBeNeutronCandidates_AmBeC2NewPMT*.csv'))

source_groups = {}
def NeutCapture(t, A, therm, tau, B):
    return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B

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
        if df.empty:
            print(f"No data in file: {filename}")
            continue

        x, y, z = df["sourceX"].iloc[0], df["sourceY"].iloc[0], df["sourceZ"].iloc[0]
        source_key = (round(x, 3), round(y, 3), round(z, 3))
        if source_key not in source_groups:
            source_groups[source_key] = []
        source_groups[source_key].append(df)
 
    time_fit_values = []
    for source_key, df_list in source_groups.items():
        combined_df = pd.concat(df_list, ignore_index=True)
        

        events_counts = combined_df['eventID'].value_counts()
        PE = combined_df['clusterPE']
        CCB = combined_df['clusterChargeBalance']
        CT = combined_df['clusterTime']/1000 # Convert to microseconds

        # Plot
        sx, sy, sz = (int(v) for v in source_key)
        plt.figure()
        plt.hist(events_counts, bins=range(1, 10, 1), log=True, edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for background events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})')
        #plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.hist(events_counts, bins=range(1, 10, 1), edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for background events')
        plt.ylabel('Counts')
        plt.title(f'AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})')
        #plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot Neutron Capture Time
        counts, bin_edges = np.histogram(CT, bins=200, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        init = [np.max(counts), 5, 25, np.min(counts)]
        fit_mask = (bin_centers > 2) & (bin_centers < 65)
        fit_x = bin_centers[fit_mask]
        fit_y = counts[fit_mask]
        fit_y_errors = np.sqrt(fit_y)
        fit_y_errors[fit_y_errors == 0] = 1e-10

        popt, pcov = curve_fit(NeutCapture, fit_x, fit_y, p0=init, sigma=fit_y_errors, absolute_sigma=True, 
                            bounds=([0, 0.1, 0.1, 0], [np.inf, 100, 100, np.inf]))
        perr = np.sqrt(np.diag(pcov))

        time_fit_values.append((f"{popt[1]:.2f}", f"{perr[1]:.2f}", f"{popt[2]:.2f}", f"{perr[2]:.2f}", (sx, sy, sz), len(events_counts)))

        fit_y_expected = NeutCapture(fit_x, *popt)
        
        chi2_value = np.sum(((fit_y - fit_y_expected) ** 2) / (fit_y_errors ** 2))
        ndof = len(fit_y) - len(popt)
        chi2_ndof = chi2_value / ndof
        p_value = 1 - chi2.cdf(chi2_value, ndof)
        print(fr"$\chi^2 = {chi2_value:.2f}$, ndof = {ndof}, $\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$, p-value = {p_value:.3f}")

        plt.figure()
        plt.hist(CT, bins=200, range=(0, 70), histtype='step', color='blue', label="Data")
        label = (
            fr"$\mathrm{{therm}} = {popt[1]:.2f} \pm {perr[1]:.2f}\ \mu s$" + "\n"
            fr"$\tau = {popt[2]:.2f} \pm {perr[2]:.2f}\ \mu s$" + "\n"
            fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof}$" + "\n"
            fr"$\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$, p = {p_value:.3f}"
            
        )

        plt.plot(fit_x, NeutCapture(fit_x, *popt), 'r-', linewidth=2, label=label)
        plt.xlabel(fr"Cluster Time [$\mu s$]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(f"Neutron Capture Time for AmBe 2.0 C1 (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})")
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

        residuals = (fit_y - fit_y_expected) / fit_y_errors
        plt.figure()
        plt.plot(fit_x, residuals)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Time [μs]")
        plt.ylabel("Normalized Residual")
        plt.title(f"Fit Residuals for AmBe 2.0 Neutron Capture Time (PE < 100, CCB < 0.45), run positions:({sx}, {sy}, {sz})")
        #plt.savefig("OutputPlots/FitResiduals_AmBe2.0PE100CB0.45.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    Info = pd.DataFrame(time_fit_values, columns=['ThermalTime','ThermalTimeErr', 'CaptureTime','CaptureTimeErr', 'SourcePosition', 'LenEvents'])
    Info['ThermalTime'] = Info['ThermalTime'].astype(float)
    Info['ThermalTimeErr'] = Info['ThermalTimeErr'].astype(float)
    Info['CaptureTime'] = Info['CaptureTime'].astype(float)
    Info['CaptureTimeErr'] = Info['CaptureTimeErr'].astype(float)

    print(Info)

    # Save the DataFrame to a CSV file


# Your port information
    port_info = {
    (0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', (0, 55, 0): 'Port 5',
    (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1',
    (75, 100, 0): 'Port 4', (75, 50, 0): 'Port 4', (75, 0, 0): 'Port 4', (75, -50, 0): 'Port 4', (75, -100, 0): 'Port 4',
    (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3', (0, -50, 102): 'Port 3', (0, -100, 102): 'Port 3',
    (0, 100, 75): 'Port 2', (0, 50, 75): 'Port 2', (0, 0, 75): 'Port 2', (0, -50, 75): 'Port 2', (0, -100, 75): 'Port 2'}

    port_order = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

    #Info["SourcePosition"] = Info["SourcePosition"].apply(ast.literal_eval)
    Info["Port"] = Info["SourcePosition"].map(port_info)
    Info["Y"] = Info["SourcePosition"].apply(lambda pos: pos[1])

    pivot_capturetime = Info.pivot(index="Y", columns="Port", values="CaptureTime")
    pivot_capturetimeerr = Info.pivot(index="Y", columns="Port", values="CaptureTimeErr")
    pivot_capturetime = pivot_capturetime.reindex(columns=port_order)
    pivot_capturetimeerr = pivot_capturetimeerr.reindex(columns=port_order)

    def make_label_se(eff, err):
        return f"{eff}${{\\pm{err}}}$"

    vectorized_label = np.vectorize(make_label_se)
    labels_SE = vectorized_label(pivot_capturetime.values, pivot_capturetimeerr.values)

    Info['LenEvents'] = Info['LenEvents'].astype(int)
    Info['LenEvents'] = Info['LenEvents'] * (100/17436)
    pivot_lenEvents = Info.pivot(index="Y", columns="Port", values="LenEvents").round(2)
    pivot_lenEvents = pivot_lenEvents.reindex(columns=port_order)

    #print(Info)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_capturetime, annot=labels_SE, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, cbar_kws={"label": "Capture Time (μs)"})
    plt.title("Capture Time of All Statistics of AmBe 2.0")
    plt.xlabel("Port")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.close()


    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_lenEvents, annot=True, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, cbar_kws={"label": "%"})
    plt.title("Normalized Statistics of all AmBe Neutron-like Events")
    plt.xlabel("Port")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.close()
