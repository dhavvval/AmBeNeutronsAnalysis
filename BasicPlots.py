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
        plt.figure()
        plt.hist(events_counts, bins=range(1, 10, 1), log=True, edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for background events')
        plt.ylabel('Counts')
        plt.title('AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 100, CCB < 0.6)')
        plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.6.png", dpi=300, bbox_inches='tight')
        #plt.show()

        plt.figure()
        plt.hist(events_counts, bins=range(1, 10, 1), edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
        plt.xlabel('Neutron multiplicity for background events')
        plt.ylabel('Counts')
        plt.title('AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 100, CCB < 0.6)')
        plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.6.png", dpi=300, bbox_inches='tight')
        #plt.show()

        # Plot Neutron Capture Time
        counts, bin_edges = np.histogram(CT, bins=200, range=(0, 70))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        init = [np.max(counts), 5, 25, np.min(counts)]


        fit_mask = (bin_centers > 2) & (bin_centers < 65)
        fit_x = bin_centers[fit_mask]
        fit_y = counts[fit_mask]
        fit_y_errors = np.sqrt(fit_y)

        popt, pcov = curve_fit(NeutCapture, fit_x, fit_y, p0=init, sigma=fit_y_errors, absolute_sigma=True, 
                            bounds=([0, 0.1, 0.1, 0], [np.inf, 100, 100, np.inf]))
        perr = np.sqrt(np.diag(pcov))


        fit_y_expected = NeutCapture(fit_x, *popt)
        fit_y_errors[fit_y_errors == 0] = 1e-10
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
            fr"$\chi^2 = {chi2_value:.2f},\ \mathrm{{ndof}} = {ndof},\ "
            fr"\frac{{\chi^2}}{{\mathrm{{ndof}}}} = {chi2_ndof:.2f}$"
        )

        plt.plot(fit_x, NeutCapture(fit_x, *popt), 'r-', linewidth=2, label=label)
        plt.xlabel(fr"Cluster Time [$\mu s$]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(f"Neutron Capture Time for AmBe 2.0 (PE < 100, CCB < 0.6)")
        plt.savefig("OutputPlots/NeutronCaptureTime_AmBe2.0PE100CB0.6.png", dpi=300, bbox_inches='tight')
        #plt.show()

        plt.figure()
        plt.hist2d(PE, CCB, bins=300, cmap='viridis', 
                range=[[-10, 120], [0.1, 1.0]], cmin=1, )
        plt.colorbar(label='Counts')
        plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0 (PE < 100, CCB < 0.6)")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        plt.savefig("OutputPlots/ClusterPE_vs_ChargeBalance_AmBe2.0PE100CBB0.6.png", dpi=300, bbox_inches='tight')
        plt.show()

        residuals = (fit_y - fit_y_expected) / fit_y_errors
        plt.figure()
        plt.plot(fit_x, residuals)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Time [μs]")
        plt.ylabel("Normalized Residual")
        plt.title("Fit Residuals for AmBe 2.0 Neutron Capture Time (PE < 100, CCB < 0.6)")
        plt.savefig("OutputPlots/FitResiduals_AmBe2.0PE100CB0.6.png", dpi=300, bbox_inches='tight')
        #plt.show()


        pdf.savefig()  # Save the current figure to the PDF
        plt.close()  # Close the figure to free memory