import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import re

def NeutCapture(t, A, therm, tau, B):
    return A * (1-np.exp(-t / therm)) * np.exp(-t / tau) + B

if __name__ == "__main__":
    # going through multiple files

    path = './'  # Directory containing the CSV files

    central_port = [4506, 4505, 4499, 4507, 4508] #port 5 with AmBe source
    outside_tank = [4707, 4708]
    no_source_central_port = [4496] #port 5 without source
    port_1=[4589, 4590, 4591, 4593, 4594, 4595, 4596, 4598, 4599, 4600, 4601, 4602]
    port_2=[4453, 4603, 4604, 4605, 4625]
    port_3=[4628, 4629, 4630, 4633, 4635, 4636, 4640, 4646, 4649, 4650, 4651]
    port_4=[4652, 4653, 4654, 4656, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4670, 4672, 4673, 4678, 4679, 4682, 4683, 4685, 4686, 4687]

    csv_files = glob.glob(os.path.join(path, 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_PE100CB0.45updated_*.csv'))



    all_df = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_df.append(df)

    '''all_df = []
    for file in csv_files:
        match = re.search(r'_(\d+)\.csv$', file)
        if match:
            value = int(match.group(1))
            if value in outside_tank:
                print(f"Processing file for Outside Tank: {file}")
                df = pd.read_csv(file)
                all_df.append(df)'''

    combined_df = pd.concat(all_df, ignore_index=True)
        
    events_counts = combined_df['eventID'].value_counts()

    unique_event_count = combined_df['eventID'].nunique()
    #print(f"Total unique events aka neutron triggers: {unique_event_count}")


    # Separate the counts into two groups
    single_events = events_counts[events_counts == 1]
    multi_events = events_counts[events_counts > 1]
    #print(multi_events)


    PE = combined_df['clusterPE']
    CCB = combined_df['clusterChargeBalance']
    CT = combined_df['clusterTime'] / 1000  # Convert to microseconds

    # Plot for single multiplicity events (if any)
    if not single_events.empty:
        plt.figure()
        counts, bins, patches=plt.hist(single_events, bins=range(1, 10, 1), edgecolor='blue', color="lightgreen", linewidth=0.5, align='left')
        text = "\n".join([f"{int(counts[i])}" for i in range(len(counts)) if counts[i] > 0])
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10, va='top')
        plt.xlabel('Multiplicity')
        plt.ylabel('Counts')
        plt.title('Events with multiplicity = 1')
        plt.show()

        plt.figure()
        plt.hist2d(PE, CCB, bins=300, cmap='viridis', 
                range=[[-10, 120], [0.1, 1.0]], cmin=1, )
        plt.colorbar(label='Counts')
        plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0 for single neutron candidate (PE < 100, CCB < 0.45)")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        plt.savefig("OutputPlots/ClusterPE_vs_ChargeBalance_AmBe2.0PE100CB0.45updated.png", dpi=300, bbox_inches='tight')
        plt.show()

    # Plot for multiple multiplicity events (if any)
    if not multi_events.empty:
        plt.figure()
        counts, bins, patches = plt.hist(multi_events, bins=range(1, 10, 1), edgecolor='blue', color="lightblue", linewidth=0.5, align='left')
        text = "\n".join([f"bin{int(bins[i])}: {int(counts[i])}" for i in range(len(counts)) if counts[i] > 0])
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10, va='top')
        plt.xlabel('Multiplicity')
        plt.ylabel('Counts')
        plt.title('Events with multiplicity > 1')
        plt.show()

        plt.figure()
        plt.hist2d(PE, CCB, bins=300, cmap='viridis', 
                range=[[-10, 120], [0.1, 1.0]], cmin=1, )
        plt.colorbar(label='Counts')
        plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0 for multiple neutron candidates(PE < 100, CCB < 0.45)")
        plt.xlabel("Cluster PE")
        plt.ylabel("Cluster Charge Balance")
        plt.savefig("OutputPlots/ClusterPE_vs_ChargeBalance_AmBe2.0PE100CB0.45updated.png", dpi=300, bbox_inches='tight')
        plt.show()




    # neutron multiplicty
    plt.figure()
    plt.hist(events_counts, bins=range(1, 10, 1), edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
    plt.xlabel('Neutron multiplicity for background events')
    plt.ylabel('Counts')
    plt.title('AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 100, CCB < 0.6)')
    plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.6.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.hist(events_counts, bins=range(1, 10, 1), edgecolor='blue', color="lightblue", linewidth=0.5, align='left', density=False)
    plt.xlabel('Neutron multiplicity for background events')
    plt.ylabel('Counts')
    plt.title('AmBe Neutron multiplicity distribution from AmBe 2.0 (PE < 100, CCB < 0.6)')
    plt.savefig("OutputPlots/NeutronMultiplicity_AmBe2.0PE100CB0.6.png", dpi=300, bbox_inches='tight')
    plt.show()

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
    plt.title(f"Neutron Capture Time for AmBe 2.0 (PE < 100, CCB < 0.45)")
    plt.savefig("OutputPlots/NeutronCaptureTime_AmBe2.0PE100CB0.45updated.png", dpi=300, bbox_inches='tight')
    #plt.show()


    plt.figure()
    plt.hist2d(PE, CCB, bins=300, cmap='viridis', 
            range=[[-10, 120], [0.1, 1.0]], cmin=1, )
    plt.colorbar(label='Counts')
    plt.title(f"Cluster PE vs Charge Balance for AmBe 2.0 (PE < 100, CCB < 0.45)")
    plt.xlabel("Cluster PE")
    plt.ylabel("Cluster Charge Balance")
    plt.savefig("OutputPlots/ClusterPE_vs_ChargeBalance_AmBe2.0PE100CBB0.45updated.png", dpi=300, bbox_inches='tight')
    plt.show()

    residuals = (fit_y - fit_y_expected) / fit_y_errors
    plt.figure()
    plt.plot(fit_x, residuals)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time [μs]")
    plt.ylabel("Normalized Residual")
    plt.title("Fit Residuals for AmBe 2.0 Neutron Capture Time (PE < 100, CCB < 0.45)")
    plt.savefig("OutputPlots/FitResiduals_AmBe2.0PE100CB0.45updated.png", dpi=300, bbox_inches='tight')
    #plt.show()



