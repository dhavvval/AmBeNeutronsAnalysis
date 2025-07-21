import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

WaveformFile = "TriggerSummary/AmBeWaveformResultsPE150CB0.4.csv"
TriggerFile = "TriggerSummary/AmBeTriggerSummaryportPE150CB0.4.csv"

Wdf = pd.read_csv(WaveformFile)
Tdf = pd.read_csv(TriggerFile)


# Merge the two DataFrames on 'x_pos', 'y_pos', and 'z_pos'
WT_df = pd.merge(Wdf, Tdf, on=['x_pos', 'y_pos', 'z_pos'], how='outer', suffixes=('_waveform', '_trigger'))



port_info = {(0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', 
 (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1', 
 (-75, 100, 0): 'Port 4', (-75, 50, 0): 'Port 4', (-75, 0, 0): 'Port 4', (-75, -50, 0): 'Port 4', (-75, -100, 0): 'Port 4', 
 (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3', (0, -50, 102): 'Port 3', (0, -100, 102): 'Port 3',
 (0, 100, 75): 'Port 2', (0, 50, 75): 'Port 2', (0, 0, 75): 'Port 2', (0, -50, 75): 'Port 2', (0, -100, 75): 'Port 2'
 }

positions = set(port_info.keys())

for pos in positions:
    if not ((WT_df["x_pos"] == pos[0]) & (WT_df["y_pos"] == pos[1]) & (WT_df["z_pos"] == pos[2])).any():
        WT_df = pd.concat([WT_df, pd.DataFrame([{
            "x_pos": pos[0],
            "y_pos": pos[1],
            "z_pos": pos[2],
            "neutron_candidates": 0,
            "ambe_triggers": 1,  # Avoid division by zero
        }])], ignore_index=True)


WT_df["port"] = WT_df.apply(lambda row: port_info.get((row["x_pos"], row["y_pos"], row["z_pos"]), "Unknown"), axis=1)
WT_df["position"]= WT_df.apply(lambda row: f"({row['x_pos']}, {row['y_pos']}, {row['z_pos']})", axis=1)
WT_df["% Accepted waveforms"] = round ((WT_df["accepted_events"]/WT_df["total_waveforms"])*100, 2)
WT_df["% Neutron candidates"] = round ((WT_df["neutron_candidates"]/WT_df["total_waveforms"])*100, 2)
WT_df["% AmBe triggers"] = round ((WT_df["ambe_triggers"]/WT_df["total_waveforms"])*100, 2)

print(WT_df.head())

WT_stats = WT_df[['port', "position", "total_waveforms", "accepted_events", "neutron_candidates", "ambe_triggers", "% Accepted waveforms", "% Neutron candidates", "% AmBe triggers"]]

WT_stats = WT_stats.sort_values(by='port')
WT_stats = WT_stats.reset_index(drop=True)





plot_df = WT_stats.copy()
plot_df[["% Accepted waveforms", "% Neutron candidates", "% AmBe triggers"]] = \
    plot_df[["% Accepted waveforms", "% Neutron candidates", "% AmBe triggers"]].fillna(0)

# Define colors (Okabe-Ito)
colors = ["#0371B1", "#E69F00", "#009E73"]  # Blue, Orange, Green

pdf_file = "Statistics_of_AmBe2.0C1.pdf"
with PdfPages(pdf_file) as pdf:
    ports = plot_df["port"].unique()
    
    for port in ports:
        port_data = plot_df[plot_df["port"] == port]
        
        positions = port_data["position"].astype(str)
        accepted = port_data["% Accepted waveforms"]
        neutron = port_data["% Neutron candidates"]
        ambe = port_data["% AmBe triggers"]
        
        x = np.arange(len(positions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Bars
        bars1 = ax.bar(x - width, accepted, width, label='% Accepted waveforms', color=colors[0])
        bars2 = ax.bar(x, ambe, width, label='% AmBe triggers', color=colors[1])
        bars3 = ax.bar(x + width, neutron, width, label='% Neutron candidates', color=colors[2])


        # Annotate bars
        def annotate(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        
        annotate(bars1)
        annotate(bars2)
        annotate(bars3)
        
        # Highlight NaNs in original WT_df by marking positions
        nan_mask = WT_df.loc[port_data.index, "% Accepted waveforms"].isna()
        xtick_labels = [f"{pos}*" if nan else pos for pos, nan in zip(positions, nan_mask)]
        
        # Customize axes
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xlabel('positions for ' + str(port), fontsize=12)
        ax.set_title(f'Statistics for - {port}', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        pdf.savefig(fig)  # Save each figure to PDF
        plt.close(fig)
    
    print(f"Saved all plots to {pdf_file}")
