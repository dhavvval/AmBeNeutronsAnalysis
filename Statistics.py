import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

WaveformFile = "TriggerSummary/AmBeWaveformResults_AmBeC1CB0.35.csv"
TriggerFile = "TriggerSummary/AmBeTriggerSummary_AmBeC1CB0.35.csv"

Wdf = pd.read_csv(WaveformFile)
Tdf = pd.read_csv(TriggerFile)


# Merge the two DataFrames on 'x_pos', 'y_pos', and 'z_pos'
WT_df = pd.merge(Wdf, Tdf, on=['x_pos', 'y_pos', 'z_pos'], how='outer', suffixes=('_waveform', '_trigger'))



port_info = {(0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', 
 (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1', 
 (75, 100, 0): 'Port 4', (75, 50, 0): 'Port 4', (75, 0, 0): 'Port 4', (75, -50, 0): 'Port 4', (75, -100, 0): 'Port 4', 
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
WT_df["% AmBe triggers"] = round ((WT_df["ambe_triggers"]/WT_df["total_waveforms"])*100, 2)
WT_df["% Cosmic events"] = round ((WT_df["cosmic_events"]/WT_df["total_waveforms"])*100, 2)
WT_df["% Uniques neutron triggers"] = round ((WT_df["unique_neutron_triggers"]/WT_df["total_waveforms"])*100, 2)
WT_df["% Multiple neutron candidates"] = round ((WT_df["multiple_neutron_candidates"]/WT_df["total_waveforms"])*100, 2)
WT_df["% Single neutron candidates"] = round ((WT_df["single_neutron_candidates"]/WT_df["total_waveforms"])*100, 2)

print(WT_df.head())

WT_stats = WT_df[['port', "position", "total_waveforms", "accepted_events", "ambe_triggers", "% Accepted waveforms", 
                  "% AmBe triggers", "% Cosmic events", "% Uniques neutron triggers", "% Single neutron candidates", "% Multiple neutron candidates"]]

WT_stats = WT_stats.sort_values(by='port')
WT_stats = WT_stats.reset_index(drop=True)
print(WT_stats)





plot_df = WT_stats.copy()
plot_df[["% Accepted waveforms","% AmBe triggers", "% Cosmic events", "% Uniques neutron triggers", "% Single neutron candidates", "% Multiple neutron candidates"]] = \
    plot_df[["% Accepted waveforms", "% AmBe triggers", "% Cosmic events", "% Uniques neutron triggers", "% Single neutron candidates", "% Multiple neutron candidates"]].fillna(0)

# Define colors (Okabe-Ito)
colors = ["#0371B1", "#E69F00", "#009E73", "#F0E442", "#D55E00", "#CC79A7"]  # Blue, Orange, Green, Yellow, Red, Pink

pdf_file = "StatsofAmBeC1CB0.35.pdf"
with PdfPages(pdf_file) as pdf:
    ports = plot_df["port"].unique()
    
    for port in ports:
        port_data = plot_df[plot_df["port"] == port].sort_values("position")
        
        # Prepare data for plotting
        positions = port_data["position"].astype(str)
        accepted = port_data["% Accepted waveforms"]
        ambe = port_data["% AmBe triggers"]
        cosmic = port_data["% Cosmic events"]
        single = port_data["% Single neutron candidates"]
        multiple = port_data["% Multiple neutron candidates"]

        x = np.arange(len(positions))
        width = 0.25 # Bar width

        fig, ax = plt.subplots(figsize=(14, 7))

        # --- Stacked Bar for Event Composition ---
        # 1. Plot Single Neutrons at the bottom
        bars_single = ax.bar(x + 2*width, single, width, label='% Single neutron candidates', color=colors[4])
        # 2. Plot Multiple Neutrons on top of Single Neutrons
        bars_multiple = ax.bar(x + 2*width, multiple, width, bottom=single, label='% Multiple neutron candidates', color=colors[5])
        # 3. Plot Cosmic Events on top of all neutrons
        bars_cosmic = ax.bar(x + 2*width, cosmic, width, bottom=single + multiple, label='% Cosmic events', color=colors[2])

        # --- Context Bars ---
        # Plot total accepted waveforms and AmBe triggers for comparison
        bars_accepted = ax.bar(x, accepted, width, label='% Accepted waveforms', color=colors[0])
        bars_ambe = ax.bar(x + width, ambe, width, label='% AmBe triggers', color=colors[1])
        
        # --- Annotate Bars ---
        def annotate_bars(bars, stacked_offset=None):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.1:  # Only label significant bars
                    y_pos = bar.get_y() + height / 2 if stacked_offset is not None else height
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                                xytext=(0, 0 if stacked_offset is not None else 3),
                                textcoords="offset points",
                                ha='center', va='center' if stacked_offset is not None else 'bottom',
                                fontsize=8, color='white' if stacked_offset is not None else 'black')
        
        annotate_bars(bars_single)
        annotate_bars(bars_multiple, stacked_offset=single)
        annotate_bars(bars_cosmic, stacked_offset=single+multiple)
        annotate_bars(bars_accepted)
        annotate_bars(bars_ambe)

        # --- Customize Axes and Labels ---
        ax.set_xticks(x + width) # Center ticks between the groups
        ax.set_xticklabels(positions, rotation=45, ha="right")
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xlabel(f'Positions for {port}', fontsize=12)
        ax.set_title(f'Event Selection Flow for - {port}', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(plot_df["% Accepted waveforms"].max(), plot_df["% AmBe triggers"].max()) * 1.15) # Dynamic y-limit

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved all plots to {pdf_file}")