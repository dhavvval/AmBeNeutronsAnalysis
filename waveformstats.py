import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


path = './'  # Directory containing the CSV files
efficiency_data = glob.glob(os.path.join(path, 'TriggerSummary/AmBeWaveformResults_AmBeC2NewPMTwithnewrun.csv'))


all_df = []
for file in efficiency_data:
    cdf = pd.read_csv(file)
    all_df.append(cdf)

df = pd.concat(all_df, ignore_index=True)

condition = (df['x_pos'] == 0) & (df['y_pos'] == -105.5) & (df['z_pos'] == 102)
if condition.any():
    df.loc[condition, 'y_pos'] = -100

port_info = {(0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', (0, 55.3, 0): 'Port 5',
 (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1',
 (75, 100, 0): 'Port 4', (75, 50, 0): 'Port 4', (75, 0, 0): 'Port 4', (75, -50, 0): 'Port 4', (75, -100, 0): 'Port 4',
 (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3', (0, -50, 102): 'Port 3', (0, -100, 102): 'Port 3',
 (0, 100, 75): 'Port 2', (0, 50, 75): 'Port 2', (0, 0, 75): 'Port 2', (0, -50, 75): 'Port 2', (0, -100, 75): 'Port 2'
 }

positions = set(port_info.keys())

# Add missing rows for any (x, y, z) not already in the DataFrame
for pos in positions:
    if not ((df["x_pos"] == pos[0]) & (df["y_pos"] == pos[1]) & (df["z_pos"] == pos[2])).any():
        df = pd.concat([df, pd.DataFrame([{
            "x_pos": pos[0],
            "y_pos": pos[1],
            "z_pos": pos[2],
            "neutron_candidates": 0,
            "ambe_triggers": 1,  # Avoid division by zero
            "total_events": 1,
        }])], ignore_index=True)

df["SourcePosition"] = list(zip(df["x_pos"], df["y_pos"], df["z_pos"]))
df["port"] = df["SourcePosition"].map(lambda pos: port_info.get(pos, "Unknown"))

#df["port"] = df.apply(lambda row: port_info.get((row["x_pos"], row["y_pos"], row["z_pos"]), "Unknown"), axis=1)
df["good_waveform"] = (df["accepted_events"]/df["total_waveforms"])*100
  



pivot_eff = df.pivot(index="y_pos", columns="port", values="good_waveform").round(2)
print(pivot_eff)
pivot_eff = pivot_eff.dropna(how='all', axis=0)  # Drop columns where all values are NaN


port_order = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

# Ensure all ports are present, even if missing — fill with NaNs
pivot_eff = pivot_eff.reindex(columns=port_order)


plt.figure(figsize=(8, 6))
sns.heatmap(pivot_eff, annot=True, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, linecolor='black', linewidths=0.2, cbar_kws={"label": "Percentage (%)"})
plt.title("Acceptance of AmBe Waveforms - AmBe 2.0v3")
plt.xlabel("Ports")
plt.ylabel("Y Position (cm)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("OutputPlots/Statistics_AmBeWaveformEfficiency_AmBe2.0v3.png", dpi=300, bbox_inches='tight')
plt.show()










