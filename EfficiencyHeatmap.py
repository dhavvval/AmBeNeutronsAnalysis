import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

efficiency_data = "AmBeTriggerSummary.csv"

df = pd.read_csv(efficiency_data)

port_info = {(0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', 
 (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1', 
 (-75, 100, 0): 'Port 4', (-75, 50, 0): 'Port 4', (-75, 0, 0): 'Port 4', (-75, -50, 0): 'Port 4', (-75, -100, 0): 'Port 4', 
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
        }])], ignore_index=True)

df["port"] = df.apply(lambda row: port_info.get((row["x_pos"], row["y_pos"], row["z_pos"]), "Unknown"), axis=1)
df["efficiency"] = df["neutron_candidates"]/df["ambe_triggers"]
df["err_A"] = np.sqrt(df["neutron_candidates"])/df["ambe_triggers"]

df["err_B"] = np.sqrt(df["efficiency"] * (1 - df["efficiency"]) / df["ambe_triggers"])

df["efficiency"] = df["efficiency"]*100  # Convert to percentage
df["err_A"] = df["err_A"]*100  # Convert to percentage
df["err_B"] = df["err_B"]*100  # Convert to percentage

print(df["err_B"])

pivot_eff = df.pivot(index="y_pos", columns="port", values="efficiency")
pivot_err = df.pivot(index="y_pos", columns="port", values="err_B")
pivot_n = df.pivot(index="y_pos", columns="port", values="neutron_candidates")


port_order = ["Port 1", "Port 5", "Port 2", "Port 3", "Port 4"]

# Ensure all ports are present, even if missing — fill with NaNs
pivot_eff = pivot_eff.reindex(columns=port_order)
pivot_err = pivot_err.reindex(columns=port_order)
pivot_n = pivot_n.reindex(columns=port_order)
mask = (pivot_n == 0)

# Label function for SE
def make_label_se(eff, err, n):
    if pd.isna(eff) or n == 0:
        return "empty"
    return f"{eff:.2f}${{\\pm{err:.2f}}}$"



vectorized_label = np.vectorize(make_label_se)
labels_SE = vectorized_label(pivot_eff.values, pivot_err.values, pivot_n.values)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_eff, annot=labels_SE, fmt="", cmap="YlOrBr", cbar=True, annot_kws={"size": 12}, mask=mask, linecolor='black', linewidths=0.2, cbar_kws={"label": "Efficiency (%)"})
plt.title("AmBe neutron efficiency from AmBe 2.0")
plt.xlabel("Ports")
plt.ylabel("Y Position (cm)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
##Residuals plot using AmBe 1.0 and AmBe 2.0 data

Ambe1_ypos = [100, 50, 0, -50, -100]

# Define the table by reading values row-wise (top to bottom)
ambe1_data = {
    "Port 1": [61, 71, 54, 58, 57],
    "Port 5": [71, 69, 69, 66, 65],
    "Port 2": [56, 67, 69, 64, 58],
    "Port 3": [57, 61, 61, 61, 54],
    "Port 4": [68, 67, None, 69, 67]  # One missing (center right white box)
}

ambe1_df = pd.DataFrame(ambe1_data, index=Ambe1_ypos)
ambe1_df.index.name = "Ambe 1.0 Y Position (cm)"
pivot_eff.index.name = "Ambe 2.0 Y Position (cm)"
pivot_eff = pivot_eff.astype(int)
residuals = pivot_eff - ambe1_df
plt.figure(figsize=(8, 6))
sns.heatmap(residuals, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar_kws={'label': 'Residual (AmBe 2.0 - AmBe 1.0)'}, mask=mask, linecolor='black', linewidths=0.2)

plt.title("Residual Efficiency of AmBe 2.0 compare to AmBe 1.0")
plt.xlabel("Ports")
plt.ylabel("Y Position (cm)")
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
