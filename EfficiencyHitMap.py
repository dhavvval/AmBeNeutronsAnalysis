import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

efficiency_data = "AmBeTriggerSummary.csv"

df = pd.read_csv(efficiency_data)

port_info = {(0, 100, 0): 'Port 5', (0, 50, 0): 'Port 5', (0, 0, 0): 'Port 5', (0, -50, 0): 'Port 5', (0, -100, 0): 'Port 5', 
 (0, 100, -75): 'Port 1', (0, 50, -75): 'Port 1', (0, 0, -75): 'Port 1', (0, -50, -75): 'Port 1', (0, -100, -75): 'Port 1', 
 (-75, 100, 0): 'Port 4', (-75, 50, 0): 'Port 4', (-75, 0, 0): 'Port 4', (-75, -50, 0): 'Port 4', (-75, -100, 0): 'Port 4', 
 (0, 100, 102): 'Port 3', (0, 50, 102): 'Port 3', (0, 0, 102): 'Port 3', (0, -50, 102): 'Port 3', (0, -100, 102): 'Port 3'}

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

# Label function for SE
def make_label_se(eff, err):
    if pd.isna(eff):
        return ""
    return f"{eff:.2f}${{\\pm{err:.2f}}}$"



vectorized_label = np.vectorize(make_label_se)
labels_SE = vectorized_label(pivot_eff.values, pivot_err.values)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_eff, annot=labels_SE, fmt="", cmap="YlOrBr", cbar=True)
plt.title("AmBe neutron efficiency from AmBe 2.0")
plt.xlabel("Ports")
plt.ylabel("Y Position (cm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

