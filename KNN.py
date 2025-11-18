import pandas as pd
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
MIN_SAMPLES = 10 

# --- Step 1: Load Data ---
input_file = 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_test2_4589.csv'
print(f"Loading {input_file}...")
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find file {input_file}")
    sys.exit()

# --- Step 2: Define Robust Parsing Function ---
def parse_coord_str(coord_str):
    if not isinstance(coord_str, str):
        return []
    number_pattern = re.compile(r"(-?\d+\.?\d*e?[+-]?\d*)")
    items = number_pattern.findall(coord_str)
    coords = []
    for item in items:
        try:
            coords.append(float(item))
        except ValueError:
            continue
    return coords

# --- Step 3: Data "Explosion" (Same as before) ---
all_hits_data = []
print("Starting data preprocessing...")
for index, row in df.iterrows():
    try:
        hit_x_list = parse_coord_str(row['hitX'])
        hit_y_list = parse_coord_str(row['hitY'])
        hit_z_list = parse_coord_str(row['hitZ'])
        
        if len(hit_x_list) == len(hit_y_list) == len(hit_z_list):
            for i in range(len(hit_x_list)):
                all_hits_data.append({
                    'x': hit_x_list[i],
                    'y': hit_y_list[i],
                    'z': hit_z_list[i]
                })
        else:
            continue
    except Exception:
        continue

hits_df = pd.DataFrame(all_hits_data)
if hits_df.empty:
    print("No valid hit data was found. Exiting.")
    sys.exit()

print(f"Processed {len(hits_df)} total hits.")

# --- Step 4: Calculate k-NN Distances ---
print("Calculating k-NN distances...")
coordinates = hits_df[['x', 'y', 'z']].values
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(coordinates)
distances, indices = nbrs.kneighbors(coordinates)
k_distances = np.sort(distances[:, MIN_SAMPLES-1])

# --- Step 5: Plot the Results ---
print("Generating *zoomed-in* k-NN distance plot...")
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title("k-NN Distance 'Elbow' Plot (Zoomed-In)")
plt.xlabel("Points (sorted by distance to k-th neighbor)")
plt.ylabel(f"Distance to {MIN_SAMPLES}-th Neighbor")

# === THIS IS THE NEW "ZOOM" LINE ===
# Change the '1.0' to be smaller or larger if needed
plt.ylim(0, 1.0) 
# ===================================

plt.grid(True)
plot_filename = 'knn_distance_plot_ZOOMED.png'
plt.savefig(plot_filename)

print(f"\n--- Plot Saved! ---")
print(f"A new plot named '{plot_filename}' has been saved.")
print("Open this new image. The 'elbow' should be much clearer.")