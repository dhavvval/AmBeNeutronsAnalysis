import pandas as pd
import numpy as np
import re # For robust string parsing
from sklearn.cluster import DBSCAN
import sys

# --- Configuration ---
# Set pandas print options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

# ---
# !! CRITICAL PARAMETERS TO TUNE !!
# !!
# eps: The maximum distance between two hits to be considered neighbors.
#      You MUST change this based on the expected scale of your coordinates.
dbscan_eps = 0.5

# min_samples: The minimum number of hits required to form a dense cluster.
dbscan_min_samples = 10
# !!
# ---

# --- Step 1: Load Data ---
input_file = 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_test2_4589.csv'
output_file = 'DBSCAN2.csv' # New file
try:
    df = pd.read_csv(input_file)
    print(f"Successfully loaded {input_file}")
except FileNotFoundError:
    print(f"Error: Could not find file {input_file}")
    sys.exit()

# --- Step 2: Define Robust Parsing Function ---
def parse_coord_str(coord_str):
    """
    Parses coordinate strings like "[0.469, 0.15, ..., -1.05]".
    It uses regex to find all valid numbers and ignores '...'.
    """
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

# --- Step 3: Data "Explosion" ---
all_hits_data = []
problematic_rows = []
print("Starting data preprocessing...")

for index, row in df.iterrows():
    try:
        event_id = row['eventTankTime']
        
        hit_x_list = parse_coord_str(row['hitX'])
        hit_y_list = parse_coord_str(row['hitY'])
        hit_z_list = parse_coord_str(row['hitZ'])
        
        if len(hit_x_list) == len(hit_y_list) == len(hit_z_list):
            if len(hit_x_list) == 0 and row['hitX'] != '[]':
                 problematic_rows.append((index, "Parsed lists are empty"))
                 continue

            for i in range(len(hit_x_list)):
                all_hits_data.append({
                    'eventTankTime': event_id,
                    'original_cluster_index': index, # This is the original row index
                    'original_hit_index': i,      # This is the order of the hit
                    'x': hit_x_list[i],
                    'y': hit_y_list[i],
                    'z': hit_z_list[i]
                })
        else:
            problematic_rows.append((index, f"List length mismatch"))
            
    except Exception as e:
        problematic_rows.append((index, str(e)))

print(f"Data preprocessing complete. Created {len(all_hits_data)} total hit entries.")
if problematic_rows:
    print(f"Encountered {len(problematic_rows)} problematic rows (skipped).")

hits_df = pd.DataFrame(all_hits_data)

if hits_df.empty:
    print("No valid hit data was processed. Exiting.")
    sys.exit()

# --- Step 4: Apply DBSCAN (Per Event) ---
print("Starting DBSCAN clustering per event...")
hits_df['dbscan_label'] = -2 # -2 as a "not-yet-processed" marker

grouped_events = hits_df.groupby('eventTankTime')
processed_events = 0
total_events = len(grouped_events)
print(f"Found {total_events} unique events to process.")

for event_id, event_hits_df in grouped_events:
    coordinates = event_hits_df[['x', 'y', 'z']].values
    
    if len(coordinates) >= dbscan_min_samples:
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = db.fit_predict(coordinates)
        hits_df.loc[event_hits_df.index, 'dbscan_label'] = labels
    else:
        hits_df.loc[event_hits_df.index, 'dbscan_label'] = -1 # Label as noise
    
    processed_events += 1
    if processed_events % 200 == 0 or processed_events == total_events:
        print(f"Processed {processed_events}/{total_events} events...")

print("DBSCAN clustering complete.")

# --- Step 5: Aggregate labels back to original cluster rows ---
print("Aggregating DBSCAN labels back to original cluster rows...")

# Sort the hits_df to ensure labels are aggregated in the correct order
hits_df_sorted = hits_df.sort_values(by=['original_cluster_index', 'original_hit_index'])

# Group by the original row index and collect the labels into a list
label_groups = hits_df_sorted.groupby('original_cluster_index')['dbscan_label'].apply(list)

# Convert the list to a string representation (e.g., "[-1, 0, 1, -1]")
label_strings = label_groups.apply(str)

# Map this new Series back to the original dataframe
# The index of 'label_strings' IS the index of 'df'
df['dbscan_labels'] = label_strings

# Handle original rows that had no valid hits (the ones we skipped)
df['dbscan_labels'] = df['dbscan_labels'].fillna('[]') # Fill NaN with empty list string

# --- Step 6: Save the modified original file ---
df.to_csv(output_file, index=False)
print(f"\nSuccessfully saved modified data to: {output_file}")
print("The original file is unchanged. The new file contains all original columns plus a new 'dbscan_labels' column.")
print("This new column contains a string-list of DBSCAN labels (-1 for noise) corresponding to the hits in hitX, hitY, and hitZ.")

# --- Final Report ---
print("\n--- Head of new file with 'dbscan_labels' column ---")
print(df[['eventTankTime', 'clusterHits', 'hitX', 'dbscan_labels']].head())
print("-----------------------------------------------------")
print("\nScript finished.")

# ---------------------------------------------------------------------------
# ambe CLI integration
# ---------------------------------------------------------------------------
def run(ctx, argv=None):
    """Not yet ported to RunContext API. Run this script directly."""
    raise NotImplementedError(
        f"This module ({__name__}) has not been ported to the ambe CLI yet. "
        "Run it directly as a Python script."
    )


def cli(ctx, argv=None):
    run(ctx, argv)
