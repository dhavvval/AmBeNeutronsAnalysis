import pandas as pd
import numpy as np
import re # For robust string parsing
from sklearn.ensemble import IsolationForest # <-- Import IsolationForest
import sys

# --- Configuration ---
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

# ---
# !! PARAMETER TO TUNE !!
#
# 'contamination' is the expected percent of data that is noise.
# 'auto' is a good, modern default.
# You can also set it to a float, e.g., 0.1 (for 10% noise)
iso_contamination = 'auto' 
#
# ---

# --- Step 1: Load Data ---
input_file = 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_test2_4589.csv'
output_file = 'IsolationForest_labels.csv' # New output file
try:
    df = pd.read_csv(input_file)
    print(f"Successfully loaded {input_file}")
except FileNotFoundError:
    print(f"Error: Could not find file {input_file}")
    sys.exit()

# --- Step 2: Define Robust Parsing Function (No changes) ---
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

# --- Step 3: Data "Explosion" (No changes) ---
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
                    'original_cluster_index': index,
                    'original_hit_index': i,
                    'x': hit_x_list[i],
                    'y': hit_y_list[i],
                    'z': hit_z_list[i]
                })
        else:
            problematic_rows.append((index, f"List length mismatch"))
    except Exception as e:
        problematic_rows.append((index, str(e)))

print(f"Data preprocessing complete. Created {len(all_hits_data)} total hit entries.")
hits_df = pd.DataFrame(all_hits_data)
if hits_df.empty:
    print("No valid hit data was processed. Exiting.")
    sys.exit()

# --- Step 4: Apply Isolation Forest (Per Event) ---
print("Starting Isolation Forest anomaly detection per event...")
hits_df['iso_label'] = 0 # New column name, 0 is "not processed"

grouped_events = hits_df.groupby('eventTankTime')
processed_events = 0
total_events = len(grouped_events)
print(f"Found {total_events} unique events to process.")

for event_id, event_hits_df in grouped_events:
    coordinates = event_hits_df[['x', 'y', 'z']].values
    
    # We can run it as long as there's data
    if len(coordinates) > 0:
        
        # --- Run Isolation Forest ---
        # Note: 'random_state' makes the results reproducible
        iso_model = IsolationForest(contamination=iso_contamination, random_state=42)
        labels = iso_model.fit_predict(coordinates)
        # ---
        
        hits_df.loc[event_hits_df.index, 'iso_label'] = labels
    else:
        # Should not happen, but good practice
        hits_df.loc[event_hits_df.index, 'iso_label'] = -1 
    
    processed_events += 1
    if processed_events % 200 == 0 or processed_events == total_events:
        print(f"Processed {processed_events}/{total_events} events...")

print("Isolation Forest detection complete.")

# --- Step 5: Aggregate labels back to original cluster rows ---
print("Aggregating Isolation Forest labels back to original cluster rows...")
hits_df_sorted = hits_df.sort_values(by=['original_cluster_index', 'original_hit_index'])
label_groups = hits_df_sorted.groupby('original_cluster_index')['iso_label'].apply(list)
label_strings = label_groups.apply(str)
df['iso_labels'] = label_strings # New column
df['iso_labels'] = df['iso_labels'].fillna('[]')

# --- Step 6: Save the modified original file ---
df.to_csv(output_file, index=False)
print(f"\nSuccessfully saved modified data to: {output_file}")
print("\n--- Head of new file with 'iso_labels' column ---")
print("REMEMBER: 1 = SIGNAL (Inlier), -1 = NOISE (Outlier)")
print(df[['eventTankTime', 'clusterHits', 'hitX', 'iso_labels']].head())
print("-----------------------------------------------------")
print("\nScript finished.")