import pandas as pd
import numpy as np
import re
from sklearn.cluster import OPTICS
import sys
import glob  
import os   

# --- Configuration ---
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

optics_min_samples = 10

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

# --- Main Processing Function (Wraps all steps) ---
def process_file(input_file_path):
    print(f"\n[INFO] Starting processing for: {input_file_path}")

    try:
        df = pd.read_csv(input_file_path)
        print(f"Successfully loaded {input_file_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file_path}. Skipping.")
        return  # Exit this function, move to next file
    except Exception as e:
        print(f"Error loading {input_file_path}: {e}. Skipping.")
        return

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
        print("No valid hit data was processed for this file. Skipping.")
        return

    # --- Step 4: Apply OPTICS (Per Event) ---
    print("Starting OPTICS clustering per event...")
    hits_df['optics_label'] = -2

    grouped_events = hits_df.groupby('eventTankTime')
    processed_events = 0
    total_events = len(grouped_events)
    print(f"Found {total_events} unique events to process.")

    for event_id, event_hits_df in grouped_events:
        coordinates = event_hits_df[['x', 'y', 'z']].values
        
        if len(coordinates) >= optics_min_samples:
            optics_model = OPTICS(min_samples=optics_min_samples)
            labels = optics_model.fit_predict(coordinates)
            hits_df.loc[event_hits_df.index, 'optics_label'] = labels
        else:
            hits_df.loc[event_hits_df.index, 'optics_label'] = -1
        
        processed_events += 1
        if processed_events % 200 == 0 or processed_events == total_events:
            print(f"Processed {processed_events}/{total_events} events...")

    print("OPTICS clustering complete.")

    # --- Step 5: Aggregate labels back to original cluster rows ---
    print("Aggregating OPTICS labels back to original cluster rows...")
    hits_df_sorted = hits_df.sort_values(by=['original_cluster_index', 'original_hit_index'])
    label_groups = hits_df_sorted.groupby('original_cluster_index')['optics_label'].apply(list)
    label_strings = label_groups.apply(str)
    df['optics_labels'] = label_strings
    df['optics_labels'] = df['optics_labels'].fillna('[]')

    # --- Step 6: Save the modified original file ---
    
    # Generate a new output filename
    base_name, ext = os.path.splitext(input_file_path)
    output_file = f"{base_name}_OPTICS{ext}"
    
    df.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Successfully saved modified data to: {output_file}")
    print("\n--- Head of new file with 'optics_labels' column ---")
    print(df[['eventTankTime', 'clusterHits', 'hitX', 'optics_labels']].head())
    print("-----------------------------------------------------")


# --- Main Execution: Find and process all files ---
if __name__ == "__main__":
    input_pattern = 'EventAmBeNeutronCandidatesData/EventAmBeNeutronCandidates_test_*.csv'
    file_list = glob.glob(input_pattern)

    if not file_list:
        print(f"Error: No files found matching pattern: {input_pattern}")
        sys.exit()

    print(f"Found {len(file_list)} files to process:")
    for f in file_list:
        print(f"  - {f}")
    
    # Process each file in the list
    for input_file in file_list:
        process_file(input_file)

    print("\nScript finished processing all files.")

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
