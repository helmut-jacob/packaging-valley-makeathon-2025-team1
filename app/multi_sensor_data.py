import psycopg2
import csv
import os
from datetime import datetime
import json

PG_CONFIG = {
    "host":"34.13.178.206",
    "password":"DigiRomMake",
    "user": "readonly",
    "dbname": "makeathon"
}

SQL_STRING_WITH_GAPFILL = """
select
time_bucket_gapfill('500 ms', ts) as tsbucket,
machine_number,
tag,
locf(max(value_dec)) as value
from extruderlift
where machine_number = %s
and tag=%s
and ts between %s and %s 
group by tsbucket, machine_number, tag
order by tsbucket asc
"""


def get_active_windows_for_machine(conn, machine_number, min_duration='60min'):
    """Get all active cycles for a machine"""
    SQL = f"""
select ts as start_ts, next_ts as stop_ts, next_ts-ts as duration, state from (
select ts,
  value_str as state,
  LEAD(value_str, 1) over (order by ts asc) as next_state,
  LEAD(ts, 1) over (order by ts asc) as next_ts
from (
select ts, value_str
from extruderlift
where machine_number = {machine_number}
and tag = 'state'
order by ts asc
) as state_changes
) as state_ranges
where state = 'active'
and next_ts-ts > '{min_duration}'
order by ts asc
"""
    cur = conn.cursor()
    cur.execute(SQL)
    records = cur.fetchall()
    cur.close()
    return records


def get_all_sensor_data(conn, machine_number, tags, timerange):
    """Get data for all sensor tags"""
    cur = conn.cursor()
    
    all_data = {}
    for tag in tags:
        cur.execute(SQL_STRING_WITH_GAPFILL, (machine_number, tag, timerange[0], timerange[1]))
        records = cur.fetchall()
        all_data[tag] = records
    
    cur.close()
    return all_data


def check_data_quality(datasets):
    """Check if data has acceptable quality (no empty values)"""
    min_length = min(len(datasets[tag]) for tag in datasets.keys())
    
    if min_length < 100:
        return False, "Too few datapoints"
    
    # Count missing values
    missing_counts = {tag: 0 for tag in datasets.keys()}
    empty_row_count = 0
    
    for i in range(min_length):
        all_none = True
        for tag in datasets.keys():
            if datasets[tag][i][3] is None:
                missing_counts[tag] += 1
            else:
                all_none = False
        
        if all_none:
            empty_row_count += 1
    
    # Calculate percentages
    missing_percentages = {tag: (count/min_length)*100 for tag, count in missing_counts.items()}
    
    # Reject if ANY sensor has >5% missing data
    max_missing = max(missing_percentages.values())
    if max_missing > 5.0:
        return False, f"High missing data: {max_missing:.1f}%"
    
    return True, missing_percentages


def write_cycle_csv(machine_number, cycle_num, start, end, datasets, output_dir):
    """
    Write cycle to CSV - ONLY complete rows (no empty values)
    Uses forward fill for sparse missing values
    """
    min_length = min(len(datasets[tag]) for tag in datasets.keys())
    
    duration_minutes = (end - start).total_seconds() / 60
    
    filename = f"machine_{machine_number}_cycle_{cycle_num:03d}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Forward fill missing values
    rows = []
    last_valid = {'torque1': None, 'position1': None, 'torque2': None, 'position2': None}
    skipped_rows = 0
    
    for i in range(min_length):
        timestamp = datasets['torque1'][i][0]
        machine_num = datasets['torque1'][i][1]
        
        # Get values with forward fill
        values = {}
        has_any_value = False
        
        for tag in ['torque1', 'position1', 'torque2', 'position2']:
            val = datasets[tag][i][3]
            
            if val is not None:
                last_valid[tag] = val
                has_any_value = True
                values[tag] = val
            elif last_valid[tag] is not None:
                # Use forward fill
                values[tag] = last_valid[tag]
                has_any_value = True
            else:
                # No valid value yet, skip this row
                values[tag] = None
        
        # Only write row if we have valid values for all sensors
        if has_any_value and all(values[tag] is not None for tag in values.keys()):
            rows.append([timestamp, machine_num, values['torque1'], values['position1'], 
                        values['torque2'], values['position2']])
        else:
            skipped_rows += 1
    
    # Write to CSV
    with open(filepath, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["timestamp", "machine_number", "torque1", "position1", "torque2", "position2"])
        writer.writerows(rows)
    
    print(f" {filename}: {len(rows)} complete rows, {duration_minutes:.1f} min")
    if skipped_rows > 0:
        print(f"    (skipped {skipped_rows} incomplete rows)")
    
    return len(rows), duration_minutes


def process_machine(conn, machine_number, output_dir):
    """Process all cycles for a single machine"""
    print(f"\n{'='*70}")
    print(f"Processing Machine {machine_number}")
    print(f"{'='*70}")
    
    windows = get_active_windows_for_machine(conn, machine_number)
    tags = ['torque1', 'position1', 'torque2', 'position2']
    
    print(f"Found {len(windows)} active cycles")
    
    cycle_metadata = []
    exported = 0
    skipped = 0
    
    for cycle_num, window in enumerate(windows, 1):
        print(f"\nCycle {cycle_num}/{len(windows)}:")
        
        # Get sensor data
        datasets = get_all_sensor_data(conn, machine_number, tags, (window[0], window[1]))
        
        # Check data quality
        is_good, quality_info = check_data_quality(datasets)
        
        if not is_good:
            print(f"  Skipping - {quality_info}")
            skipped += 1
            continue
        
        # Write to CSV
        num_rows, duration = write_cycle_csv(machine_number, cycle_num, window[0], window[1], 
                                             datasets, output_dir)
        
        # Save metadata
        cycle_metadata.append({
            'machine_number': machine_number,
            'cycle_number': cycle_num,
            'filename': f"machine_{machine_number}_cycle_{cycle_num:03d}.csv",
            'start_time': str(window[0]),
            'end_time': str(window[1]),
            'duration_minutes': round(duration, 2),
            'num_datapoints': num_rows,
            'quality': 'good'
        })
        
        exported += 1
    
    print(f"\n Machine {machine_number}: Exported {exported}, Skipped {skipped}")
    return cycle_metadata


def save_dataset_manifest(all_metadata, output_dir):
    """Save manifest file with all cycle information"""
    manifest = {
        'total_cycles': len(all_metadata),
        'machines': sorted(list(set(m['machine_number'] for m in all_metadata))),
        'cycles': all_metadata,
        'generation_time': str(datetime.now())
    }
    
    manifest_path = os.path.join(output_dir, 'dataset_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n Dataset manifest saved: {manifest_path}")
    return manifest


def generate_dataset(machines=[1, 2, 3], output_dir='anomaly_detection_data'):
    """Main function to generate complete dataset"""
    
    print(f"{'='*70}")
    print(f"DATASET GENERATION FOR ANOMALY DETECTION")
    print(f"{'='*70}")
    print(f"Machines: {machines}")
    print(f"Output: {output_dir}/")
    print(f"\nQuality criteria:")
    print(f" - Minimum 100 datapoints per cycle")
    print(f" - Maximum 5% missing data per sensor")
    print(f" - Forward fill for sparse missing values")
    print(f" - Only complete rows exported (no empty values)")
    print(f"{'='*70}")
    
    # Connect to database
    conn = psycopg2.connect(**PG_CONFIG)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each machine
    all_metadata = []
    
    for machine_number in machines:
        metadata = process_machine(conn, machine_number, output_dir)
        all_metadata.extend(metadata)
    
    conn.close()
    
    # Save manifest
    manifest = save_dataset_manifest(all_metadata, output_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" DATASET GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  Total cycles exported: {len(all_metadata)}")
    print(f"  Machines processed: {manifest['machines']}")
    print(f"  Files location: {output_dir}/")
    print(f"\nNext step:")
    print(f"  Run: python multivariate_anomaly_detection_ML_pipeline.py")
    
    return manifest


if __name__ == '__main__':
    # Generate dataset for machines 1, 2, 3
    manifest = generate_dataset(machines=[1, 2, 3], output_dir='anomaly_detection_data')