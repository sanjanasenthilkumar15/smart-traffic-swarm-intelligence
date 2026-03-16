"""
generate_dashboard_csv.py
─────────────────────────
Reads baseline_tripinfo.xml  +  adaptive_tripinfo.xml (your ONNX Edge AI run)
and writes comparison_results.csv in the format the dashboard expects.

Run ONCE after each simulation pair:
    python generate_dashboard_csv.py

Then start the dashboard:
    streamlit run app.py
"""

import xml.etree.ElementTree as ET
import os
import sys
from statistics import mean, median, stdev

# ─── FILE PATHS ────────────────────────────────────────────────────────────
BASELINE_XML = "baseline_tripinfo.xml"
ADAPTIVE_XML = "adaptive_tripinfo.xml"   # your Edge-AI / MQTT run
OUTPUT_CSV   = "comparison_results.csv"  # dashboard reads this directly


# ─── PARSE TRIPINFO ─────────────────────────────────────────────────────────
def parse_tripinfo(filename):
    if not os.path.exists(filename):
        print(f"✗  File not found: {filename}")
        return None

    tree = ET.parse(filename)
    root = tree.getroot()

    durations, waits, losses, departs = [], [], [], []

    for t in root.findall('tripinfo'):
        try:
            durations.append(float(t.get('duration',  0)))
            waits.    append(float(t.get('waitingTime', 0)))
            losses.   append(float(t.get('timeLoss',   0)))
            departs.  append(float(t.get('depart',     0)))
        except (ValueError, TypeError):
            continue

    if not durations:
        print(f"✗  No valid tripinfo entries in {filename}")
        return None

    print(f"✓  Parsed {len(durations)} vehicles from {filename}")
    return dict(durations=durations, waits=waits, losses=losses, departs=departs)


# ─── STATS ──────────────────────────────────────────────────────────────────
def stats(data):
    d, w, l = data['durations'], data['waits'], data['losses']
    n = len(d)
    return {
        'vehicle_count':      n,
        'avg_trip_duration':  mean(d),
        'median_trip_duration': median(d),
        'std_duration':       stdev(d) if n > 1 else 0,
        'avg_waiting_time':   mean(w),
        'median_wait_time':   median(w),
        'std_wait_time':      stdev(w) if n > 1 else 0,
        'total_waiting_time': sum(w),
        'avg_time_loss':      mean(l),
        'median_time_loss':   median(l),
        'total_time_loss':    sum(l),
    }


# ─── WRITE CSV ───────────────────────────────────────────────────────────────
def write_csv(b, e, output_file):
    metrics = [
        ("Vehicle Count",           'vehicle_count',        False),
        ("Avg Trip Duration (s)",   'avg_trip_duration',    True),
        ("Median Duration (s)",     'median_trip_duration', True),
        ("Std Dev Duration (s)",    'std_duration',         True),
        ("Avg Wait Time (s)",       'avg_waiting_time',     True),
        ("Median Wait Time (s)",    'median_wait_time',     True),
        ("Std Dev Wait Time (s)",   'std_wait_time',        True),
        ("Total Wait Time (s)",     'total_waiting_time',   True),
        ("Avg Time Loss (s)",       'avg_time_loss',        True),
        ("Median Time Loss (s)",    'median_time_loss',     True),
        ("Total Time Loss (s)",     'total_time_loss',      True),
    ]

    with open(output_file, 'w') as f:
        f.write("Metric,Baseline,Adaptive,Change,Change_Percent\n")
        for name, key, lower_better in metrics:
            bv = b[key]
            ev = e[key]
            change = ev - bv
            # Change_Percent: positive = improvement (Adaptive is better)
            if bv != 0 and lower_better:
                pct = (bv - ev) / bv * 100   # positive when ev < bv
            elif bv != 0:
                pct = (ev - bv) / bv * 100   # positive when ev > bv
            else:
                pct = 0.0
            f.write(f"{name},{bv:.4f},{ev:.4f},{change:.4f},{pct:.4f}\n")

    print(f"✓  CSV written: {output_file}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Generating Dashboard CSV from SUMO TripInfo")
    print("="*60)

    b_data = parse_tripinfo(BASELINE_XML)
    e_data = parse_tripinfo(ADAPTIVE_XML)

    if not b_data or not e_data:
        print("\n✗  Aborted — fix missing files above.")
        sys.exit(1)

    b_stats = stats(b_data)
    e_stats = stats(e_data)

    # Quick terminal preview
    print(f"\n{'─'*60}")
    print(f"  {'METRIC':<30} {'BASELINE':>12} {'EDGE AI':>12} {'IMPROVE':>10}")
    print(f"{'─'*60}")
    for name, key, lb in [
        ("Avg Waiting Time (s)",  'avg_waiting_time',   True),
        ("Avg Time Loss (s)",     'avg_time_loss',       True),
        ("Avg Trip Duration (s)", 'avg_trip_duration',   True),
        ("Total Wait Time (s)",   'total_waiting_time',  True),
    ]:
        bv = b_stats[key]; ev = e_stats[key]
        pct = (bv-ev)/bv*100 if lb and bv else 0
        print(f"  {name:<30} {bv:>12.2f} {ev:>12.2f} {pct:>9.1f}%")
    print(f"{'─'*60}\n")

    write_csv(b_stats, e_stats, OUTPUT_CSV)

    print("\n  Next steps:")
    print(f"  1. Keep {OUTPUT_CSV} next to app.py")
    print("  2. Run:  streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()